from __future__ import annotations

import datetime as dt
import json
import logging
import math
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import AudioCapture, FFmpegAudioCapture
from .config import AppConfig
from .detectors import SoundInference, SoundMatch, YamnetTFLiteBarkDetector
from .export import EventStore


@dataclass(slots=True)
class SoundEvent:
    detected_at: str
    event_type: str
    event_label: str
    event_score: float
    clip_path: str | None
    target_scores: dict[str, float]
    source: str
    pending_key: str | None = None


@dataclass(slots=True)
class SoundStatus:
    detected_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
    event_type: str | None = None
    event_label: str | None = None
    event_score: float = 0.0
    last_chunk_at: str | None = None
    reconnect_count: int = 0
    last_capture_error: str | None = None
    sounds: tuple[SoundMatch, ...] = field(default_factory=tuple)
    target_scores: dict[str, float] = field(default_factory=dict)
    recent_events: list[SoundEvent] = field(default_factory=list)


class TriggerGate:
    def __init__(self, cooldown_seconds: float):
        self._cooldown_seconds = cooldown_seconds
        self._last_trigger_times: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        last_trigger_time = self._last_trigger_times.get(key, 0.0)
        if now - last_trigger_time < self._cooldown_seconds:
            return False
        self._last_trigger_times[key] = now
        return True


class PendingClip:
    def __init__(
        self,
        prefix_chunks: list[np.ndarray],
        post_samples: int,
        source: str,
        pending_key: str,
        clip_label: str,
        detected_at: dt.datetime,
    ):
        self.chunks = [chunk.copy() for chunk in prefix_chunks]
        self.remaining_samples = post_samples
        self.source = source
        self.pending_key = pending_key
        self.clip_label = clip_label
        self.detected_at = detected_at

    def append(self, chunk: np.ndarray) -> bool:
        self.chunks.append(chunk.copy())
        self.remaining_samples -= len(chunk)
        return self.remaining_samples <= 0


class BarkMonitor:
    def __init__(self, config: AppConfig):
        self._logger = logging.getLogger("Woofalytics")
        self._config = config
        self._sound_rules = config.sound_rules
        self._event_store = EventStore(config.events_csv_path)
        self._trigger_gate = TriggerGate(config.trigger_cooldown_seconds)
        self._status = SoundStatus(sounds=self._blank_sound_matches())
        self._status_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._rolling_chunks: deque[np.ndarray] = deque()
        self._rolling_sample_count = 0
        self._analysis_buffer = np.array([], dtype=np.int16)
        self._pending_clips: list[PendingClip] = []
        self._demo_tick = 0
        self._last_chunk_monotonic: float | None = None

        if config.demo_mode:
            self._detector = None
            self._capture = None
        else:
            self._detector = YamnetTFLiteBarkDetector(
                model_path=config.model_path,
                sound_rules=self._sound_rules,
            )
            if config.audio_source == "ffmpeg":
                self._capture = FFmpegAudioCapture(
                    stream_url=config.stream_url,
                    ffmpeg_path=config.ffmpeg_path,
                    sample_rate=config.sample_rate,
                    channels=config.channels,
                    chunk_samples=config.chunk_samples,
                )
            elif config.audio_source == "pyaudio":
                self._capture = AudioCapture(
                    sample_rate=config.sample_rate,
                    channels=config.channels,
                    chunk_samples=config.chunk_samples,
                    device_name_hint=config.device_name_hint,
                    input_device_index=config.input_device_index,
                )
            else:
                raise RuntimeError(
                    f"Unsupported WOOF_AUDIO_SOURCE={config.audio_source!r}. Use `ffmpeg` or `pyaudio`."
                )

    def start(self) -> None:
        self._config.clips_dir.mkdir(parents=True, exist_ok=True)
        self._prune_old_clips()
        if self._capture is not None:
            self._capture.start()
        self._worker_thread = threading.Thread(target=self._run, name="woof-monitor")
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture is not None:
            self._capture.stop()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)

    def snapshot(self) -> dict:
        with self._status_lock:
            payload = {
                "detected_at": self._status.detected_at,
                "event_type": self._status.event_type,
                "event_label": self._status.event_label,
                "event_score": self._status.event_score,
                "worker_alive": self._worker_thread.is_alive() if self._worker_thread else False,
                "capture_healthy": self._capture_is_healthy(),
                "capture_stall_seconds": self._config.capture_stall_seconds,
                "last_chunk_at": self._status.last_chunk_at,
                "reconnect_count": self._status.reconnect_count,
                "last_capture_error": self._status.last_capture_error,
                "sounds": [self._sound_payload(sound) for sound in self._status.sounds],
                "target_scores": dict(self._status.target_scores),
                "demo_mode": self._config.demo_mode,
                "events_csv_path": str(self._event_store.path),
                "recent_events": [
                    {
                        "detected_at": event.detected_at,
                        "event_type": event.event_type,
                        "event_label": event.event_label,
                        "event_score": event.event_score,
                        "clip_path": event.clip_path,
                        "clip_url": self._clip_url_for_path(event.clip_path),
                        "target_scores": dict(event.target_scores),
                        "source": event.source,
                    }
                    for event in self._status.recent_events
                ],
            }
            payload.update(self._legacy_sound_payload())
            return payload

    def trigger_manual_clip(self) -> dict:
        if not self._config.save_manual_clips:
            return {"ok": False, "message": "Manual clip capture is disabled."}

        pending_key = self._next_pending_key()
        self._begin_clip_capture("manual", pending_key, clip_label="manual")
        self._record_manual_event(pending_key)
        return {"ok": True, "message": "Manual clip capture queued."}

    def _run(self) -> None:
        if self._config.demo_mode:
            self._run_demo_loop()
            return

        thresholds = ", ".join(
            f"{rule.label} {rule.threshold:.2f}" for rule in self._sound_rules
        )
        self._logger.info(
            "Starting sound monitor at %s Hz with %s",
            self._config.sample_rate,
            thresholds,
        )

        while not self._stop_event.is_set():
            try:
                chunk = self._capture.read_chunk()
            except (OSError, RuntimeError) as exc:
                if self._stop_event.is_set():
                    break
                if not self._recover_capture(exc):
                    break
                continue
            self._mark_chunk_received()
            self._append_chunk(chunk)
            self._tick_pending_clips(chunk)
            self._analysis_buffer = np.concatenate((self._analysis_buffer, chunk))

            while len(self._analysis_buffer) >= self._config.inference_window_samples:
                window = self._analysis_buffer[: self._config.inference_window_samples]
                self._analysis_buffer = self._analysis_buffer[
                    self._config.inference_hop_samples :
                ]
                inference = self._detector.infer(window)
                self._update_status(inference)
                triggered_sounds = tuple(
                    sound
                    for sound in inference.active_sounds
                    if self._trigger_gate.allow(sound.key)
                )
                if triggered_sounds:
                    pending_key = self._next_pending_key()
                    self._begin_clip_capture(
                        "auto",
                        pending_key,
                        clip_label=triggered_sounds[0].key,
                    )
                    for sound in triggered_sounds:
                        self._record_event(sound, inference, None, "auto", pending_key)
                    self._send_ifttt_event(inference)

    def _recover_capture(self, exc: Exception) -> bool:
        self._logger.warning("Audio capture failed: %s", exc)
        self._analysis_buffer = np.array([], dtype=np.int16)
        self._rolling_chunks.clear()
        self._rolling_sample_count = 0
        self._pending_clips.clear()
        self._mark_capture_error(str(exc))

        try:
            self._capture.stop()
        except Exception as stop_exc:  # pragma: no cover - defensive cleanup
            self._logger.warning("Audio capture cleanup failed: %s", stop_exc)

        for attempt in range(1, 6):
            if self._stop_event.wait(min(attempt, 3)):
                return False

            try:
                self._capture.start()
            except (OSError, RuntimeError) as restart_exc:
                self._logger.warning(
                    "Audio capture reconnect attempt %s failed: %s",
                    attempt,
                    restart_exc,
                )
                continue

            self._logger.info("Audio capture reconnected on attempt %s", attempt)
            self._mark_capture_reconnected()
            return True

        self._logger.error("Audio capture could not be reconnected after repeated failures")
        return False

    def _run_demo_loop(self) -> None:
        self._logger.info("Starting Woofalytics in demo mode")
        while not self._stop_event.is_set():
            self._demo_tick += 1
            score_by_key = {
                "bark": min(
                    max(0.08 + ((math.sin(self._demo_tick / 4.0) + 1.0) / 2.0) * 0.85, 0.0),
                    0.99,
                ),
                "thunder": min(
                    max(
                        0.04
                        + ((math.sin(self._demo_tick / 7.0 + 1.4) + 1.0) / 2.0) * 0.78,
                        0.0,
                    ),
                    0.99,
                ),
                "train_whistle": min(
                    max(
                        0.03
                        + ((math.sin(self._demo_tick / 9.0 + 0.8) + 1.0) / 2.0) * 0.70,
                        0.0,
                    ),
                    0.99,
                ),
                "speech": min(
                    max(
                        0.06
                        + ((math.sin(self._demo_tick / 5.5 + 2.1) + 1.0) / 2.0) * 0.74,
                        0.0,
                    ),
                    0.99,
                ),
            }

            sounds: list[SoundMatch] = []
            for rule in self._sound_rules:
                score = score_by_key.get(rule.key, 0.03)
                target_scores = {}
                label_values = list(rule.target_labels.values()) or [rule.label.lower()]
                for index, label in enumerate(label_values):
                    target_scores[label] = max(0.0, min(0.99, score * max(0.65, 1.0 - index * 0.1)))
                sounds.append(
                    SoundMatch(
                        key=rule.key,
                        label=rule.label,
                        score=score,
                        threshold=rule.threshold,
                        target_scores=target_scores,
                    )
                )

            event_type = None
            provisional = SoundInference(sounds=tuple(sounds), event_type=None)
            if provisional.active_sounds:
                event_type = provisional.active_sounds[0].key
            inference = SoundInference(sounds=tuple(sounds), event_type=event_type)
            self._mark_chunk_received()
            self._update_status(inference)
            triggered_sounds = tuple(
                sound
                for sound in inference.active_sounds
                if self._trigger_gate.allow(sound.key)
            )
            if triggered_sounds:
                pending_key = self._next_pending_key()
                self._begin_demo_clip_capture(
                    "auto",
                    pending_key,
                    clip_label=triggered_sounds[0].key,
                )
                for sound in triggered_sounds:
                    self._record_event(sound, inference, None, "auto", pending_key)
            time.sleep(max(self._config.inference_hop_seconds, 0.5))

    def _append_chunk(self, chunk: np.ndarray) -> None:
        self._rolling_chunks.append(chunk.copy())
        self._rolling_sample_count += len(chunk)
        max_samples = (
            self._config.clip_pre_samples
            + self._config.clip_post_samples
            + self._config.inference_window_samples
        )
        while self._rolling_sample_count > max_samples and self._rolling_chunks:
            removed = self._rolling_chunks.popleft()
            self._rolling_sample_count -= len(removed)

    def _tick_pending_clips(self, chunk: np.ndarray) -> None:
        if not self._pending_clips:
            return

        finished = []
        for pending in self._pending_clips:
            if pending.append(chunk):
                finished.append(pending)

        for pending in finished:
            self._pending_clips.remove(pending)
            clip_path = self._write_clip(
                pending.chunks,
                clip_label=pending.clip_label,
                detected_at=pending.detected_at,
            )
            self._record_or_patch_event(clip_path, pending.source, pending.pending_key)

    def _recent_prefix_chunks(self) -> list[np.ndarray]:
        chunks = list(self._rolling_chunks)
        if not chunks:
            return []

        total = 0
        selected = []
        for chunk in reversed(chunks):
            selected.append(chunk)
            total += len(chunk)
            if total >= self._config.clip_pre_samples:
                break
        return list(reversed(selected))

    def _begin_clip_capture(
        self,
        source: str,
        pending_key: str,
        clip_label: str,
    ) -> None:
        pending = PendingClip(
            prefix_chunks=self._recent_prefix_chunks(),
            post_samples=self._config.clip_post_samples,
            source=source,
            pending_key=pending_key,
            clip_label=self._safe_clip_label(clip_label),
            detected_at=dt.datetime.now().astimezone(),
        )
        self._pending_clips.append(pending)

    def _begin_demo_clip_capture(
        self,
        source: str,
        pending_key: str,
        clip_label: str,
    ) -> Path:
        duration = int(self._config.sample_rate * max(self._config.clip_post_seconds, 2))
        timeline = np.arange(duration, dtype=np.float32) / self._config.sample_rate
        waveform = 0.2 * np.sin(2 * np.pi * 440 * timeline)
        envelope = np.where((timeline % 0.45) < 0.08, 1.0, 0.0)
        audio = (waveform * envelope * np.iinfo(np.int16).max).astype(np.int16)
        clip_path = self._write_clip(
            [audio],
            clip_label=self._safe_clip_label(clip_label),
            detected_at=dt.datetime.now().astimezone(),
        )
        self._record_or_patch_event(clip_path, source, pending_key)
        return clip_path

    def _write_clip(
        self,
        chunks: list[np.ndarray],
        *,
        clip_label: str,
        detected_at: dt.datetime,
    ) -> Path:
        timestamp = detected_at.astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        path = self._config.clips_dir / (
            f"{timestamp}_{clip_label}_{time.time_ns()}.wav"
        )
        merged = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self._config.sample_rate)
            handle.writeframes(merged.astype(np.int16).tobytes())
        self._logger.info("Saved clip %s", path)
        self._prune_old_clips()
        return path

    def _prune_old_clips(self) -> None:
        retention_days = self._config.clip_retention_days
        if retention_days <= 0:
            return

        cutoff = time.time() - (retention_days * 86400)
        deleted = 0
        for clip_path in self._config.clips_dir.glob("*.wav"):
            try:
                if clip_path.stat().st_mtime < cutoff:
                    clip_path.unlink()
                    deleted += 1
            except FileNotFoundError:
                continue
            except OSError as exc:
                self._logger.warning("Failed to prune clip %s: %s", clip_path, exc)

        if deleted:
            self._logger.info(
                "Pruned %s retained clip(s) older than %.1f day(s)",
                deleted,
                retention_days,
            )

    def _update_status(self, inference: SoundInference) -> None:
        timestamp = dt.datetime.now(dt.UTC).isoformat()
        active_sound = inference.active_sound
        with self._status_lock:
            self._status.detected_at = timestamp
            self._status.event_type = inference.event_type
            self._status.event_label = active_sound.label if active_sound else None
            self._status.event_score = active_sound.score if active_sound else 0.0
            self._status.sounds = inference.sounds
            self._status.target_scores = dict(inference.target_scores)

    def _record_event(
        self,
        sound: SoundMatch,
        inference: SoundInference,
        clip_path: Path | None,
        source: str,
        pending_key: str | None,
    ) -> None:
        event = SoundEvent(
            detected_at=dt.datetime.now(dt.UTC).isoformat(),
            event_type=sound.key,
            event_label=sound.label,
            event_score=sound.score,
            clip_path=str(clip_path) if clip_path else None,
            target_scores=dict(sound.target_scores),
            source=source,
            pending_key=pending_key,
        )
        with self._status_lock:
            self._status.recent_events = [event, *self._status.recent_events][:20]

    def _record_manual_event(self, pending_key: str) -> None:
        event = SoundEvent(
            detected_at=dt.datetime.now(dt.UTC).isoformat(),
            event_type="manual",
            event_label="Manual",
            event_score=0.0,
            clip_path=None,
            target_scores={},
            source="manual",
            pending_key=pending_key,
        )
        with self._status_lock:
            self._status.recent_events = [event, *self._status.recent_events][:20]

    def _record_or_patch_event(self, clip_path: Path, source: str, pending_key: str) -> None:
        with self._status_lock:
            matched_events: list[SoundEvent] = []
            for event in self._status.recent_events:
                if (
                    event.source == source
                    and event.pending_key == pending_key
                    and event.clip_path is None
                ):
                    event.clip_path = str(clip_path)
                    matched_events.append(event)

            if matched_events:
                for event in matched_events:
                    self._event_store.append(event)
                return

            event = SoundEvent(
                detected_at=dt.datetime.now(dt.UTC).isoformat(),
                event_type=self._status.event_type or "manual",
                event_label=self._status.event_label or "Manual",
                event_score=self._status.event_score,
                clip_path=str(clip_path),
                target_scores=dict(self._status.target_scores),
                source=source,
                pending_key=pending_key,
            )
            self._status.recent_events = [event, *self._status.recent_events][:20]
            self._event_store.append(event)

    def _send_ifttt_event(self, inference: SoundInference) -> None:
        if not self._config.ifttt_event_name or not self._config.ifttt_key:
            return

        body = json.dumps(
            {
                "value1": round(self._score_for_key(inference, "bark"), 4),
                "value2": round(self._score_for_key(inference, "thunder"), 4),
                "value3": inference.event_type or "",
            }
        ).encode("utf-8")
        url = (
            "https://maker.ifttt.com/trigger/"
            f"{self._config.ifttt_event_name}/json/with/key/{self._config.ifttt_key}"
        )
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5):
                pass
        except urllib.error.URLError as exc:
            self._logger.warning("IFTTT trigger failed: %s", exc)

    def _blank_sound_matches(self) -> tuple[SoundMatch, ...]:
        return tuple(
            SoundMatch(
                key=rule.key,
                label=rule.label,
                score=0.0,
                threshold=rule.threshold,
                target_scores={},
            )
            for rule in self._sound_rules
        )

    def _legacy_sound_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key in ("bark", "thunder", "train_whistle", "aircraft", "speech"):
            sound = self._sound_for_key(self._status.sounds, key)
            payload[f"{key}_score"] = sound.score if sound else 0.0
            payload[f"{key}_threshold"] = sound.threshold if sound else 0.0
            payload[f"is_{key}"] = sound.is_active if sound else False
        return payload

    def _sound_payload(self, sound: SoundMatch) -> dict[str, object]:
        return {
            "key": sound.key,
            "label": sound.label,
            "score": sound.score,
            "threshold": sound.threshold,
            "is_active": sound.is_active,
            "target_scores": dict(sound.target_scores),
        }

    def _sound_for_key(
        self, sounds: tuple[SoundMatch, ...], key: str
    ) -> SoundMatch | None:
        for sound in sounds:
            if sound.key == key:
                return sound
        return None

    def _score_for_key(self, inference: SoundInference, key: str) -> float:
        sound = self._sound_for_key(inference.sounds, key)
        return sound.score if sound else 0.0

    def _capture_is_healthy(self) -> bool:
        if self._config.demo_mode:
            return True
        if self._last_chunk_monotonic is None:
            return False
        return (time.monotonic() - self._last_chunk_monotonic) <= self._config.capture_stall_seconds

    def _mark_chunk_received(self) -> None:
        timestamp = dt.datetime.now(dt.UTC).isoformat()
        self._last_chunk_monotonic = time.monotonic()
        with self._status_lock:
            self._status.last_chunk_at = timestamp
            self._status.last_capture_error = None

    def _mark_capture_error(self, message: str) -> None:
        with self._status_lock:
            self._status.last_capture_error = message

    def _mark_capture_reconnected(self) -> None:
        with self._status_lock:
            self._status.reconnect_count += 1
            self._status.last_capture_error = None

    def _clip_url_for_path(self, clip_path: str | None) -> str | None:
        if not clip_path:
            return None
        return f"/clips/{Path(clip_path).name}"

    def _next_pending_key(self) -> str:
        return str(time.time_ns())

    def _safe_clip_label(self, value: str) -> str:
        normalized = "".join(
            char.lower() if char.isalnum() else "_"
            for char in value.strip()
        ).strip("_")
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized or "event"
