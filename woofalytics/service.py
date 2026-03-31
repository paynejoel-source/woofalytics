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
from .detectors import BarkInference, YamnetTFLiteBarkDetector
from .export import EventStore


@dataclass(slots=True)
class BarkEvent:
    detected_at: str
    event_type: str
    bark_score: float
    thunder_score: float
    clip_path: str | None
    target_scores: dict[str, float]
    source: str


@dataclass(slots=True)
class BarkStatus:
    detected_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
    bark_score: float = 0.0
    thunder_score: float = 0.0
    bark_threshold: float = 0.0
    thunder_threshold: float = 0.0
    is_bark: bool = False
    is_thunder: bool = False
    event_type: str | None = None
    target_scores: dict[str, float] = field(default_factory=dict)
    recent_events: list[BarkEvent] = field(default_factory=list)


class TriggerGate:
    def __init__(self, cooldown_seconds: float):
        self._cooldown_seconds = cooldown_seconds
        self._last_trigger_time = 0.0

    def allow(self) -> bool:
        now = time.monotonic()
        if now - self._last_trigger_time < self._cooldown_seconds:
            return False
        self._last_trigger_time = now
        return True


class PendingClip:
    def __init__(self, prefix_chunks: list[np.ndarray], post_samples: int, source: str):
        self.chunks = [chunk.copy() for chunk in prefix_chunks]
        self.remaining_samples = post_samples
        self.source = source

    def append(self, chunk: np.ndarray) -> bool:
        self.chunks.append(chunk.copy())
        self.remaining_samples -= len(chunk)
        return self.remaining_samples <= 0


class BarkMonitor:
    def __init__(self, config: AppConfig):
        self._logger = logging.getLogger("Woofalytics")
        self._config = config
        self._event_store = EventStore(config.events_csv_path)
        self._trigger_gate = TriggerGate(config.trigger_cooldown_seconds)
        self._status = BarkStatus(
            bark_threshold=config.bark_threshold,
            thunder_threshold=config.thunder_threshold,
        )
        self._status_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._rolling_chunks: deque[np.ndarray] = deque()
        self._rolling_sample_count = 0
        self._analysis_buffer = np.array([], dtype=np.int16)
        self._pending_clips: list[PendingClip] = []
        self._demo_tick = 0

        if config.demo_mode:
            self._detector = None
            self._capture = None
        else:
            self._detector = YamnetTFLiteBarkDetector(
                model_path=config.model_path,
                bark_threshold=config.bark_threshold,
                thunder_threshold=config.thunder_threshold,
                bark_target_indices=config.bark_class_indices,
                thunder_target_indices=config.thunder_class_indices,
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
            return {
                "detected_at": self._status.detected_at,
                "event_type": self._status.event_type,
                "bark_score": self._status.bark_score,
                "thunder_score": self._status.thunder_score,
                "bark_threshold": self._status.bark_threshold,
                "thunder_threshold": self._status.thunder_threshold,
                "is_bark": self._status.is_bark,
                "is_thunder": self._status.is_thunder,
                "target_scores": dict(self._status.target_scores),
                "demo_mode": self._config.demo_mode,
                "events_csv_path": str(self._event_store.path),
                "recent_events": [
                    {
                        "detected_at": event.detected_at,
                        "event_type": event.event_type,
                        "bark_score": event.bark_score,
                        "thunder_score": event.thunder_score,
                        "clip_path": event.clip_path,
                        "target_scores": dict(event.target_scores),
                        "source": event.source,
                    }
                    for event in self._status.recent_events
                ],
            }

    def trigger_manual_clip(self) -> dict:
        if not self._config.save_manual_clips:
            return {"ok": False, "message": "Manual clip capture is disabled."}

        self._begin_clip_capture("manual")
        return {"ok": True, "message": "Manual clip capture queued."}

    def _run(self) -> None:
        if self._config.demo_mode:
            self._run_demo_loop()
            return

        self._logger.info(
            "Starting sound monitor at %s Hz with bark %.2f and thunder %.2f",
            self._config.sample_rate,
            self._config.bark_threshold,
            self._config.thunder_threshold,
        )

        while not self._stop_event.is_set():
            try:
                chunk = self._capture.read_chunk()
            except (OSError, RuntimeError):
                if self._stop_event.is_set():
                    break
                raise
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
                if inference.is_trigger and self._trigger_gate.allow():
                    self._begin_clip_capture("auto")
                    self._record_event(inference, None, "auto")
                    self._send_ifttt_event(inference)

    def _run_demo_loop(self) -> None:
        self._logger.info("Starting Woofalytics in demo mode")
        while not self._stop_event.is_set():
            self._demo_tick += 1
            bark_base = (math.sin(self._demo_tick / 4.0) + 1.0) / 2.0
            bark_score = 0.08 + bark_base * 0.85
            bark_score = min(max(bark_score, 0.0), 0.99)
            thunder_base = (math.sin(self._demo_tick / 7.0 + 1.4) + 1.0) / 2.0
            thunder_score = min(max(0.04 + thunder_base * 0.78, 0.0), 0.99)
            event_type = None
            if bark_score >= self._config.bark_threshold or thunder_score >= self._config.thunder_threshold:
                if bark_score >= thunder_score and bark_score >= self._config.bark_threshold:
                    event_type = "bark"
                elif thunder_score >= self._config.thunder_threshold:
                    event_type = "thunder"
            inference = BarkInference(
                bark_score=bark_score,
                thunder_score=thunder_score,
                bark_threshold=self._config.bark_threshold,
                thunder_threshold=self._config.thunder_threshold,
                bark_target_scores={
                    "bark": bark_score,
                    "dog": min(0.99, bark_score * 0.92),
                    "growling": max(0.02, bark_score * 0.48),
                    "howl": max(0.01, bark_score * 0.32),
                },
                thunder_target_scores={
                    "thunder": thunder_score,
                    "thunderstorm": min(0.99, thunder_score * 0.91),
                },
                event_type=event_type,
            )
            self._update_status(inference)
            if inference.is_trigger and self._trigger_gate.allow():
                self._begin_demo_clip_capture("auto")
                self._record_event(inference, None, "auto")
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
            clip_path = self._write_clip(pending.chunks)
            self._record_or_patch_event(clip_path, pending.source)

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

    def _begin_clip_capture(self, source: str) -> None:
        pending = PendingClip(
            prefix_chunks=self._recent_prefix_chunks(),
            post_samples=self._config.clip_post_samples,
            source=source,
        )
        self._pending_clips.append(pending)

    def _begin_demo_clip_capture(self, source: str) -> Path:
        duration = int(self._config.sample_rate * max(self._config.clip_post_seconds, 2))
        timeline = np.arange(duration, dtype=np.float32) / self._config.sample_rate
        waveform = 0.2 * np.sin(2 * np.pi * 440 * timeline)
        envelope = np.where((timeline % 0.45) < 0.08, 1.0, 0.0)
        audio = (waveform * envelope * np.iinfo(np.int16).max).astype(np.int16)
        clip_path = self._write_clip([audio])
        self._record_or_patch_event(clip_path, source)
        return clip_path

    def _write_clip(self, chunks: list[np.ndarray]) -> Path:
        path = self._config.clips_dir / f"{time.time_ns()}.wav"
        merged = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self._config.sample_rate)
            handle.writeframes(merged.astype(np.int16).tobytes())
        self._logger.info("Saved clip %s", path)
        return path

    def _update_status(self, inference: BarkInference) -> None:
        timestamp = dt.datetime.now(dt.UTC).isoformat()
        with self._status_lock:
            self._status.detected_at = timestamp
            self._status.event_type = inference.event_type
            self._status.bark_score = inference.bark_score
            self._status.thunder_score = inference.thunder_score
            self._status.bark_threshold = inference.bark_threshold
            self._status.thunder_threshold = inference.thunder_threshold
            self._status.is_bark = inference.is_bark
            self._status.is_thunder = inference.is_thunder
            self._status.target_scores = dict(inference.target_scores)

    def _record_event(
        self, inference: BarkInference, clip_path: Path | None, source: str
    ) -> None:
        event = BarkEvent(
            detected_at=dt.datetime.now(dt.UTC).isoformat(),
            event_type=inference.event_type or "manual",
            bark_score=inference.bark_score,
            thunder_score=inference.thunder_score,
            clip_path=str(clip_path) if clip_path else None,
            target_scores=dict(inference.target_scores),
            source=source,
        )
        with self._status_lock:
            self._status.recent_events = [event, *self._status.recent_events][:20]
            self._event_store.replace(self._status.recent_events)

    def _record_or_patch_event(self, clip_path: Path, source: str) -> None:
        with self._status_lock:
            for event in self._status.recent_events:
                if event.source == source and event.clip_path is None:
                    event.clip_path = str(clip_path)
                    self._event_store.replace(self._status.recent_events)
                    return

            event = BarkEvent(
                detected_at=dt.datetime.now(dt.UTC).isoformat(),
                event_type=self._status.event_type or "manual",
                bark_score=self._status.bark_score,
                thunder_score=self._status.thunder_score,
                clip_path=str(clip_path),
                target_scores=dict(self._status.target_scores),
                source=source,
            )
            self._status.recent_events = [event, *self._status.recent_events][:20]
            self._event_store.replace(self._status.recent_events)

    def _send_ifttt_event(self, inference: BarkInference) -> None:
        if not self._config.ifttt_event_name or not self._config.ifttt_key:
            return

        body = json.dumps(
            {
                "value1": round(inference.bark_score, 4),
                "value2": round(inference.thunder_score, 4),
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
