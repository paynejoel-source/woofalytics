from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sound_key(value: str) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else "_"
        for char in value.strip()
    )
    return "_".join(part for part in normalized.split("_") if part)


@dataclass(frozen=True, slots=True)
class SoundRule:
    key: str
    label: str
    threshold: float
    target_indices: tuple[int, ...]
    target_labels: dict[int, str]


def _coerce_target_labels(raw: object) -> dict[int, str]:
    if not isinstance(raw, dict):
        return {}
    result: dict[int, str] = {}
    for key, value in raw.items():
        try:
            result[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return result


@dataclass(slots=True)
class AppConfig:
    host: str = os.getenv("WOOF_HOST", "0.0.0.0")
    port: int = int(os.getenv("WOOF_PORT", "8000"))
    model_path: Path = Path(os.getenv("WOOF_MODEL_PATH", "./models/yamnet.tflite"))
    clips_dir: Path = Path(os.getenv("WOOF_CLIPS_DIR", "./clips"))
    events_csv_path: Path = Path(os.getenv("WOOF_EVENTS_CSV_PATH", "./events/events.csv"))
    audio_source: str = os.getenv("WOOF_AUDIO_SOURCE", "ffmpeg")
    # Default to the Frigate substream so bark detection does not compete with
    # Frigate's record path on the primary stream.
    stream_url: str = os.getenv("WOOF_STREAM_URL", "rtsp://127.0.0.1:8554/front_yard_sub")
    ffmpeg_path: str = os.getenv("WOOF_FFMPEG_PATH", "/usr/bin/ffmpeg")
    sample_rate: int = int(os.getenv("WOOF_SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("WOOF_CHANNELS", "1"))
    chunk_samples: int = int(os.getenv("WOOF_CHUNK_SAMPLES", "1600"))
    inference_window_seconds: float = float(
        os.getenv("WOOF_INFERENCE_WINDOW_SECONDS", "0.96")
    )
    inference_hop_seconds: float = float(
        os.getenv("WOOF_INFERENCE_HOP_SECONDS", "0.48")
    )
    clip_pre_seconds: float = float(os.getenv("WOOF_CLIP_PRE_SECONDS", "8"))
    clip_post_seconds: float = float(os.getenv("WOOF_CLIP_POST_SECONDS", "8"))
    clip_retention_days: float = float(os.getenv("WOOF_CLIP_RETENTION_DAYS", "30"))
    bark_threshold: float = float(os.getenv("WOOF_BARK_THRESHOLD", "0.55"))
    thunder_threshold: float = float(os.getenv("WOOF_THUNDER_THRESHOLD", "0.65"))
    train_whistle_threshold: float = float(
        os.getenv("WOOF_TRAIN_WHISTLE_THRESHOLD", "0.60")
    )
    speech_threshold: float = float(os.getenv("WOOF_SPEECH_THRESHOLD", "0.75"))
    trigger_cooldown_seconds: float = float(
        os.getenv("WOOF_TRIGGER_COOLDOWN_SECONDS", "8")
    )
    capture_stall_seconds: float = float(
        os.getenv("WOOF_CAPTURE_STALL_SECONDS", "15")
    )
    device_name_hint: str = os.getenv("WOOF_DEVICE_NAME_HINT", "")
    input_device_index: int | None = (
        int(os.getenv("WOOF_INPUT_DEVICE_INDEX"))
        if os.getenv("WOOF_INPUT_DEVICE_INDEX")
        else None
    )
    ifttt_event_name: str = os.getenv("WOOF_IFTTT_EVENT_NAME", "")
    ifttt_key: str = os.getenv("WOOF_IFTTT_KEY", "")
    save_manual_clips: bool = _bool_env("WOOF_SAVE_MANUAL_CLIPS", True)
    demo_mode: bool = _bool_env("WOOF_DEMO_MODE", False)

    @property
    def bark_class_indices(self) -> tuple[int, ...]:
        # YAMNet AudioSet label indices for bark-adjacent dog vocalizations.
        return (70, 71, 72, 73, 74, 75, 76)

    @property
    def thunder_class_indices(self) -> tuple[int, ...]:
        # YAMNet AudioSet label indices for thunder sounds.
        return (280, 281)

    @property
    def train_whistle_class_indices(self) -> tuple[int, ...]:
        # YAMNet AudioSet label index for train whistle.
        return (324,)

    @property
    def speech_class_indices(self) -> tuple[int, ...]:
        # YAMNet AudioSet label indices for spoken human voice context.
        return (0, 1, 2, 3, 65)

    @property
    def sound_rules(self) -> tuple[SoundRule, ...]:
        base_rules = [
            SoundRule(
                key="bark",
                label="Bark",
                threshold=self.bark_threshold,
                target_indices=self.bark_class_indices,
                target_labels={
                    70: "dog",
                    71: "bark",
                    72: "yip",
                    73: "howl",
                    74: "bow-wow",
                    75: "growling",
                    76: "whimper (dog)",
                },
            ),
            SoundRule(
                key="thunder",
                label="Thunder",
                threshold=self.thunder_threshold,
                target_indices=self.thunder_class_indices,
                target_labels={
                    280: "thunderstorm",
                    281: "thunder",
                },
            ),
            SoundRule(
                key="train_whistle",
                label="Train Whistle",
                threshold=self.train_whistle_threshold,
                target_indices=self.train_whistle_class_indices,
                target_labels={
                    324: "train whistle",
                },
            ),
            SoundRule(
                key="speech",
                label="Speech",
                threshold=self.speech_threshold,
                target_indices=self.speech_class_indices,
                target_labels={
                    0: "speech",
                    1: "child speech",
                    2: "conversation",
                    3: "narration",
                    65: "speech babble",
                },
            ),
        ]
        return tuple([*base_rules, *self._extra_sound_rules()])

    def _extra_sound_rules(self) -> list[SoundRule]:
        raw = os.getenv("WOOF_EXTRA_SOUNDS_JSON", "").strip()
        if not raw:
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(payload, list):
            return []

        extra_rules: list[SoundRule] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            label = str(item.get("label") or item.get("name") or "").strip()
            key = _sound_key(str(item.get("key") or label))
            if not key or not label:
                continue

            raw_indices = item.get("target_indices")
            if not isinstance(raw_indices, list):
                continue

            try:
                target_indices = tuple(int(index) for index in raw_indices)
            except (TypeError, ValueError):
                continue

            if not target_indices:
                continue

            try:
                threshold = float(item.get("threshold", 0.6))
            except (TypeError, ValueError):
                threshold = 0.6

            extra_rules.append(
                SoundRule(
                    key=key,
                    label=label,
                    threshold=threshold,
                    target_indices=target_indices,
                    target_labels=_coerce_target_labels(item.get("target_labels")),
                )
            )

        return extra_rules

    @property
    def inference_window_samples(self) -> int:
        return int(self.sample_rate * self.inference_window_seconds)

    @property
    def inference_hop_samples(self) -> int:
        return int(self.sample_rate * self.inference_hop_seconds)

    @property
    def clip_pre_samples(self) -> int:
        return int(self.sample_rate * self.clip_pre_seconds)

    @property
    def clip_post_samples(self) -> int:
        return int(self.sample_rate * self.clip_post_seconds)
