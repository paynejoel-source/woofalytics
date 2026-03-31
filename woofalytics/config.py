from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppConfig:
    host: str = os.getenv("WOOF_HOST", "0.0.0.0")
    port: int = int(os.getenv("WOOF_PORT", "8000"))
    model_path: Path = Path(os.getenv("WOOF_MODEL_PATH", "./models/yamnet.tflite"))
    clips_dir: Path = Path(os.getenv("WOOF_CLIPS_DIR", "./clips"))
    events_csv_path: Path = Path(os.getenv("WOOF_EVENTS_CSV_PATH", "./events/events.csv"))
    audio_source: str = os.getenv("WOOF_AUDIO_SOURCE", "ffmpeg")
    stream_url: str = os.getenv("WOOF_STREAM_URL", "rtsp://127.0.0.1:8554/front_yard")
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
    bark_threshold: float = float(os.getenv("WOOF_BARK_THRESHOLD", "0.55"))
    thunder_threshold: float = float(os.getenv("WOOF_THUNDER_THRESHOLD", "0.55"))
    trigger_cooldown_seconds: float = float(
        os.getenv("WOOF_TRIGGER_COOLDOWN_SECONDS", "8")
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
