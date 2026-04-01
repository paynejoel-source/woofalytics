from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlopen


MODEL_URL = "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
SOURCE_TREE_MODEL = Path(__file__).resolve().parent.parent / "models" / "yamnet.tflite"


def model_destination() -> Path:
    configured = os.getenv("WOOF_MODEL_PATH")
    if configured:
        return Path(configured).expanduser()
    if SOURCE_TREE_MODEL.exists():
        return SOURCE_TREE_MODEL
    return Path.cwd() / "models" / "yamnet.tflite"


def main() -> int:
    destination = model_destination()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(MODEL_URL, timeout=30) as response:
        destination.write_bytes(response.read())
    print(f"Downloaded YAMNet model to {destination}")
    return 0


__all__ = ["MODEL_URL", "main", "model_destination"]
