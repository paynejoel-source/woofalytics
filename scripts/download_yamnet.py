#!/usr/bin/env python3
from pathlib import Path
from urllib.request import urlopen


MODEL_URL = "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
DESTINATION = Path(__file__).resolve().parent.parent / "models" / "yamnet.tflite"


def main() -> int:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(MODEL_URL, timeout=30) as response:
        DESTINATION.write_bytes(response.read())
    print(f"Downloaded YAMNet model to {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
