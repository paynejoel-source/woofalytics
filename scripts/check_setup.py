#!/usr/bin/env python3
import importlib
import sys
from pathlib import Path


REQUIRED_MODULES = ["numpy"]
OPTIONAL_AUDIO_MODULES = ["pyaudio"]
OPTIONAL_RUNTIMES = ["ai_edge_litert.interpreter"]
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yamnet.tflite"


def check_module(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return False, str(exc)
    return True, "imported"


def main() -> int:
    failures = 0

    py_ok = sys.version_info[:2] in {(3, 10), (3, 11), (3, 12), (3, 13)}
    print(f"[{'OK' if py_ok else 'FAIL'}] python: {sys.version.split()[0]}")
    if not py_ok:
        failures += 1
        print("Recommendation: use Python 3.10 through 3.13 for the bark runtime.")

    for module_name in REQUIRED_MODULES:
        ok, detail = check_module(module_name)
        print(f"[{'OK' if ok else 'FAIL'}] {module_name}: {detail}")
        failures += 0 if ok else 1

    for module_name in OPTIONAL_AUDIO_MODULES:
        ok, detail = check_module(module_name)
        print(f"[{'OK' if ok else 'WARN'}] {module_name}: {detail}")

    ffmpeg_ok = Path("/usr/bin/ffmpeg").exists()
    print(f"[{'OK' if ffmpeg_ok else 'FAIL'}] ffmpeg: /usr/bin/ffmpeg")
    if not ffmpeg_ok:
        failures += 1

    runtime_ok = False
    for module_name in OPTIONAL_RUNTIMES:
        ok, detail = check_module(module_name)
        label = "ai-edge-litert"
        print(f"[{'OK' if ok else 'FAIL'}] {label}: {detail}")
        runtime_ok = runtime_ok or ok

    if not runtime_ok:
        failures += 1
        print("Install the LiteRT runtime: `ai-edge-litert`.")

    model_ok = MODEL_PATH.exists()
    print(f"[{'OK' if model_ok else 'FAIL'}] model: {MODEL_PATH}")
    if not model_ok:
        failures += 1
        print("Run `python scripts/download_yamnet.py` after network access is available.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
