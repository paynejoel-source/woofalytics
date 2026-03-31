from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError as exc:  # pragma: no cover - setup-time path
    Interpreter = None
    TFLITE_IMPORT_ERROR = exc
else:
    TFLITE_IMPORT_ERROR = None


@dataclass(slots=True)
class BarkInference:
    bark_score: float
    threshold: float
    target_scores: dict[str, float]

    @property
    def is_bark(self) -> bool:
        return self.bark_score >= self.threshold


class YamnetTFLiteBarkDetector:
    LABELS = {
        70: "dog",
        71: "bark",
        72: "yip",
        73: "howl",
        74: "bow-wow",
        75: "growling",
        76: "whimper (dog)",
    }

    def __init__(
        self,
        model_path: Path,
        bark_threshold: float,
        target_indices: tuple[int, ...],
    ):
        if Interpreter is None:
            raise RuntimeError(
                f"LiteRT is not installed: {TFLITE_IMPORT_ERROR}. Install `ai-edge-litert`."
            )
        if not model_path.exists():
            raise RuntimeError(
                f"YAMNet model not found at {model_path}. Download it before starting Woofalytics."
            )

        self._threshold = bark_threshold
        self._target_indices = target_indices
        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()[0]
        self._input_index = self._input_details["index"]
        self._output_index = self._output_details["index"]
        self._input_length = int(self._input_details["shape"][0])

    def infer(self, samples: np.ndarray) -> BarkInference:
        waveform = samples.astype(np.float32) / np.iinfo(np.int16).max
        if len(waveform) < self._input_length:
            waveform = np.pad(waveform, (0, self._input_length - len(waveform)))
        elif len(waveform) > self._input_length:
            waveform = waveform[: self._input_length]
        self._interpreter.set_tensor(self._input_index, waveform)
        self._interpreter.invoke()

        scores = self._interpreter.get_tensor(self._output_index)
        if scores.ndim == 1:
            scores = scores[np.newaxis, :]
        max_scores = np.max(scores[:, list(self._target_indices)], axis=0)
        target_scores = {
            self.LABELS.get(index, f"class_{index}"): float(score)
            for index, score in zip(self._target_indices, max_scores, strict=True)
        }
        bark_score = max(target_scores.values()) if target_scores else 0.0
        return BarkInference(
            bark_score=bark_score,
            threshold=self._threshold,
            target_scores=target_scores,
        )
