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
    thunder_score: float
    bark_threshold: float
    thunder_threshold: float
    bark_target_scores: dict[str, float]
    thunder_target_scores: dict[str, float]
    event_type: str | None

    @property
    def is_bark(self) -> bool:
        return self.event_type == "bark"

    @property
    def is_thunder(self) -> bool:
        return self.event_type == "thunder"

    @property
    def is_trigger(self) -> bool:
        return self.event_type is not None

    @property
    def target_scores(self) -> dict[str, float]:
        return {
            **self.bark_target_scores,
            **self.thunder_target_scores,
        }


class YamnetTFLiteBarkDetector:
    LABELS = {
        70: "dog",
        71: "bark",
        72: "yip",
        73: "howl",
        74: "bow-wow",
        75: "growling",
        76: "whimper (dog)",
        280: "thunderstorm",
        281: "thunder",
    }

    def __init__(
        self,
        model_path: Path,
        bark_threshold: float,
        thunder_threshold: float,
        bark_target_indices: tuple[int, ...],
        thunder_target_indices: tuple[int, ...],
    ):
        if Interpreter is None:
            raise RuntimeError(
                f"LiteRT is not installed: {TFLITE_IMPORT_ERROR}. Install `ai-edge-litert`."
            )
        if not model_path.exists():
            raise RuntimeError(
                f"YAMNet model not found at {model_path}. Download it before starting Woofalytics."
            )

        self._bark_threshold = bark_threshold
        self._thunder_threshold = thunder_threshold
        self._bark_target_indices = bark_target_indices
        self._thunder_target_indices = thunder_target_indices
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

        bark_scores = np.max(scores[:, list(self._bark_target_indices)], axis=0)
        bark_target_scores = {
            self.LABELS.get(index, f"class_{index}"): float(score)
            for index, score in zip(
                self._bark_target_indices, bark_scores, strict=True
            )
        }
        thunder_scores = np.max(scores[:, list(self._thunder_target_indices)], axis=0)
        thunder_target_scores = {
            self.LABELS.get(index, f"class_{index}"): float(score)
            for index, score in zip(
                self._thunder_target_indices, thunder_scores, strict=True
            )
        }
        bark_score = max(bark_target_scores.values()) if bark_target_scores else 0.0
        thunder_score = (
            max(thunder_target_scores.values()) if thunder_target_scores else 0.0
        )

        event_type = None
        if bark_score >= self._bark_threshold or thunder_score >= self._thunder_threshold:
            if bark_score >= thunder_score and bark_score >= self._bark_threshold:
                event_type = "bark"
            elif thunder_score >= self._thunder_threshold:
                event_type = "thunder"

        return BarkInference(
            bark_score=bark_score,
            thunder_score=thunder_score,
            bark_threshold=self._bark_threshold,
            thunder_threshold=self._thunder_threshold,
            bark_target_scores=bark_target_scores,
            thunder_target_scores=thunder_target_scores,
            event_type=event_type,
        )
