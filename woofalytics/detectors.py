from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SoundRule

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError as exc:  # pragma: no cover - setup-time path
    Interpreter = None
    TFLITE_IMPORT_ERROR = exc
else:
    TFLITE_IMPORT_ERROR = None


@dataclass(frozen=True, slots=True)
class SoundMatch:
    key: str
    label: str
    score: float
    threshold: float
    target_scores: dict[str, float]

    @property
    def is_active(self) -> bool:
        return self.score >= self.threshold


@dataclass(frozen=True, slots=True)
class SoundInference:
    sounds: tuple[SoundMatch, ...]
    event_type: str | None

    @property
    def is_trigger(self) -> bool:
        return self.event_type is not None

    @property
    def target_scores(self) -> dict[str, float]:
        merged: dict[str, float] = {}
        for sound in self.sounds:
            merged.update(sound.target_scores)
        return merged

    @property
    def active_sounds(self) -> tuple[SoundMatch, ...]:
        active = [sound for sound in self.sounds if sound.is_active]
        sound_by_key = {sound.key: sound for sound in active}
        aircraft = sound_by_key.get("aircraft")
        thunder = sound_by_key.get("thunder")
        if aircraft and thunder and aircraft.score >= thunder.score:
            active = [sound for sound in active if sound.key != "thunder"]
        return tuple(active)

    @property
    def event_score(self) -> float:
        active = self.active_sound
        return active.score if active else 0.0

    @property
    def active_sound(self) -> SoundMatch | None:
        if self.event_type is None:
            return None
        for sound in self.sounds:
            if sound.key == self.event_type:
                return sound
        return None


class YamnetTFLiteBarkDetector:
    def __init__(self, model_path: Path, sound_rules: tuple[SoundRule, ...]):
        if Interpreter is None:
            raise RuntimeError(
                f"LiteRT is not installed: {TFLITE_IMPORT_ERROR}. Install `ai-edge-litert`."
            )
        if not model_path.exists():
            raise RuntimeError(
                f"YAMNet model not found at {model_path}. Download it before starting Woofalytics."
            )
        if not sound_rules:
            raise RuntimeError("At least one sound rule must be configured.")

        self._sound_rules = sound_rules
        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()[0]
        self._input_index = self._input_details["index"]
        self._output_index = self._output_details["index"]
        self._input_length = int(self._input_details["shape"][0])

    def infer(self, samples: np.ndarray) -> SoundInference:
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

        matches: list[SoundMatch] = []
        for rule in self._sound_rules:
            rule_scores = np.max(scores[:, list(rule.target_indices)], axis=0)
            target_scores = {
                rule.target_labels.get(index, f"class_{index}"): float(score)
                for index, score in zip(rule.target_indices, rule_scores, strict=True)
            }
            matches.append(
                SoundMatch(
                    key=rule.key,
                    label=rule.label,
                    score=max(target_scores.values()) if target_scores else 0.0,
                    threshold=rule.threshold,
                    target_scores=target_scores,
                )
            )

        inference = SoundInference(sounds=tuple(matches), event_type=None)
        active_sounds = inference.active_sounds
        event_type = active_sounds[0].key if active_sounds else None
        return SoundInference(sounds=tuple(matches), event_type=event_type)
