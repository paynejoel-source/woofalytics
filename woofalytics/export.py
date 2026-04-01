from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import SoundEvent


class EventStore:
    HEADERS = [
        "timestamp",
        "event",
        "event_label",
        "event_confidence",
        "clip_name",
        "source",
        "top_class",
        "top_confidence",
        "class_scores_json",
    ]

    def __init__(self, csv_path: Path):
        self._csv_path = csv_path
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._csv_path.exists():
            with self._csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.HEADERS)
                writer.writeheader()

    @property
    def path(self) -> Path:
        return self._csv_path

    def append(self, event: SoundEvent) -> None:
        with self._csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.HEADERS)
            top_label = ""
            top_score = ""
            if event.target_scores:
                top_label, top_score = max(
                    event.target_scores.items(), key=lambda item: item[1]
                )

            row = {
                "timestamp": event.detected_at,
                "event": event.event_type,
                "event_label": event.event_label,
                "event_confidence": f"{event.event_score:.6f}",
                "clip_name": Path(event.clip_path).name if event.clip_path else "",
                "source": event.source,
                "top_class": top_label,
                "top_confidence": f"{float(top_score):.6f}" if top_score != "" else "",
                "class_scores_json": json.dumps(event.target_scores, sort_keys=True),
            }
            writer.writerow(row)
