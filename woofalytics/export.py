from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import BarkEvent


class EventStore:
    HEADERS = [
        "detected_at",
        "bark_score",
        "clip_path",
        "source",
        "top_label",
        "top_score",
        "target_scores_json",
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

    def replace(self, events: list[BarkEvent]) -> None:
        with self._csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.HEADERS)
            writer.writeheader()
            for event in events:
                top_label = ""
                top_score = ""
                if event.target_scores:
                    top_label, top_score = max(
                        event.target_scores.items(), key=lambda item: item[1]
                    )

                row = {
                    "detected_at": event.detected_at,
                    "bark_score": f"{event.bark_score:.6f}",
                    "clip_path": event.clip_path or "",
                    "source": event.source,
                    "top_label": top_label,
                    "top_score": f"{float(top_score):.6f}" if top_score != "" else "",
                    "target_scores_json": str(event.target_scores),
                }
                writer.writerow(row)
