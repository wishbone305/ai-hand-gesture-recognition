from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hand_gesture_system.config import DTWConfig
from hand_gesture_system.models.sequence_base import SequenceGestureModelAdapter
from hand_gesture_system.types import GesturePrediction


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    if seq_a.shape[1] != seq_b.shape[1]:
        raise ValueError(
            f"DTW sequences must share feature dimension, got {seq_a.shape[1]} and {seq_b.shape[1]}"
        )

    len_a, len_b = seq_a.shape[0], seq_b.shape[0]
    dtw = np.full((len_a + 1, len_b + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )

    return float(dtw[len_a, len_b] / max(len_a + len_b, 1))


class DTWGestureAdapter(SequenceGestureModelAdapter):
    """Template-based fallback recognizer for dynamic gestures with tiny datasets."""

    def __init__(
        self,
        templates: dict[str, list[np.ndarray]],
        cfg: DTWConfig | None = None,
    ) -> None:
        self.cfg = cfg or DTWConfig()
        self.templates: dict[str, list[np.ndarray]] = templates

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        cfg: DTWConfig | None = None,
    ) -> "DTWGestureAdapter":
        payload = json.loads(Path(path).read_text())

        templates: dict[str, list[np.ndarray]] = {}
        for label, sequences in payload.items():
            parsed_sequences: list[np.ndarray] = []
            for sequence in sequences:
                array = np.asarray(sequence, dtype=np.float32)
                if array.ndim != 2:
                    raise ValueError(
                        f"Template sequence for label '{label}' must be [frames, features], got {array.shape}"
                    )
                parsed_sequences.append(array)

            if parsed_sequences:
                templates[str(label)] = parsed_sequences

        return cls(templates=templates, cfg=cfg)

    def predict_sequence(self, sequence_features: np.ndarray) -> list[GesturePrediction]:
        sequence = np.asarray(sequence_features, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(
                f"Expected sequence feature tensor with shape [frames, features], got {sequence.shape}"
            )

        if not self.templates:
            return []

        distances: list[tuple[str, float]] = []
        for label, label_templates in self.templates.items():
            best = min(_dtw_distance(sequence, template) for template in label_templates)
            distances.append((label, best))

        ranked = sorted(distances, key=lambda item: item[1])
        if not ranked:
            return []

        best_label, best_distance = ranked[0]
        if best_distance > self.cfg.distance_threshold:
            return []

        predictions: list[GesturePrediction] = []
        for label, distance in ranked[: self.cfg.top_k]:
            score = float(np.exp(-distance * self.cfg.score_scale))
            predictions.append(GesturePrediction(label=label, score=score, source="dtw"))

        return predictions
