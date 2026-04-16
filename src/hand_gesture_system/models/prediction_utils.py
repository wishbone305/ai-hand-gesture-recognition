from __future__ import annotations

import math

import numpy as np

from hand_gesture_system.types import GesturePrediction


def _softmax(vector: np.ndarray) -> np.ndarray:
    stable = vector - np.max(vector)
    exp = np.exp(stable)
    return exp / max(np.sum(exp), 1e-8)


def _label_from_index(index: int, labels: list[str] | None) -> str:
    if labels is not None and 0 <= index < len(labels):
        return labels[index]
    return str(index)


def best_prediction_from_output(
    output: np.ndarray | list[float] | float,
    labels: list[str] | None,
    source: str,
) -> list[GesturePrediction]:
    raw = np.asarray(output)

    if raw.size == 0:
        return []

    if raw.ndim >= 2 and raw.shape[0] == 1:
        raw = raw[0]

    if raw.ndim == 0:
        if np.issubdtype(raw.dtype, np.integer):
            index = int(raw.item())
            return [GesturePrediction(_label_from_index(index, labels), 1.0, source)]

        value = float(raw.item())
        probability = 1.0 / (1.0 + math.exp(-value))
        index = 1 if probability >= 0.5 else 0
        score = probability if index == 1 else (1.0 - probability)
        return [GesturePrediction(_label_from_index(index, labels), float(score), source)]

    if np.issubdtype(raw.dtype, np.integer):
        index = int(raw.reshape(-1)[0])
        return [GesturePrediction(_label_from_index(index, labels), 1.0, source)]

    vector = raw.astype(np.float32).reshape(-1)
    if vector.size == 1:
        probability = 1.0 / (1.0 + math.exp(-float(vector[0])))
        index = 1 if probability >= 0.5 else 0
        score = probability if index == 1 else (1.0 - probability)
        return [GesturePrediction(_label_from_index(index, labels), float(score), source)]

    looks_like_probabilities = bool(np.all(vector >= 0.0) and np.isclose(np.sum(vector), 1.0, atol=1e-3))
    probabilities = vector if looks_like_probabilities else _softmax(vector)

    index = int(np.argmax(probabilities))
    score = float(probabilities[index])
    return [GesturePrediction(_label_from_index(index, labels), score, source)]
