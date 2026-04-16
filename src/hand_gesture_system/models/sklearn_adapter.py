from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from hand_gesture_system.models.base import GestureModelAdapter
from hand_gesture_system.types import GesturePrediction


class SklearnGestureAdapter(GestureModelAdapter):
    def __init__(self, model_path: str | Path) -> None:
        self.model = joblib.load(model_path)
        classes = getattr(self.model, "classes_", None)
        self.labels = [str(label) for label in classes] if classes is not None else None

    def predict(self, feature_vector: np.ndarray) -> list[GesturePrediction]:
        x = feature_vector.reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(x)[0]
            index = int(np.argmax(probabilities))
            label = (
                self.labels[index]
                if self.labels is not None
                else str(self.model.predict(x)[0])
            )
            score = float(probabilities[index])
            return [GesturePrediction(label=label, score=score, source="sklearn")]

        label = str(self.model.predict(x)[0])
        return [GesturePrediction(label=label, score=0.5, source="sklearn")]
