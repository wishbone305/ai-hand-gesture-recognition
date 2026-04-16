from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from hand_gesture_system.types import GesturePrediction


class GestureModelAdapter(ABC):
    """Interface for plugging in any model backend."""

    @abstractmethod
    def predict(self, feature_vector: np.ndarray) -> list[GesturePrediction]:
        """Predict one or more gestures for one hand feature vector."""

    def predict_batch(self, feature_vectors: np.ndarray) -> list[list[GesturePrediction]]:
        return [self.predict(vector) for vector in feature_vectors]
