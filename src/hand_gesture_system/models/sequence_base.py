from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from hand_gesture_system.types import GesturePrediction


class SequenceGestureModelAdapter(ABC):
    """Interface for temporal gesture models over frame sequences."""

    @abstractmethod
    def predict_sequence(self, sequence_features: np.ndarray) -> list[GesturePrediction]:
        """Predict one or more gestures from [frames, features] temporal input."""
