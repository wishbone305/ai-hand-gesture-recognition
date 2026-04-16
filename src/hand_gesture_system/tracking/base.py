from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from hand_gesture_system.types import HandMesh


class HandTracker(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[HandMesh]:
        """Return detected hands as normalized landmark meshes."""

    def close(self) -> None:
        """Release resources if needed."""
        return None
