from __future__ import annotations

import numpy as np

from hand_gesture_system.config import RuleConfig
from hand_gesture_system.features.landmark_utils import (
    finger_states,
    normalized_landmark_vector,
    pinch_distance,
)
from hand_gesture_system.types import HandMesh


class FeatureExtractor:
    """Create a stable feature vector that can be fed to any classifier."""

    def __init__(self, cfg: RuleConfig | None = None) -> None:
        self.cfg = cfg or RuleConfig()

    @staticmethod
    def extract_coordinates(mesh: HandMesh) -> np.ndarray:
        if len(mesh.landmarks) != 21:
            raise ValueError(
                f"Expected 21 landmarks for hand mesh, got {len(mesh.landmarks)}"
            )
        return normalized_landmark_vector(mesh)

    def extract(self, mesh: HandMesh) -> np.ndarray:
        coords = self.extract_coordinates(mesh)
        states = finger_states(mesh, self.cfg)
        state_vector = np.array(
            [
                float(states["thumb"]),
                float(states["index"]),
                float(states["middle"]),
                float(states["ring"]),
                float(states["pinky"]),
            ],
            dtype=np.float32,
        )
        pinch = np.array([pinch_distance(mesh)], dtype=np.float32)
        return np.concatenate([coords, state_vector, pinch]).astype(np.float32)
