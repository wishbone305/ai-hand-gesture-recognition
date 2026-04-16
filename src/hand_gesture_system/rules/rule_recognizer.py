from __future__ import annotations

from hand_gesture_system.config import RuleConfig
from hand_gesture_system.features.landmark_utils import (
    THUMB_TIP,
    WRIST,
    finger_count,
    finger_states,
    pinch_distance,
)
from hand_gesture_system.types import GesturePrediction, HandMesh


class RuleBasedRecognizer:
    """Fast fallback recognizer for simple gestures and finger counting."""

    def __init__(self, cfg: RuleConfig | None = None) -> None:
        self.cfg = cfg or RuleConfig()

    def predict(self, mesh: HandMesh) -> list[GesturePrediction]:
        if len(mesh.landmarks) != 21:
            return []

        states = finger_states(mesh, self.cfg)
        count = finger_count(states)
        predictions = [
            GesturePrediction(label=f"fingers_{count}", score=1.0, source="rules")
        ]

        thumb = states["thumb"]
        index = states["index"]
        middle = states["middle"]
        ring = states["ring"]
        pinky = states["pinky"]

        if count == 0:
            predictions.append(GesturePrediction("fist", 0.95, "rules"))

        if count == 5:
            predictions.append(GesturePrediction("open_palm", 0.95, "rules"))

        if index and middle and not ring and not pinky:
            predictions.append(GesturePrediction("peace", 0.9, "rules"))

        if thumb and not index and not middle and not ring and not pinky:
            wrist = mesh.landmarks[WRIST]
            thumb_tip = mesh.landmarks[THUMB_TIP]
            if thumb_tip.y < wrist.y:
                predictions.append(GesturePrediction("thumbs_up", 0.9, "rules"))

        if index and not thumb and not middle and not ring and not pinky:
            predictions.append(GesturePrediction("pointing", 0.85, "rules"))

        if pinch_distance(mesh) < self.cfg.ok_pinch_threshold and middle and ring and pinky:
            predictions.append(GesturePrediction("ok", 0.85, "rules"))

        if index and pinky and not middle and not ring:
            predictions.append(GesturePrediction("rock_sign", 0.75, "rules"))

        return predictions
