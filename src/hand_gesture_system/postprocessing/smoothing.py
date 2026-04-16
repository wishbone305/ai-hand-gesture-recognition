from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from hand_gesture_system.config import ConfusionFixConfig, SmoothingConfig
from hand_gesture_system.features.landmark_utils import (
    INDEX_TIP,
    PINKY_TIP,
    THUMB_IP,
    THUMB_TIP,
)
from hand_gesture_system.types import GesturePrediction, HandMesh


@dataclass
class _SmoothingState:
    stable_key: str | None = None
    stable_label: str | None = None
    candidate_key: str | None = None
    candidate_label: str | None = None
    candidate_count: int = 0
    history: deque[str] | None = None


class PredictionPostprocessor:
    """Apply confusion fixes and temporal debouncing for stable real-time output."""

    def __init__(
        self,
        smoothing_cfg: SmoothingConfig | None = None,
        confusion_cfg: ConfusionFixConfig | None = None,
    ) -> None:
        self.smoothing_cfg = smoothing_cfg or SmoothingConfig()
        self.confusion_cfg = confusion_cfg or ConfusionFixConfig()
        self._states: dict[str, _SmoothingState] = {}

    def drop_hand(self, hand_id: str) -> None:
        self._states.pop(hand_id, None)

    def process(
        self,
        hand_id: str,
        predictions: list[GesturePrediction],
        mesh: HandMesh,
    ) -> list[GesturePrediction]:
        if not predictions:
            return predictions

        ranked = sorted(predictions, key=lambda item: item.score, reverse=True)
        top = ranked[0]
        corrected_label = self._fix_confusions(top.label, mesh)
        if corrected_label != top.label:
            ranked[0] = GesturePrediction(
                label=corrected_label,
                score=top.score,
                source=f"{top.source}+confusion_fix",
            )

        if not self.smoothing_cfg.enabled:
            return ranked

        state = self._states.get(hand_id)
        if state is None:
            state = _SmoothingState(history=deque(maxlen=self.smoothing_cfg.window_size))
            self._states[hand_id] = state
        assert state.history is not None

        current_label = ranked[0].label
        current_key = current_label.lower()
        state.history.append(current_key)

        if state.stable_key is None:
            state.stable_key = current_key
            state.stable_label = current_label
            state.candidate_key = None
            state.candidate_label = None
            state.candidate_count = 0
        elif current_key == state.stable_key:
            state.stable_label = current_label
            state.candidate_key = None
            state.candidate_label = None
            state.candidate_count = 0
        else:
            if state.candidate_key == current_key:
                state.candidate_count += 1
            else:
                state.candidate_key = current_key
                state.candidate_label = current_label
                state.candidate_count = 1

            if state.candidate_count >= self.smoothing_cfg.min_consecutive_frames:
                state.stable_key = current_key
                state.stable_label = current_label
                state.candidate_key = None
                state.candidate_label = None
                state.candidate_count = 0

        stable_key = state.stable_key
        stable_label = state.stable_label or current_label
        vote_ratio = (
            sum(1 for key in state.history if key == stable_key) / max(len(state.history), 1)
        )

        by_key: dict[str, GesturePrediction] = {}
        for prediction in ranked:
            key = prediction.label.lower()
            existing = by_key.get(key)
            if existing is None or prediction.score > existing.score:
                by_key[key] = prediction

        stable_prediction = by_key.get(stable_key or "")
        if stable_prediction is None:
            stable_prediction = GesturePrediction(
                label=stable_label,
                score=max(vote_ratio, self.smoothing_cfg.stable_score),
                source="postprocessor",
            )
        else:
            stable_score = max(
                float(stable_prediction.score),
                float(vote_ratio),
            )
            if stable_key != current_key:
                stable_score = max(stable_score, self.smoothing_cfg.stable_score)
            stable_prediction = GesturePrediction(
                label=stable_prediction.label,
                score=stable_score,
                source=f"{stable_prediction.source}+smoothed",
            )

        output = [stable_prediction]
        for prediction in ranked:
            if prediction.label.lower() == stable_prediction.label.lower():
                continue
            output.append(prediction)

        return output

    def _fix_confusions(self, label: str, mesh: HandMesh) -> str:
        if not self.confusion_cfg.enabled:
            return label
        if len(mesh.landmarks) != 21:
            return label

        normalized = label.upper()
        if normalized in {"A", "T"}:
            replacement = self._resolve_a_t(mesh)
            return replacement if label.isupper() else replacement.lower()

        if normalized in {"D", "I"}:
            replacement = self._resolve_d_i(mesh)
            return replacement if label.isupper() else replacement.lower()

        if normalized in {"F", "W"}:
            replacement = self._resolve_f_w(mesh)
            return replacement if label.isupper() else replacement.lower()

        return label

    @staticmethod
    def _resolve_a_t(mesh: HandMesh) -> str:
        thumb_tip = mesh.landmarks[THUMB_TIP]
        thumb_ip = mesh.landmarks[THUMB_IP]
        index_tip = mesh.landmarks[INDEX_TIP]

        if mesh.handedness == "left":
            return "A" if thumb_tip.x > index_tip.x and thumb_ip.x > index_tip.x else "T"
        return "A" if thumb_tip.x < index_tip.x and thumb_ip.x < index_tip.x else "T"

    @staticmethod
    def _resolve_d_i(mesh: HandMesh) -> str:
        index_tip = mesh.landmarks[INDEX_TIP]
        pinky_tip = mesh.landmarks[PINKY_TIP]
        return "I" if index_tip.y > pinky_tip.y else "D"

    @staticmethod
    def _resolve_f_w(mesh: HandMesh) -> str:
        index_tip = mesh.landmarks[INDEX_TIP]
        pinky_tip = mesh.landmarks[PINKY_TIP]
        return "F" if index_tip.y > pinky_tip.y else "W"
