from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from hand_gesture_system.config import PipelineConfig
from hand_gesture_system.features.extractor import FeatureExtractor
from hand_gesture_system.features.temporal import TemporalFeatureExtractor
from hand_gesture_system.models.base import GestureModelAdapter
from hand_gesture_system.models.sequence_base import SequenceGestureModelAdapter
from hand_gesture_system.postprocessing.smoothing import PredictionPostprocessor
from hand_gesture_system.rules.rule_recognizer import RuleBasedRecognizer
from hand_gesture_system.tracking.base import HandTracker
from hand_gesture_system.types import FrameResult, GesturePrediction, HandMesh, HandResult


@dataclass
class _TrackedHandState:
    hand_id: str
    handedness: str
    last_wrist: np.ndarray
    coordinate_history: deque[np.ndarray] = field(default_factory=deque)
    missed_frames: int = 0


class GesturePipeline:
    def __init__(
        self,
        tracker: HandTracker,
        feature_extractor: FeatureExtractor | None = None,
        rule_engine: RuleBasedRecognizer | None = None,
        model_adapter: GestureModelAdapter | None = None,
        sequence_model_adapter: SequenceGestureModelAdapter | None = None,
        dtw_adapter: SequenceGestureModelAdapter | None = None,
        temporal_extractor: TemporalFeatureExtractor | None = None,
        postprocessor: PredictionPostprocessor | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.tracker = tracker
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.temporal_extractor = temporal_extractor or TemporalFeatureExtractor(
            self.config.temporal
        )

        self.rule_engine = rule_engine
        self.model_adapter = model_adapter
        self.sequence_model_adapter = sequence_model_adapter
        self.dtw_adapter = dtw_adapter
        self.postprocessor = postprocessor or PredictionPostprocessor(
            smoothing_cfg=self.config.smoothing,
            confusion_cfg=self.config.confusion_fix,
        )

        self._next_hand_index = 1
        self._tracked_hands: dict[str, _TrackedHandState] = {}

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        for tracked_hand in self._tracked_hands.values():
            tracked_hand.missed_frames += 1

        hands = self.tracker.detect(frame_bgr)
        hand_assignments = self._assign_hand_ids(hands)

        self._drop_stale_hands()

        results: list[HandResult] = []
        for hand_id, mesh in hand_assignments:
            tracked_hand = self._tracked_hands[hand_id]
            predictions: list[GesturePrediction] = []

            try:
                feature_vector = self.feature_extractor.extract(mesh)
                coordinate_vector = self.feature_extractor.extract_coordinates(mesh)
            except Exception:
                # Skip malformed hand entries but keep pipeline alive.
                continue

            tracked_hand.coordinate_history.append(coordinate_vector)
            history_array = np.stack(tracked_hand.coordinate_history, axis=0)

            sequence_feature_vector: np.ndarray | None = None
            sequence_for_model: np.ndarray | None = None

            if history_array.shape[0] >= 2:
                sequence_feature_vector = self.temporal_extractor.extract(
                    history_array,
                    force_fixed_length=False,
                )

            if (
                history_array.shape[0] >= self.config.temporal.min_sequence_frames
                and self.sequence_model_adapter is not None
            ):
                sequence_for_model = self.temporal_extractor.extract(
                    history_array,
                    force_fixed_length=True,
                )

            if self.rule_engine is not None and self.config.include_rules:
                predictions.extend(self.rule_engine.predict(mesh))

            if self.model_adapter is not None:
                try:
                    predictions.extend(self.model_adapter.predict(feature_vector))
                except Exception:
                    pass

            if sequence_for_model is not None and self.sequence_model_adapter is not None:
                try:
                    predictions.extend(
                        self.sequence_model_adapter.predict_sequence(sequence_for_model)
                    )
                except Exception:
                    pass

            if (
                sequence_feature_vector is not None
                and history_array.shape[0] >= self.config.temporal.min_sequence_frames
                and self.dtw_adapter is not None
            ):
                try:
                    predictions.extend(
                        self.dtw_adapter.predict_sequence(sequence_feature_vector)
                    )
                except Exception:
                    pass

            merged = self._merge_predictions(predictions, self.config.top_k_predictions)
            if self.postprocessor is not None:
                merged = self.postprocessor.process(hand_id, merged, mesh)
                merged = merged[: self.config.top_k_predictions]

            results.append(
                HandResult(
                    hand_id=hand_id,
                    mesh=mesh,
                    predictions=merged,
                    feature_vector=feature_vector,
                    sequence_feature_vector=sequence_feature_vector,
                )
            )

        return FrameResult(hands=results)

    def _assign_hand_ids(self, hands: list[HandMesh]) -> list[tuple[str, HandMesh]]:
        assignments: list[tuple[str, HandMesh]] = []
        used_hand_ids: set[str] = set()

        for mesh in hands:
            if not mesh.landmarks:
                continue
            wrist = self._wrist_vector(mesh)
            best_hand_id: str | None = None
            best_distance = float("inf")

            for hand_id, tracked_hand in self._tracked_hands.items():
                if hand_id in used_hand_ids:
                    continue
                if not self._handedness_compatible(mesh.handedness, tracked_hand.handedness):
                    continue

                distance = float(np.linalg.norm(wrist[:2] - tracked_hand.last_wrist[:2]))
                if distance < best_distance:
                    best_distance = distance
                    best_hand_id = hand_id

            if (
                best_hand_id is not None
                and best_distance <= self.config.hand_assignment_distance_threshold
            ):
                tracked_hand = self._tracked_hands[best_hand_id]
                tracked_hand.last_wrist = wrist
                tracked_hand.missed_frames = 0
                if mesh.handedness != "unknown":
                    tracked_hand.handedness = mesh.handedness

                assignments.append((best_hand_id, mesh))
                used_hand_ids.add(best_hand_id)
                continue

            new_hand_id = f"hand_{self._next_hand_index}"
            self._next_hand_index += 1

            history = deque(maxlen=self.config.temporal.sequence_length)
            self._tracked_hands[new_hand_id] = _TrackedHandState(
                hand_id=new_hand_id,
                handedness=mesh.handedness,
                last_wrist=wrist,
                coordinate_history=history,
                missed_frames=0,
            )

            assignments.append((new_hand_id, mesh))
            used_hand_ids.add(new_hand_id)

        return assignments

    def _drop_stale_hands(self) -> None:
        stale_ids = [
            hand_id
            for hand_id, tracked_hand in self._tracked_hands.items()
            if tracked_hand.missed_frames > self.config.max_missed_frames
        ]

        for hand_id in stale_ids:
            del self._tracked_hands[hand_id]
            if self.postprocessor is not None:
                self.postprocessor.drop_hand(hand_id)

    @staticmethod
    def _wrist_vector(mesh: HandMesh) -> np.ndarray:
        if not mesh.landmarks:
            return np.zeros(3, dtype=np.float32)
        wrist = mesh.landmarks[0]
        return np.array([wrist.x, wrist.y, wrist.z], dtype=np.float32)

    @staticmethod
    def _handedness_compatible(current: str, previous: str) -> bool:
        if current == "unknown" or previous == "unknown":
            return True
        return current == previous

    @staticmethod
    def _merge_predictions(
        predictions: list[GesturePrediction],
        top_k: int,
    ) -> list[GesturePrediction]:
        by_label: dict[str, GesturePrediction] = {}
        for prediction in predictions:
            existing = by_label.get(prediction.label)
            if existing is None or prediction.score > existing.score:
                by_label[prediction.label] = prediction

        ranked = sorted(by_label.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
