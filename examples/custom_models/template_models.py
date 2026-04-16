from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# CUSTOM MODEL FRAMEWORK (IN-CODE DOCUMENTATION)
#
# This file is a ready-to-copy template for integrating your own AI model
# into the hand gesture system.
#
# 1) STATIC MODEL INTEGRATION (single frame -> gesture)
#    Required method signature:
#      predict(self, feature_vector: np.ndarray)
#
#    Input:
#      feature_vector shape ~= [N]
#      where N = 63 coordinate values + 5 finger-state bits + 1 pinch value.
#
#    Output options (any one of these is accepted):
#      - list[GesturePrediction]
#      - dict[str, float]                    (label -> score)
#      - logits/probabilities/class index    (np.ndarray/list/int/float)
#
# 2) SEQUENCE MODEL INTEGRATION (multi-frame -> gesture)
#    Required method signature:
#      predict_sequence(self, sequence_features: np.ndarray)
#
#    Input:
#      sequence_features shape [frames, features]
#
#    Output options:
#      Same accepted output formats as static.
#
# 3) CONSTRUCTOR CONTRACT
#    The custom adapter passes these optional kwargs into your class:
#      model_path: str | None
#      labels: list[str] | None
#      config: dict
#
#    You can accept all/any subset of these arguments.
#
# 4) CLI USAGE
#    Static:
#      PYTHONPATH=src python3 -m hand_gesture_system.cli \
#        --model-type custom \
#        --custom-model-entrypoint examples.custom_models.template_models:MyStaticGestureModel
#
#    Sequence:
#      PYTHONPATH=src python3 -m hand_gesture_system.cli \
#        --dynamic-model-type custom \
#        --custom-dynamic-entrypoint examples.custom_models.template_models:MySequenceGestureModel
#
#    From file path style entrypoint:
#      --custom-model-entrypoint examples/custom_models/template_models.py:MyStaticGestureModel
#
# ---------------------------------------------------------------------------


class MyStaticGestureModel:
    def __init__(
        self,
        model_path: str | None = None,
        labels: list[str] | None = None,
        config: dict | None = None,
    ) -> None:
        self.model_path = model_path
        self.labels = labels or []
        self.config = config or {}

    def predict(self, feature_vector: np.ndarray):
        # Minimal heuristic placeholder.
        # Replace this block with your real model inference code.
        vector = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
        finger_bits = vector[-6:-1] if vector.size >= 6 else np.zeros(5, dtype=np.float32)
        count = int(np.sum(finger_bits > 0.5))

        if count >= 4:
            return {"open_palm": 0.90, f"fingers_{count}": 0.85}
        if count == 0:
            return {"fist": 0.92, "fingers_0": 0.80}
        return {f"fingers_{count}": 0.88}


class MySequenceGestureModel:
    def __init__(
        self,
        model_path: str | None = None,
        labels: list[str] | None = None,
        config: dict | None = None,
    ) -> None:
        self.model_path = model_path
        self.labels = labels or []
        self.config = config or {}
        self.motion_threshold = float(self.config.get("motion_threshold", 0.015))

    def predict_sequence(self, sequence_features: np.ndarray):
        # Minimal motion-based placeholder.
        # Replace with your true temporal model (Transformer/RNN/TCN/etc.).
        sequence = np.asarray(sequence_features, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[0] < 2:
            return {"unknown_dynamic": 0.2}

        motion = float(np.mean(np.abs(np.diff(sequence, axis=0))))
        if motion > self.motion_threshold:
            return {"wave": 0.78, "dynamic_motion": 0.64}
        return {"static_hold": 0.74}
