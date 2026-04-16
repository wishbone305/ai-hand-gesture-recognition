from __future__ import annotations

from pathlib import Path

import numpy as np

from hand_gesture_system.models.base import GestureModelAdapter
from hand_gesture_system.models.prediction_utils import best_prediction_from_output
from hand_gesture_system.types import GesturePrediction

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


class OnnxGestureAdapter(GestureModelAdapter):
    def __init__(
        self,
        model_path: str | Path,
        labels: list[str] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is required for OnnxGestureAdapter. "
                "Install with: pip install onnxruntime"
            )

        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.labels = labels

    def predict(self, feature_vector: np.ndarray) -> list[GesturePrediction]:
        x = feature_vector.astype(np.float32).reshape(1, -1)
        outputs = self.session.run(None, {self.input_name: x})
        return best_prediction_from_output(outputs[0], self.labels, source="onnx")
