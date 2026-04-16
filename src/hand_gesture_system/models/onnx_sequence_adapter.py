from __future__ import annotations

from pathlib import Path

import numpy as np

from hand_gesture_system.models.prediction_utils import best_prediction_from_output
from hand_gesture_system.models.sequence_base import SequenceGestureModelAdapter
from hand_gesture_system.types import GesturePrediction

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


class OnnxSequenceGestureAdapter(SequenceGestureModelAdapter):
    def __init__(
        self,
        model_path: str | Path,
        labels: list[str] | None = None,
        input_layout: str = "btd",
        providers: list[str] | None = None,
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is required for OnnxSequenceGestureAdapter. "
                "Install with: pip install onnxruntime"
            )

        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.labels = labels
        self.input_layout = input_layout

        if self.input_layout not in {"btd", "td", "flat"}:
            raise ValueError(
                f"Unsupported input layout '{self.input_layout}'. "
                "Choose one of: btd, td, flat"
            )

    def _prepare_input(self, sequence_features: np.ndarray) -> np.ndarray:
        sequence = np.asarray(sequence_features, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(
                f"Expected sequence feature tensor with shape [frames, features], got {sequence.shape}"
            )

        if self.input_layout == "btd":
            return sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        if self.input_layout == "td":
            return sequence
        return sequence.reshape(1, -1)

    def predict_sequence(self, sequence_features: np.ndarray) -> list[GesturePrediction]:
        model_input = self._prepare_input(sequence_features)
        outputs = self.session.run(None, {self.input_name: model_input})
        return best_prediction_from_output(outputs[0], self.labels, source="onnx_sequence")
