from __future__ import annotations

from pathlib import Path

import numpy as np

from hand_gesture_system.models.prediction_utils import best_prediction_from_output
from hand_gesture_system.models.sequence_base import SequenceGestureModelAdapter
from hand_gesture_system.types import GesturePrediction

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class TorchScriptSequenceGestureAdapter(SequenceGestureModelAdapter):
    def __init__(
        self,
        model_path: str | Path,
        labels: list[str] | None = None,
        input_layout: str = "btd",
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "torch is required for TorchScriptSequenceGestureAdapter. "
                "Install with: pip install torch"
            )

        self.model = torch.jit.load(str(model_path), map_location="cpu")
        self.model.eval()
        self.labels = labels
        self.input_layout = input_layout

        if self.input_layout not in {"btd", "td", "flat"}:
            raise ValueError(
                f"Unsupported input layout '{self.input_layout}'. "
                "Choose one of: btd, td, flat"
            )

    def _prepare_input(self, sequence_features: np.ndarray):
        assert torch is not None
        sequence = np.asarray(sequence_features, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(
                f"Expected sequence feature tensor with shape [frames, features], got {sequence.shape}"
            )

        if self.input_layout == "btd":
            return torch.from_numpy(sequence).reshape(1, sequence.shape[0], sequence.shape[1])
        if self.input_layout == "td":
            return torch.from_numpy(sequence)
        return torch.from_numpy(sequence).reshape(1, -1)

    def predict_sequence(self, sequence_features: np.ndarray) -> list[GesturePrediction]:
        assert torch is not None
        model_input = self._prepare_input(sequence_features)
        with torch.no_grad():
            output = self.model(model_input)

        if isinstance(output, tuple):
            output = output[0]
        return best_prediction_from_output(
            output.detach().cpu().numpy(),
            self.labels,
            source="torchscript_sequence",
        )
