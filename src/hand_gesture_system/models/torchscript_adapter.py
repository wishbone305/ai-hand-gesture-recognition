from __future__ import annotations

from pathlib import Path

import numpy as np

from hand_gesture_system.models.base import GestureModelAdapter
from hand_gesture_system.models.prediction_utils import best_prediction_from_output
from hand_gesture_system.types import GesturePrediction

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class TorchScriptGestureAdapter(GestureModelAdapter):
    def __init__(self, model_path: str | Path, labels: list[str] | None = None) -> None:
        if torch is None:
            raise RuntimeError(
                "torch is required for TorchScriptGestureAdapter. "
                "Install with: pip install torch"
            )

        self.model = torch.jit.load(str(model_path), map_location="cpu")
        self.model.eval()
        self.labels = labels

    def predict(self, feature_vector: np.ndarray) -> list[GesturePrediction]:
        assert torch is not None
        x = torch.from_numpy(feature_vector.astype(np.float32)).reshape(1, -1)
        with torch.no_grad():
            output = self.model(x)

        if isinstance(output, tuple):
            output = output[0]
        return best_prediction_from_output(output.detach().cpu().numpy(), self.labels, source="torchscript")
