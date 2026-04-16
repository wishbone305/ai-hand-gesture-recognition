from __future__ import annotations

import numpy as np
import requests

from hand_gesture_system.models.base import GestureModelAdapter
from hand_gesture_system.types import GesturePrediction


class HttpGestureAdapter(GestureModelAdapter):
    """Call a remote model server to keep inference backend-agnostic."""

    def __init__(
        self,
        endpoint: str,
        timeout_seconds: float = 2.0,
        api_key: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def predict(self, feature_vector: np.ndarray) -> list[GesturePrediction]:
        payload = {"features": feature_vector.tolist()}
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=self.headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()

        predictions: list[GesturePrediction] = []
        for item in data.get("predictions", []):
            predictions.append(
                GesturePrediction(
                    label=str(item.get("label", "unknown")),
                    score=float(item.get("score", 0.0)),
                    source="http",
                )
            )

        if not predictions and "label" in data:
            predictions.append(
                GesturePrediction(
                    label=str(data["label"]),
                    score=float(data.get("score", 0.0)),
                    source="http",
                )
            )

        return predictions
