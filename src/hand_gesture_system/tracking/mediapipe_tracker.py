from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hand_gesture_system.tracking.base import HandTracker
from hand_gesture_system.types import HandMesh, Landmark

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None

HAND_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _default_task_model_path() -> Path:
    return Path.home() / ".cache" / "hand_gesture_system" / "models" / "hand_landmarker.task"


def _ensure_task_model(task_model_path: str | Path | None) -> Path:
    model_path = Path(task_model_path) if task_model_path else _default_task_model_path()
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "requests is required to auto-download the MediaPipe task model."
        ) from exc

    response = requests.get(HAND_LANDMARKER_TASK_URL, timeout=120)
    response.raise_for_status()
    model_path.write_bytes(response.content)
    return model_path


class MediaPipeHandTracker(HandTracker):
    def __init__(
        self,
        max_hands: int = 2,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        task_model_path: str | None = None,
    ) -> None:
        if mp is None:
            raise RuntimeError(
                "mediapipe is required for MediaPipeHandTracker. "
                "Install dependencies from requirements.txt."
            )

        self._backend = "tasks"
        self._hands = None
        self._landmarker = None
        self._static_image_mode = static_image_mode
        self._video_timestamp_ms = 0

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._backend = "solutions"
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "MediaPipe installation does not expose solutions or tasks APIs."
            ) from exc

        model_path = _ensure_task_model(task_model_path)
        running_mode = (
            vision.RunningMode.IMAGE
            if static_image_mode
            else vision.RunningMode.VIDEO
        )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> list[HandMesh]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            assert self._hands is not None
            results = self._hands.process(rgb)

            if not results.multi_hand_landmarks:
                return []

            output: list[HandMesh] = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = "unknown"
                confidence = 0.0

                if results.multi_handedness and idx < len(results.multi_handedness):
                    classification = results.multi_handedness[idx].classification[0]
                    handedness = classification.label.lower()
                    confidence = float(classification.score)

                points = [
                    Landmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks.landmark
                ]
                output.append(
                    HandMesh(
                        landmarks=points,
                        handedness=handedness,
                        confidence=confidence,
                    )
                )

            return output

        assert self._landmarker is not None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._static_image_mode:
            result = self._landmarker.detect(mp_image)
        else:
            self._video_timestamp_ms += 33
            result = self._landmarker.detect_for_video(
                mp_image, self._video_timestamp_ms
            )

        if not result.hand_landmarks:
            return []

        output: list[HandMesh] = []
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = "unknown"
            confidence = 0.0

            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                category = result.handedness[idx][0]
                handedness = str(category.category_name).lower()
                confidence = float(category.score)

            points = [
                Landmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ]
            output.append(
                HandMesh(
                    landmarks=points,
                    handedness=handedness,
                    confidence=confidence,
                )
            )

        return output

    def close(self) -> None:
        if self._backend == "solutions" and self._hands is not None:
            self._hands.close()
            return
        if self._backend == "tasks" and self._landmarker is not None:
            self._landmarker.close()
