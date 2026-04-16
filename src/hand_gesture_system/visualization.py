from __future__ import annotations

import cv2
import numpy as np

from hand_gesture_system.types import FrameResult, GesturePrediction, HandMesh

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _landmark_to_pixel(mesh: HandMesh, index: int, width: int, height: int) -> tuple[int, int]:
    point = mesh.landmarks[index]
    x = int(np.clip(point.x * width, 0, width - 1))
    y = int(np.clip(point.y * height, 0, height - 1))
    return x, y


def _finger_count_from_predictions(predictions: list[GesturePrediction]) -> int | None:
    for prediction in predictions:
        if not prediction.label.startswith("fingers_"):
            continue
        suffix = prediction.label.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _best_gesture_label(predictions: list[GesturePrediction]) -> str:
    for prediction in predictions:
        if not prediction.label.startswith("fingers_"):
            return prediction.label
    if predictions:
        return predictions[0].label
    return "unknown"


def _draw_corner_summary(
    annotated: np.ndarray,
    frame_result: FrameResult,
    recording: bool,
    status_text: str | None,
) -> None:
    height, width = annotated.shape[:2]
    lines: list[str] = []

    if not frame_result.hands:
        lines.append("No hands detected")
    else:
        for hand_result in frame_result.hands:
            finger_count = _finger_count_from_predictions(hand_result.predictions)
            finger_text = (
                f"{finger_count} fingers"
                if finger_count is not None
                else "fingers ?"
            )
            gesture_text = _best_gesture_label(hand_result.predictions)
            lines.append(f"{hand_result.hand_id}: {finger_text} | {gesture_text}")

    if recording:
        lines.append("Recording: ON")
    if status_text:
        lines.append(status_text)

    title = "Gesture Summary"
    line_height = 24
    panel_width = min(520, max(260, int(width * 0.6)))
    panel_height = 20 + line_height * (len(lines) + 1)
    x1, y1 = 14, 14
    x2 = min(width - 14, x1 + panel_width)
    y2 = min(height - 14, y1 + panel_height)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.58, annotated, 0.42, 0, annotated)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 220, 220), 1)

    cv2.putText(
        annotated,
        title,
        (x1 + 12, y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for idx, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (x1 + 12, y1 + 24 + line_height * (idx + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def _draw_record_button(
    annotated: np.ndarray,
    record_button_rect: tuple[int, int, int, int] | None,
    recording: bool,
) -> None:
    if record_button_rect is None:
        return

    x1, y1, x2, y2 = record_button_rect
    fill_color = (30, 30, 210) if recording else (90, 90, 90)
    text_color = (255, 255, 255)
    text = "REC ON" if recording else "REC"

    cv2.rectangle(annotated, (x1, y1), (x2, y2), fill_color, -1)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (235, 235, 235), 2)

    circle_color = (255, 255, 255) if recording else (210, 210, 210)
    cv2.circle(annotated, (x1 + 16, (y1 + y2) // 2), 6, circle_color, -1)
    cv2.putText(
        annotated,
        text,
        (x1 + 30, y1 + int((y2 - y1) * 0.67)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        text_color,
        2,
        cv2.LINE_AA,
    )


def annotate_frame(
    frame_bgr: np.ndarray,
    frame_result: FrameResult,
    recording: bool = False,
    record_button_rect: tuple[int, int, int, int] | None = None,
    status_text: str | None = None,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]

    for hand_result in frame_result.hands:
        mesh = hand_result.mesh

        for start, end in CONNECTIONS:
            x1, y1 = _landmark_to_pixel(mesh, start, width, height)
            x2, y2 = _landmark_to_pixel(mesh, end, width, height)
            cv2.line(annotated, (x1, y1), (x2, y2), (70, 120, 255), 2)

        for idx in range(len(mesh.landmarks)):
            x, y = _landmark_to_pixel(mesh, idx, width, height)
            cv2.circle(annotated, (x, y), 4, (80, 255, 150), -1)

        wrist_x, wrist_y = _landmark_to_pixel(mesh, 0, width, height)
        labels = [f"{p.label}:{p.score:.2f}" for p in hand_result.predictions[:3]]
        header = f"{hand_result.hand_id} {mesh.handedness} ({mesh.confidence:.2f})"

        cv2.putText(
            annotated,
            header,
            (wrist_x + 10, wrist_y - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for offset, label in enumerate(labels):
            cv2.putText(
                annotated,
                label,
                (wrist_x + 10, wrist_y - 20 + (offset * 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    _draw_corner_summary(
        annotated=annotated,
        frame_result=frame_result,
        recording=recording,
        status_text=status_text,
    )
    _draw_record_button(
        annotated=annotated,
        record_button_rect=record_button_rect,
        recording=recording,
    )

    return annotated
