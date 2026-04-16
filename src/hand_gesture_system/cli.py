from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from hand_gesture_system.config import (
    ConfusionFixConfig,
    DTWConfig,
    PipelineConfig,
    SmoothingConfig,
    TemporalConfig,
)
from hand_gesture_system.models.base import GestureModelAdapter
from hand_gesture_system.models.sequence_base import SequenceGestureModelAdapter
from hand_gesture_system.pipeline import GesturePipeline
from hand_gesture_system.rules.rule_recognizer import RuleBasedRecognizer
from hand_gesture_system.visualization import annotate_frame


def _load_labels(path: str | None) -> list[str] | None:
    if not path:
        return None
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    return [line.strip() for line in label_path.read_text().splitlines() if line.strip()]


def _build_static_model_adapter(args: argparse.Namespace) -> GestureModelAdapter | None:
    if args.model_type == "none":
        return None

    if args.model_type in {"sklearn", "onnx", "torchscript"} and not args.model_path:
        raise ValueError("--model-path is required for sklearn/onnx/torchscript backends")

    labels = _load_labels(args.labels_path)

    if args.model_type == "sklearn":
        from hand_gesture_system.models.sklearn_adapter import SklearnGestureAdapter

        return SklearnGestureAdapter(args.model_path)

    if args.model_type == "onnx":
        from hand_gesture_system.models.onnx_adapter import OnnxGestureAdapter

        return OnnxGestureAdapter(args.model_path, labels=labels)

    if args.model_type == "torchscript":
        from hand_gesture_system.models.torchscript_adapter import TorchScriptGestureAdapter

        return TorchScriptGestureAdapter(args.model_path, labels=labels)

    if args.model_type == "custom":
        from hand_gesture_system.models.custom_adapter import (
            CustomGestureAdapter,
            load_custom_config,
        )

        if not args.custom_model_entrypoint:
            raise ValueError("--custom-model-entrypoint is required for custom backend")

        return CustomGestureAdapter(
            entrypoint=args.custom_model_entrypoint,
            model_path=args.model_path,
            labels=labels,
            config=load_custom_config(args.custom_model_config),
        )

    if args.model_type == "http":
        from hand_gesture_system.models.http_adapter import HttpGestureAdapter

        if not args.http_endpoint:
            raise ValueError("--http-endpoint is required for HTTP backend")
        api_key = os.environ.get("GESTURE_API_KEY")
        return HttpGestureAdapter(
            endpoint=args.http_endpoint,
            timeout_seconds=args.http_timeout,
            api_key=api_key,
        )

    raise ValueError(f"Unsupported model type: {args.model_type}")


def _build_sequence_model_adapter(
    args: argparse.Namespace,
) -> SequenceGestureModelAdapter | None:
    if args.dynamic_model_type == "none":
        return None

    if args.dynamic_model_type in {"onnx", "torchscript"} and not args.dynamic_model_path:
        raise ValueError(
            "--dynamic-model-path is required for dynamic model backends"
        )

    labels = _load_labels(args.dynamic_labels_path)

    if args.dynamic_model_type == "onnx":
        from hand_gesture_system.models.onnx_sequence_adapter import (
            OnnxSequenceGestureAdapter,
        )

        return OnnxSequenceGestureAdapter(
            model_path=args.dynamic_model_path,
            labels=labels,
            input_layout=args.dynamic_input_layout,
        )

    if args.dynamic_model_type == "torchscript":
        from hand_gesture_system.models.torchscript_sequence_adapter import (
            TorchScriptSequenceGestureAdapter,
        )

        return TorchScriptSequenceGestureAdapter(
            model_path=args.dynamic_model_path,
            labels=labels,
            input_layout=args.dynamic_input_layout,
        )

    if args.dynamic_model_type == "custom":
        from hand_gesture_system.models.custom_adapter import (
            CustomSequenceGestureAdapter,
            load_custom_config,
        )

        if not args.custom_dynamic_entrypoint:
            raise ValueError(
                "--custom-dynamic-entrypoint is required for custom dynamic backend"
            )

        return CustomSequenceGestureAdapter(
            entrypoint=args.custom_dynamic_entrypoint,
            model_path=args.dynamic_model_path,
            labels=labels,
            config=load_custom_config(args.custom_dynamic_config),
        )

    raise ValueError(f"Unsupported dynamic model type: {args.dynamic_model_type}")


def _build_dtw_adapter(args: argparse.Namespace, cfg: DTWConfig) -> SequenceGestureModelAdapter | None:
    if not args.dtw_templates_path:
        return None

    from hand_gesture_system.models.dtw_adapter import DTWGestureAdapter

    return DTWGestureAdapter.from_json(args.dtw_templates_path, cfg=cfg)


def _record_button_rect(frame_width: int) -> tuple[int, int, int, int]:
    button_width = 130
    button_height = 42
    margin = 14
    x2 = max(margin + button_width, frame_width - margin)
    x1 = x2 - button_width
    y1 = margin
    y2 = y1 + button_height
    return x1, y1, x2, y2


def _point_in_rect(x: int, y: int, rect: tuple[int, int, int, int] | None) -> bool:
    if rect is None:
        return False
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time hand mesh + gesture recognition")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-hands", type=int, default=2)

    parser.add_argument(
        "--model-type",
        choices=["none", "sklearn", "onnx", "torchscript", "http", "custom"],
        default="none",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--labels-path", type=str, default=None)
    parser.add_argument(
        "--custom-model-entrypoint",
        type=str,
        default=None,
        help=(
            "Custom static model entrypoint as 'module.path:Symbol' or "
            "'/path/to/file.py:Symbol'"
        ),
    )
    parser.add_argument(
        "--custom-model-config",
        type=str,
        default=None,
        help="JSON config file passed into custom model constructor",
    )
    parser.add_argument("--http-endpoint", type=str, default=None)
    parser.add_argument("--http-timeout", type=float, default=2.0)

    parser.add_argument(
        "--dynamic-model-type",
        choices=["none", "onnx", "torchscript", "custom"],
        default="none",
    )
    parser.add_argument("--dynamic-model-path", type=str, default=None)
    parser.add_argument("--dynamic-labels-path", type=str, default=None)
    parser.add_argument(
        "--custom-dynamic-entrypoint",
        type=str,
        default=None,
        help=(
            "Custom sequence model entrypoint as 'module.path:Symbol' or "
            "'/path/to/file.py:Symbol'"
        ),
    )
    parser.add_argument(
        "--custom-dynamic-config",
        type=str,
        default=None,
        help="JSON config file passed into custom sequence model constructor",
    )
    parser.add_argument(
        "--dynamic-input-layout",
        choices=["btd", "td", "flat"],
        default="btd",
        help="Expected input layout for temporal sequence model",
    )

    parser.add_argument(
        "--dtw-templates-path",
        type=str,
        default=None,
        help="JSON file containing template sequences for DTW fallback",
    )
    parser.add_argument("--dtw-distance-threshold", type=float, default=0.35)
    parser.add_argument("--dtw-score-scale", type=float, default=6.0)
    parser.add_argument("--dtw-top-k", type=int, default=3)

    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--min-sequence-frames", type=int, default=8)

    parser.add_argument("--disable-rules", action="store_true", help="Only use learned model predictions")
    parser.add_argument("--disable-smoothing", action="store_true")
    parser.add_argument("--disable-confusion-fix", action="store_true")
    parser.add_argument("--smoothing-window", type=int, default=8)
    parser.add_argument("--smoothing-min-consecutive", type=int, default=3)
    parser.add_argument("--smoothing-stable-score", type=float, default=0.9)

    parser.add_argument("--hand-assignment-threshold", type=float, default=0.25)
    parser.add_argument("--max-missed-frames", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument("--window-name", type=str, default="Hand Gesture Recognition")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal frame flip")
    parser.add_argument("--disable-record-button", action="store_true")
    parser.add_argument("--recordings-dir", type=str, default="recordings")
    parser.add_argument("--recording-fps", type=float, default=30.0)
    parser.add_argument(
        "--recording-codec",
        type=str,
        default="mp4v",
        help="FourCC codec, e.g. mp4v or MJPG",
    )
    return parser


def main() -> int:
    import cv2

    args = build_parser().parse_args()

    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be >= 2")
    if args.min_sequence_frames < 2:
        raise ValueError("--min-sequence-frames must be >= 2")
    if args.min_sequence_frames > args.sequence_length:
        raise ValueError("--min-sequence-frames cannot be larger than --sequence-length")
    if args.smoothing_window < 1:
        raise ValueError("--smoothing-window must be >= 1")
    if args.smoothing_min_consecutive < 1:
        raise ValueError("--smoothing-min-consecutive must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.max_missed_frames < 1:
        raise ValueError("--max-missed-frames must be >= 1")
    if args.hand_assignment_threshold <= 0:
        raise ValueError("--hand-assignment-threshold must be > 0")
    if args.recording_fps <= 0:
        raise ValueError("--recording-fps must be > 0")
    if len(args.recording_codec) != 4:
        raise ValueError("--recording-codec must be exactly 4 characters")

    temporal_cfg = TemporalConfig(
        sequence_length=args.sequence_length,
        min_sequence_frames=args.min_sequence_frames,
        include_velocity=True,
        include_acceleration=True,
    )
    smoothing_cfg = SmoothingConfig(
        enabled=not args.disable_smoothing,
        window_size=args.smoothing_window,
        min_consecutive_frames=args.smoothing_min_consecutive,
        stable_score=args.smoothing_stable_score,
    )
    confusion_cfg = ConfusionFixConfig(enabled=not args.disable_confusion_fix)
    dtw_cfg = DTWConfig(
        distance_threshold=args.dtw_distance_threshold,
        score_scale=args.dtw_score_scale,
        top_k=args.dtw_top_k,
    )

    config = PipelineConfig(
        max_hands=args.max_hands,
        top_k_predictions=args.top_k,
        include_rules=not args.disable_rules,
        hand_assignment_distance_threshold=args.hand_assignment_threshold,
        max_missed_frames=args.max_missed_frames,
        temporal=temporal_cfg,
        smoothing=smoothing_cfg,
        confusion_fix=confusion_cfg,
        dtw=dtw_cfg,
    )

    static_model_adapter = _build_static_model_adapter(args)
    sequence_model_adapter = _build_sequence_model_adapter(args)
    dtw_adapter = _build_dtw_adapter(args, cfg=dtw_cfg)

    from hand_gesture_system.tracking.mediapipe_tracker import MediaPipeHandTracker

    tracker = MediaPipeHandTracker(max_hands=args.max_hands)
    rule_engine = None if args.disable_rules else RuleBasedRecognizer()

    pipeline = GesturePipeline(
        tracker=tracker,
        rule_engine=rule_engine,
        model_adapter=static_model_adapter,
        sequence_model_adapter=sequence_model_adapter,
        dtw_adapter=dtw_adapter,
        config=config,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    cv2.namedWindow(args.window_name)

    record_button_enabled = not args.disable_record_button
    recordings_dir = Path(args.recordings_dir)
    recording_state: dict[str, object] = {
        "active": False,
        "writer": None,
        "output_path": None,
        "button_rect": None,
        "pending_toggle": False,
    }

    def _stop_recording() -> None:
        writer = recording_state["writer"]
        if writer is not None:
            writer.release()
        recording_state["active"] = False
        recording_state["writer"] = None
        output_path = recording_state["output_path"]
        recording_state["output_path"] = None
        if output_path is not None:
            print(f"Recording saved: {output_path}")

    def _start_recording(frame_width: int, frame_height: int) -> None:
        recordings_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = recordings_dir / f"gesture_recording_{timestamp}.mp4"

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = (
            source_fps
            if 1.0 <= source_fps <= 120.0
            else float(args.recording_fps)
        )

        fourcc = cv2.VideoWriter_fourcc(*args.recording_codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            writer.release()
            print("Failed to start recording: could not initialize VideoWriter")
            return

        recording_state["active"] = True
        recording_state["writer"] = writer
        recording_state["output_path"] = output_path
        print(f"Recording started: {output_path}")

    def _toggle_recording(frame_width: int, frame_height: int) -> None:
        if bool(recording_state["active"]):
            _stop_recording()
        else:
            _start_recording(frame_width=frame_width, frame_height=frame_height)

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if not record_button_enabled:
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if _point_in_rect(x, y, recording_state["button_rect"]):  # type: ignore[arg-type]
            recording_state["pending_toggle"] = True

    cv2.setMouseCallback(args.window_name, _on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            if record_button_enabled:
                recording_state["button_rect"] = _record_button_rect(frame.shape[1])
            else:
                recording_state["button_rect"] = None

            if bool(recording_state["pending_toggle"]):
                _toggle_recording(frame_width=frame.shape[1], frame_height=frame.shape[0])
                recording_state["pending_toggle"] = False

            frame_result = pipeline.process(frame)
            status_text = None
            output_path = recording_state["output_path"]
            if output_path is not None and bool(recording_state["active"]):
                status_text = f"File: {Path(output_path).name}"

            annotated = annotate_frame(
                frame,
                frame_result,
                recording=bool(recording_state["active"]),
                record_button_rect=recording_state["button_rect"],  # type: ignore[arg-type]
                status_text=status_text,
            )

            writer = recording_state["writer"]
            if writer is not None and bool(recording_state["active"]):
                writer.write(annotated)

            cv2.imshow(args.window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("r"), ord("R")}:
                _toggle_recording(frame_width=frame.shape[1], frame_height=frame.shape[0])
                continue
            if key in {27, ord("q")}:
                break
    finally:
        if bool(recording_state["active"]):
            _stop_recording()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
