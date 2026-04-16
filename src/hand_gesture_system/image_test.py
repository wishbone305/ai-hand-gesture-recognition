from __future__ import annotations

import argparse
import os
from pathlib import Path

from hand_gesture_system.features.extractor import FeatureExtractor
from hand_gesture_system.models.base import GestureModelAdapter
from hand_gesture_system.rules.rule_recognizer import RuleBasedRecognizer
from hand_gesture_system.tracking.mediapipe_tracker import MediaPipeHandTracker
from hand_gesture_system.types import FrameResult, GesturePrediction, HandResult
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


def _merge_predictions(predictions: list[GesturePrediction]) -> list[GesturePrediction]:
    best_by_label: dict[str, GesturePrediction] = {}
    for prediction in predictions:
        existing = best_by_label.get(prediction.label)
        if existing is None or prediction.score > existing.score:
            best_by_label[prediction.label] = prediction

    return sorted(
        best_by_label.values(),
        key=lambda item: item.score,
        reverse=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hand gesture inference on a single image")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--disable-rules", action="store_true")

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
        "--save-annotated",
        type=str,
        default=None,
        help="Write annotated output image to this path",
    )
    parser.add_argument("--show", action="store_true", help="Display annotated image window")
    return parser


def main() -> int:
    import cv2

    args = build_parser().parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if args.max_hands < 1:
        raise ValueError("--max-hands must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if not (0.0 < args.min_detection_confidence <= 1.0):
        raise ValueError("--min-detection-confidence must be in (0, 1]")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not decode image: {image_path}")

    model_adapter = _build_static_model_adapter(args)
    rule_engine = None if args.disable_rules else RuleBasedRecognizer()
    feature_extractor = FeatureExtractor()

    tracker = MediaPipeHandTracker(
        max_hands=args.max_hands,
        static_image_mode=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_detection_confidence,
    )

    try:
        meshes = tracker.detect(frame)
    finally:
        tracker.close()

    if not meshes:
        print("No hands detected.")
        if args.save_annotated:
            output_path = Path(args.save_annotated)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), frame)
            print(f"Saved unannotated image to {output_path}")
        return 0

    hand_results: list[HandResult] = []
    for index, mesh in enumerate(meshes, start=1):
        predictions: list[GesturePrediction] = []
        feature_vector = feature_extractor.extract(mesh)

        if model_adapter is not None:
            predictions.extend(model_adapter.predict(feature_vector))
        if rule_engine is not None:
            predictions.extend(rule_engine.predict(mesh))

        merged = _merge_predictions(predictions)
        hand_result = HandResult(
            hand_id=f"hand_{index}",
            mesh=mesh,
            predictions=merged[: args.top_k],
            feature_vector=feature_vector,
        )
        hand_results.append(hand_result)

    frame_result = FrameResult(hands=hand_results)
    annotated = annotate_frame(frame, frame_result)

    print(f"Detected hands: {len(hand_results)}")
    for hand in hand_results:
        print(f"\n{hand.hand_id} ({hand.mesh.handedness}, conf={hand.mesh.confidence:.2f})")
        if not hand.predictions:
            print("  no predictions")
            continue
        for prediction in hand.predictions:
            print(
                f"  {prediction.label:16s} score={prediction.score:.3f} source={prediction.source}"
            )

    if args.save_annotated:
        output_path = Path(args.save_annotated)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"\nSaved annotated image to {output_path}")

    if args.show:
        cv2.imshow("Hand Gesture Image Test", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
