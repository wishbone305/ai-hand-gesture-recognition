from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hand_gesture_system.config import DTWConfig
from hand_gesture_system.models.base import GestureModelAdapter
from hand_gesture_system.models.sequence_base import SequenceGestureModelAdapter


def _load_labels(path: str | None) -> list[str] | None:
    if not path:
        return None
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    return [line.strip() for line in label_path.read_text().splitlines() if line.strip()]


def _build_static_adapter(args: argparse.Namespace) -> GestureModelAdapter:
    labels = _load_labels(args.labels_path)

    if args.backend == "sklearn":
        from hand_gesture_system.models.sklearn_adapter import SklearnGestureAdapter

        return SklearnGestureAdapter(args.model_path)

    if args.backend == "onnx":
        from hand_gesture_system.models.onnx_adapter import OnnxGestureAdapter

        return OnnxGestureAdapter(args.model_path, labels=labels)

    if args.backend == "torchscript":
        from hand_gesture_system.models.torchscript_adapter import TorchScriptGestureAdapter

        return TorchScriptGestureAdapter(args.model_path, labels=labels)

    if args.backend == "http":
        from hand_gesture_system.models.http_adapter import HttpGestureAdapter

        return HttpGestureAdapter(
            endpoint=args.http_endpoint,
            timeout_seconds=args.http_timeout,
            api_key=None,
        )

    raise ValueError(f"Unsupported static backend: {args.backend}")


def _build_dynamic_adapter(args: argparse.Namespace) -> SequenceGestureModelAdapter:
    labels = _load_labels(args.labels_path)

    if args.backend == "onnx":
        from hand_gesture_system.models.onnx_sequence_adapter import (
            OnnxSequenceGestureAdapter,
        )

        return OnnxSequenceGestureAdapter(
            model_path=args.model_path,
            labels=labels,
            input_layout=args.input_layout,
        )

    if args.backend == "torchscript":
        from hand_gesture_system.models.torchscript_sequence_adapter import (
            TorchScriptSequenceGestureAdapter,
        )

        return TorchScriptSequenceGestureAdapter(
            model_path=args.model_path,
            labels=labels,
            input_layout=args.input_layout,
        )

    if args.backend == "dtw":
        from hand_gesture_system.models.dtw_adapter import DTWGestureAdapter

        cfg = DTWConfig(
            distance_threshold=args.dtw_distance_threshold,
            score_scale=args.dtw_score_scale,
            top_k=args.dtw_top_k,
        )
        return DTWGestureAdapter.from_json(args.dtw_templates_path, cfg=cfg)

    raise ValueError(f"Unsupported dynamic backend: {args.backend}")


def _compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    labels = sorted(set(y_true) | set(y_pred))

    per_class: list[dict[str, float | str]] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(1 for t in y_true if t == label),
            }
        )

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)
    macro_precision = float(np.mean([row["precision"] for row in per_class])) if per_class else 0.0
    macro_recall = float(np.mean([row["recall"] for row in per_class])) if per_class else 0.0
    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def _print_metrics(metrics: dict[str, object]) -> None:
    print("\nSummary")
    print(f"  accuracy        : {metrics['accuracy']:.4f}")
    print(f"  macro_precision : {metrics['macro_precision']:.4f}")
    print(f"  macro_recall    : {metrics['macro_recall']:.4f}")
    print(f"  macro_f1        : {metrics['macro_f1']:.4f}")

    print("\nPer-class")
    print("label\tprecision\trecall\tf1\tsupport")
    for row in metrics["per_class"]:
        print(
            f"{row['label']}\t{row['precision']:.4f}\t{row['recall']:.4f}\t"
            f"{row['f1']:.4f}\t{int(row['support'])}"
        )


def _latency_summary(samples_ms: list[float]) -> dict[str, float]:
    array = np.asarray(samples_ms, dtype=np.float32)
    return {
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def _run_static(args: argparse.Namespace) -> int:
    dataset = np.load(args.dataset_path, allow_pickle=False)
    if args.x_key not in dataset or args.y_key not in dataset:
        raise KeyError(
            f"Dataset must contain keys '{args.x_key}' and '{args.y_key}'. "
            f"Available keys: {list(dataset.keys())}"
        )
    x = np.asarray(dataset[args.x_key], dtype=np.float32)
    y = [str(item) for item in dataset[args.y_key]]

    adapter = _build_static_adapter(args)

    predictions: list[str] = []
    latency_ms: list[float] = []

    for sample in x:
        start = time.perf_counter()
        output = adapter.predict(sample)
        end = time.perf_counter()

        latency_ms.append((end - start) * 1000.0)
        predictions.append(output[0].label if output else "__none__")

    metrics = _compute_metrics(y_true=y, y_pred=predictions)
    latency = _latency_summary(latency_ms)

    print("Static Adapter Benchmark")
    print(f"backend: {args.backend}")
    print(f"samples: {len(y)}")
    print(f"latency mean/median/p95 (ms): {latency['mean_ms']:.3f} / {latency['median_ms']:.3f} / {latency['p95_ms']:.3f}")
    _print_metrics(metrics)

    if args.output_json:
        payload = {
            "mode": "static",
            "backend": args.backend,
            "samples": len(y),
            "latency": latency,
            "metrics": metrics,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote report to {args.output_json}")

    return 0


def _run_dynamic(args: argparse.Namespace) -> int:
    dataset = np.load(args.dataset_path, allow_pickle=False)
    if args.x_key not in dataset or args.y_key not in dataset:
        raise KeyError(
            f"Dataset must contain keys '{args.x_key}' and '{args.y_key}'. "
            f"Available keys: {list(dataset.keys())}"
        )
    x = np.asarray(dataset[args.x_key], dtype=np.float32)
    y = [str(item) for item in dataset[args.y_key]]

    adapter = _build_dynamic_adapter(args)

    predictions: list[str] = []
    latency_ms: list[float] = []

    for sample in x:
        start = time.perf_counter()
        output = adapter.predict_sequence(sample)
        end = time.perf_counter()

        latency_ms.append((end - start) * 1000.0)
        predictions.append(output[0].label if output else "__none__")

    metrics = _compute_metrics(y_true=y, y_pred=predictions)
    latency = _latency_summary(latency_ms)

    print("Dynamic Adapter Benchmark")
    print(f"backend: {args.backend}")
    print(f"samples: {len(y)}")
    print(f"latency mean/median/p95 (ms): {latency['mean_ms']:.3f} / {latency['median_ms']:.3f} / {latency['p95_ms']:.3f}")
    _print_metrics(metrics)

    if args.output_json:
        payload = {
            "mode": "dynamic",
            "backend": args.backend,
            "samples": len(y),
            "latency": latency,
            "metrics": metrics,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote report to {args.output_json}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark gesture model adapters")
    parser.add_argument("--dataset-path", required=True, help="Path to benchmark dataset in .npz format")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    static = subparsers.add_parser("static", help="Benchmark static-feature adapters")
    static.add_argument("--y-key", default="y", help="Dataset key for labels")
    static.add_argument("--labels-path", default=None, help="Optional label file")
    static.add_argument("--output-json", default=None, help="Optional output report path")
    static.add_argument("--backend", required=True, choices=["sklearn", "onnx", "torchscript", "http"])
    static.add_argument("--x-key", default="X", help="Dataset key for static features")
    static.add_argument("--model-path", default=None)
    static.add_argument("--http-endpoint", default=None)
    static.add_argument("--http-timeout", type=float, default=2.0)

    dynamic = subparsers.add_parser("dynamic", help="Benchmark temporal adapters")
    dynamic.add_argument("--y-key", default="y", help="Dataset key for labels")
    dynamic.add_argument("--labels-path", default=None, help="Optional label file")
    dynamic.add_argument("--output-json", default=None, help="Optional output report path")
    dynamic.add_argument("--backend", required=True, choices=["onnx", "torchscript", "dtw"])
    dynamic.add_argument("--x-key", default="X_seq", help="Dataset key for temporal features")
    dynamic.add_argument("--model-path", default=None)
    dynamic.add_argument("--input-layout", choices=["btd", "td", "flat"], default="btd")
    dynamic.add_argument("--dtw-templates-path", default=None)
    dynamic.add_argument("--dtw-distance-threshold", type=float, default=0.35)
    dynamic.add_argument("--dtw-score-scale", type=float, default=6.0)
    dynamic.add_argument("--dtw-top-k", type=int, default=3)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "static":
        if args.backend in {"sklearn", "onnx", "torchscript"} and not args.model_path:
            raise ValueError("--model-path is required for static backend")
        if args.backend == "http" and not args.http_endpoint:
            raise ValueError("--http-endpoint is required for http static backend")
        return _run_static(args)

    if args.mode == "dynamic":
        if args.backend in {"onnx", "torchscript"} and not args.model_path:
            raise ValueError("--model-path is required for dynamic backend")
        if args.backend == "dtw" and not args.dtw_templates_path:
            raise ValueError("--dtw-templates-path is required for DTW backend")
        return _run_dynamic(args)

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
