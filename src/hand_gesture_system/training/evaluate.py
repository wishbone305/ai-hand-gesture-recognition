from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hand_gesture_system.training.dataio import (
    compute_classification_metrics,
    encode_labels,
    load_sequence_dataset,
)
from hand_gesture_system.training.model import build_model


def _device(prefer: str):
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "cuda":
        raise RuntimeError("CUDA requested but no CUDA device available")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained temporal gesture model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, help="Dataset path (.npz or class-folder directory)")
    parser.add_argument("--x-key", default="X_seq")
    parser.add_argument("--y-key", default="y")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for sequence evaluation. Install with: pip install torch"
        ) from exc

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    labels = [str(item) for item in checkpoint["labels"]]

    x, y_text = load_sequence_dataset(
        path=args.dataset,
        x_key=args.x_key,
        y_key=args.y_key,
        sequence_length=args.sequence_length or int(checkpoint["sequence_length"]),
    )

    y, discovered_labels, _ = encode_labels(y_text)
    if discovered_labels != labels:
        unknown = sorted(set(discovered_labels) - set(labels))
        if unknown:
            raise ValueError(
                f"Dataset has labels not present in checkpoint: {unknown}"
            )
        checkpoint_mapping = {label: idx for idx, label in enumerate(labels)}
        y = np.array([checkpoint_mapping[label] for label in y_text], dtype=np.int64)

    model = build_model(
        input_dim=int(checkpoint["input_dim"]),
        num_classes=len(labels),
        hidden_dim=int(checkpoint["model_config"]["hidden_dim"]),
        num_heads=int(checkpoint["model_config"]["num_heads"]),
        num_layers=int(checkpoint["model_config"]["num_layers"]),
        dropout=float(checkpoint["model_config"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])

    device = _device(args.device)
    model.to(device)
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()

            y_true.extend(y_batch.numpy().tolist())
            y_pred.extend(predictions)

    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, labels=labels)

    print("Evaluation Results")
    print(f"accuracy       : {metrics['accuracy']:.4f}")
    print(f"macro_precision: {metrics['macro_precision']:.4f}")
    print(f"macro_recall   : {metrics['macro_recall']:.4f}")
    print(f"macro_f1       : {metrics['macro_f1']:.4f}")
    print("\nPer-class")
    print("label\tprecision\trecall\tf1\tsupport")
    for row in metrics["per_class"]:
        print(
            f"{row['label']}\t{row['precision']:.4f}\t{row['recall']:.4f}\t"
            f"{row['f1']:.4f}\t{int(row['support'])}"
        )

    if args.output_json:
        payload = {
            "checkpoint": str(args.checkpoint),
            "dataset": str(args.dataset),
            "metrics": metrics,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote evaluation report to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
