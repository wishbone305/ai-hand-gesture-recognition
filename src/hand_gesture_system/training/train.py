from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hand_gesture_system.training.dataio import (
    compute_classification_metrics,
    encode_labels,
    load_sequence_dataset,
    save_labels_file,
    stratified_split_indices,
)
from hand_gesture_system.training.model import MODEL_SIZE_PRESETS, build_model


def _set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _device_for_training(prefer: str):
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


def _run_evaluation(model, loader, device, labels: list[str]) -> tuple[float, dict[str, object]]:
    import torch

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item()) * int(y_batch.shape[0])
            total_samples += int(y_batch.shape[0])

            predictions = torch.argmax(logits, dim=1)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, labels=labels)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train temporal gesture sequence model")
    parser.add_argument("--dataset", required=True, help="Dataset path (.npz or class-folder directory)")
    parser.add_argument("--x-key", default="X_seq")
    parser.add_argument("--y-key", default="y")
    parser.add_argument("--sequence-length", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_SIZE_PRESETS),
        default=None,
        help=(
            "Preset model size (overrides --hidden-dim / --num-heads / --num-layers / --dropout). "
            "small=128d/2L, medium=192d/2L, large=384d/4L, xlarge=512d/6L"
        ),
    )

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--out-dir", default="artifacts/sequence")
    parser.add_argument("--run-name", default="temporal_gesture_net")

    parser.add_argument("--export-torchscript", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    return parser


def _export_torchscript(model, example, out_path) -> None:
    import torch

    model.eval()
    try:
        scripted = torch.jit.script(model)
        scripted.save(str(out_path))
    except Exception:
        traced = torch.jit.trace(model, example, check_trace=False)
        traced.save(str(out_path))


def main() -> int:
    args = build_parser().parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for sequence training. Install with: pip install torch"
        ) from exc

    _set_seed(args.seed)

    x, y_text = load_sequence_dataset(
        path=args.dataset,
        x_key=args.x_key,
        y_key=args.y_key,
        sequence_length=args.sequence_length,
    )
    y, labels, _ = encode_labels(y_text)

    if x.ndim != 3:
        raise ValueError(f"Expected input tensor [samples, frames, features], got {x.shape}")

    train_idx, val_idx = stratified_split_indices(y, val_ratio=args.val_ratio, seed=args.seed)

    x_train = torch.tensor(x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    x_val = torch.tensor(x[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = _device_for_training(args.device)

    model = build_model(
        input_dim=int(x.shape[2]),
        num_classes=len(labels),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_size=args.model_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_epoch = -1
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(y_batch.shape[0])
            sample_count += int(y_batch.shape[0])

        train_loss = running_loss / max(sample_count, 1)
        val_loss, val_metrics = _run_evaluation(model, val_loader, device, labels)

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_row)

        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if float(val_metrics["accuracy"]) > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_epoch = epoch

            checkpoint = {
                "model_state": model.state_dict(),
                "labels": labels,
                "input_dim": int(x.shape[2]),
                "sequence_length": int(x.shape[1]),
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "model_size": args.model_size,
                },
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_acc,
            }
            torch.save(checkpoint, out_dir / "best_model.pt")

    save_labels_file(labels, out_dir / "labels.txt")

    summary = {
        "dataset": str(args.dataset),
        "num_samples": int(x.shape[0]),
        "num_classes": len(labels),
        "classes": labels,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "history": history,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if args.export_torchscript:
        example = torch.randn(1, int(x.shape[1]), int(x.shape[2]), device=device)
        _export_torchscript(model, example, out_dir / "model_sequence.pt")

    if args.export_onnx:
        example = torch.randn(1, int(x.shape[1]), int(x.shape[2]), device=device)
        torch.onnx.export(
            model,
            example,
            str(out_dir / "model_sequence.onnx"),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch", 1: "frames"}, "logits": {0: "batch"}},
            opset_version=17,
        )

    print(f"Best checkpoint written to {out_dir / 'best_model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
