"""Training + fine-tuning script for STGCNGestureNet.

Usage examples
--------------
# Train from scratch (medium, default)
PYTHONPATH=src python3 -m hand_gesture_system.training.train_stgcn \\
    --dataset data/sequences \\
    --model-size medium \\
    --epochs 60 \\
    --out-dir artifacts/stgcn

# Train the large preset with more regularisation
PYTHONPATH=src python3 -m hand_gesture_system.training.train_stgcn \\
    --dataset data/sequences \\
    --model-size large \\
    --epochs 80 --batch-size 32 --learning-rate 1e-3 \\
    --out-dir artifacts/stgcn_large

# Fine-tune from a saved checkpoint — freeze first 6 blocks, re-train the rest
PYTHONPATH=src python3 -m hand_gesture_system.training.train_stgcn \\
    --dataset data/sequences_new \\
    --finetune-from artifacts/stgcn/stgcn_medium/best_model.pt \\
    --freeze-blocks 6 \\
    --epochs 20 --learning-rate 1e-4 \\
    --out-dir artifacts/stgcn_finetuned

Data format
-----------
Same .npy / .npz sequences used by train.py.
Each sample is a float32 array of shape [frames, features] where the first
63 features are the normalised (x,y,z) coords for all 21 joints in order
[x0,y0,z0, x1,y1,z1, ..., x20,y20,z20].

These are automatically reshaped to [3, T, 21] (C, T, V) before being fed to
STGCNGestureNet.  Any extra features beyond the first 63 are ignored.
"""
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
from hand_gesture_system.training.stgcn_model import (
    STGCN_SIZE_PRESETS,
    STGCNGestureNet,
    build_stgcn,
    count_parameters,
)

# Number of (x,y,z) values per frame — 21 joints × 3 coordinates
_COORD_DIM = 63
_NUM_JOINTS = 21
_COORD_CHANNELS = 3  # x, y, z


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(prefer: str):
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    # "auto" or "mps"
    if prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")
    # auto: prefer MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _coords_to_graph(x_seq: np.ndarray) -> np.ndarray:
    """Reshape [N, T, F] → [N, 3, T, 21].

    Uses the first 63 features (normalised landmark coords) and discards the
    remaining hand-crafted features (finger states, pinch distance) since the
    graph convolution learns its own spatial features from raw coordinates.
    """
    N, T, F = x_seq.shape
    if F < _COORD_DIM:
        raise ValueError(
            f"Expected at least {_COORD_DIM} features per frame, got {F}. "
            "Make sure data was collected with the default FeatureExtractor."
        )
    coords = x_seq[:, :, :_COORD_DIM]                          # [N, T, 63]
    coords = coords.reshape(N, T, _NUM_JOINTS, _COORD_CHANNELS) # [N, T, 21, 3]
    coords = coords.transpose(0, 3, 1, 2)                       # [N, 3, T, 21]
    return coords.astype(np.float32)


def _run_eval(
    model: STGCNGestureNet,
    loader,
    device,
    labels: list[str],
    label_smoothing: float = 0.0,
) -> tuple[float, dict]:
    import torch

    model.eval()
    y_true, y_pred = [], []
    total_loss, total_n = 0.0, 0
    # Use the same label_smoothing as training so val_loss is on the same scale
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * int(yb.size(0))
            total_n += int(yb.size(0))
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, labels=labels)
    return total_loss / max(total_n, 1), metrics


def _export_torchscript(model: STGCNGestureNet, example, path: Path) -> None:
    import torch

    model.eval()
    try:
        scripted = torch.jit.script(model)
        scripted.save(str(path))
    except Exception:
        traced = torch.jit.trace(model, example, check_trace=False)
        traced.save(str(path))


def _freeze_blocks(model: STGCNGestureNet, n: int) -> None:
    """Freeze the first *n* ST-GCN blocks and the input BatchNorm."""
    for param in model.data_bn.parameters():
        param.requires_grad = False
    for i, block in enumerate(model.blocks):
        if i < n:
            for param in block.parameters():
                param.requires_grad = False
    frozen = n
    total = len(model.blocks)
    print(f"Fine-tuning: froze data_bn + blocks 0–{frozen - 1} / {total}; "
          f"training blocks {frozen}–{total - 1} + head.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train / fine-tune STGCNGestureNet (graph-based, no attention)"
    )
    # Data
    p.add_argument("--dataset", required=True,
                   help="Sequence dataset: class-folder directory or .npz file")
    p.add_argument("--x-key", default="X_seq")
    p.add_argument("--y-key", default="y")
    p.add_argument("--sequence-length", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes (default 4; set 0 to disable)")

    # Model
    p.add_argument("--model-size", choices=list(STGCN_SIZE_PRESETS), default="medium",
                   help="ST-GCN size preset (small/medium/large). "
                        "small≈1M, medium≈5M, large≈20M parameters.")

    # Training hyper-params
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Label smoothing for CrossEntropyLoss (helps generalisation)")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    # Fine-tuning
    p.add_argument("--finetune-from", default=None,
                   help="Path to a STGCNGestureNet checkpoint (.pt) to fine-tune from. "
                        "Labels in the checkpoint must match the dataset.")
    p.add_argument("--freeze-blocks", type=int, default=6,
                   help="Number of ST-GCN blocks to freeze when fine-tuning (default 6/9). "
                        "Ignored unless --finetune-from is set.")

    # LR scheduling
    p.add_argument("--lr-scheduler", choices=["none", "cosine", "step"], default="cosine",
                   help="Learning-rate schedule.")
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Linear LR warm-up epochs (used with cosine schedule).")

    # Output
    p.add_argument("--out-dir", default="artifacts/stgcn")
    p.add_argument("--run-name", default=None,
                   help="Sub-folder name. Defaults to 'stgcn_<model_size>'.")
    p.add_argument("--export-torchscript", action="store_true")
    p.add_argument("--export-onnx", action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile() to the model (PyTorch 2+ only).")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_parser().parse_args()

    try:
        import torch
        from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Install with: pip install torch"
        ) from exc

    _set_seed(args.seed)
    device = _device(args.device)
    print(f"Device: {device}")

    # ---- Load and reshape data ------------------------------------------
    x_raw, y_text = load_sequence_dataset(
        path=args.dataset,
        x_key=args.x_key,
        y_key=args.y_key,
        sequence_length=args.sequence_length,
    )
    y, labels, _ = encode_labels(y_text)

    # Convert [N, T, F] → [N, 3, T, 21] for ST-GCN
    x = _coords_to_graph(x_raw)
    print(f"Dataset: {x.shape[0]} samples, {len(labels)} classes, "
          f"input shape per sample: {x.shape[1:]}")

    train_idx, val_idx = stratified_split_indices(y, val_ratio=args.val_ratio, seed=args.seed)

    x_train = torch.tensor(x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    x_val   = torch.tensor(x[val_idx],   dtype=torch.float32)
    y_val   = torch.tensor(y[val_idx],   dtype=torch.long)

    # TensorDataset lives in CPU memory; pin_memory speeds up .to(device) on MPS/CUDA
    _pin = device.type in ("cuda", "mps")
    _nw  = args.num_workers if device.type != "mps" else 0  # MPS: workers cause issues
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=_nw, pin_memory=_pin, persistent_workers=(_nw > 0),
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=_nw, pin_memory=_pin, persistent_workers=(_nw > 0),
    )

    # ---- Build / load model --------------------------------------------
    run_name = args.run_name or f"stgcn_{args.model_size}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_stgcn(
        num_classes=len(labels),
        model_size=args.model_size,
    )

    if args.finetune_from:
        ckpt_path = Path(args.finetune_from)
        print(f"Fine-tuning from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Allow partial load so the head can differ if num_classes changed
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            print(f"  Missing keys (re-initialised): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
        _freeze_blocks(model, args.freeze_blocks)

    model = model.to(device)
    n_params = count_parameters(model)
    print(f"Model: STGCNGestureNet-{args.model_size}  |  "
          f"trainable params: {n_params:,}")

    if args.compile:
        if hasattr(torch, "compile"):
            print("Applying torch.compile() ...", flush=True)
            try:
                # "aot_eager" works on MPS; "inductor" requires CUDA/CPU
                backend = "aot_eager" if device.type == "mps" else "inductor"
                model = torch.compile(model, backend=backend)
                print(f"  compiled with backend='{backend}'")
            except Exception as exc:
                print(f"  torch.compile skipped: {exc}")
        else:
            print("torch.compile not available (PyTorch < 2.0) — skipping")

    # ---- Optimiser & scheduler -----------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scheduler = None
    if args.lr_scheduler == "cosine":
        # Linear warm-up handled manually; CosineAnnealingLR for the rest
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=args.learning_rate * 1e-2,
        )
    elif args.lr_scheduler == "step":
        scheduler = StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)

    # ---- Training loop -------------------------------------------------
    best_val_acc = -1.0
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        # Linear warm-up for cosine schedule
        if args.lr_scheduler == "cosine" and epoch <= args.warmup_epochs:
            warmup_factor = epoch / max(args.warmup_epochs, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = args.learning_rate * warmup_factor

        model.train()
        running_loss, n_samples = 0.0, 0
        n_batches = len(train_loader)

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * int(yb.size(0))
            n_samples += int(yb.size(0))
            # Print progress every 100 batches so training is visible
            if (batch_idx + 1) % 100 == 0:
                avg = running_loss / max(n_samples, 1)
                print(f"  epoch {epoch:03d} [{batch_idx+1:4d}/{n_batches}]"
                      f"  loss={avg:.4f}", flush=True)

        if scheduler is not None and epoch > args.warmup_epochs:
            scheduler.step()

        train_loss = running_loss / max(n_samples, 1)
        val_loss, val_metrics = _run_eval(model, val_loader, device, labels,
                                          label_smoothing=args.label_smoothing)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": current_lr,
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f} | lr={current_lr:.2e}"
        )

        if float(val_metrics["accuracy"]) > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": labels,
                    "model_size": args.model_size,
                    "sequence_length": args.sequence_length,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_acc,
                    "finetune_from": str(args.finetune_from) if args.finetune_from else None,
                },
                out_dir / "best_model.pt",
            )

    save_labels_file(labels, out_dir / "labels.txt")

    summary = {
        "model": "STGCNGestureNet",
        "model_size": args.model_size,
        "trainable_params": n_params,
        "dataset": str(args.dataset),
        "num_samples": int(x.shape[0]),
        "num_classes": len(labels),
        "classes": labels,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "finetune_from": str(args.finetune_from) if args.finetune_from else None,
        "history": history,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # ---- Export --------------------------------------------------------
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    T = args.sequence_length
    example = torch.randn(1, _COORD_CHANNELS, T, _NUM_JOINTS, device=device)

    if args.export_torchscript:
        ts_path = out_dir / "model_stgcn.pt"
        _export_torchscript(model, example, ts_path)
        print(f"TorchScript saved → {ts_path}")

    if args.export_onnx:
        onnx_path = out_dir / "model_stgcn.onnx"
        import torch.onnx
        torch.onnx.export(
            model,
            example,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input":  {0: "batch", 2: "frames"},
                "logits": {0: "batch"},
            },
            opset_version=17,
        )
        print(f"ONNX saved → {onnx_path}")

    print(f"Checkpoint → {out_dir / 'best_model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
