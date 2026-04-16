from __future__ import annotations

import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Export trained sequence model to TorchScript/ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torchscript", action="store_true")
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
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
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for sequence export. Install with: pip install torch"
        ) from exc

    if not args.torchscript and not args.onnx:
        raise ValueError("Select at least one export target: --torchscript and/or --onnx")

    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model = build_model(
        input_dim=int(checkpoint["input_dim"]),
        num_classes=len(checkpoint["labels"]),
        hidden_dim=int(checkpoint["model_config"]["hidden_dim"]),
        num_heads=int(checkpoint["model_config"]["num_heads"]),
        num_layers=int(checkpoint["model_config"]["num_layers"]),
        dropout=float(checkpoint["model_config"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    sequence_length = int(checkpoint["sequence_length"])
    input_dim = int(checkpoint["input_dim"])
    example = torch.randn(1, sequence_length, input_dim, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.torchscript:
        _export_torchscript(model, example, out_dir / "model_sequence.pt")
        print(f"TorchScript exported: {out_dir / 'model_sequence.pt'}")

    if args.onnx:
        torch.onnx.export(
            model,
            example,
            str(out_dir / "model_sequence.onnx"),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch", 1: "frames"}, "logits": {0: "batch"}},
            opset_version=args.opset,
        )
        print(f"ONNX exported: {out_dir / 'model_sequence.onnx'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
