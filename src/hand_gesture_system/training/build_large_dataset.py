"""Build a large synthetic gesture dataset (target ≥ 1 GB on disk).

Usage
-----
# Build to data/large_sequences with default 1 GB target
PYTHONPATH=src python3 -m hand_gesture_system.training.build_large_dataset

# Custom target and sequence length
PYTHONPATH=src python3 -m hand_gesture_system.training.build_large_dataset \\
    --out-dir data/large_sequences \\
    --target-gb 1.5 \\
    --seq-len 32 \\
    --seed 42

# Also augment any existing .npy sequences in data/sequences/
PYTHONPATH=src python3 -m hand_gesture_system.training.build_large_dataset \\
    --existing-dir data/sequences \\
    --aug-factor 20

Output layout
-------------
data/large_sequences/
    open_palm/   seq_000000.npy, seq_000001.npy, ...
    fist/        ...
    thumbs_up/   ...
    ...           (20 gesture classes)

Each .npy file is a float32 array of shape [seq_len, 63].
A summary.json is written to the root with per-class counts and total size.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _bytes_to_gb(b: int) -> float:
    return b / (1024 ** 3)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build large synthetic gesture dataset")
    p.add_argument("--out-dir", default="data/large_sequences")
    p.add_argument("--target-gb", type=float, default=1.0,
                   help="Target total dataset size in GB (default 1.0)")
    p.add_argument("--seq-len", type=int, default=32,
                   help="Number of frames per sequence (default 32)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--existing-dir", default=None,
                   help="Optional: path to existing .npy sequences to also augment")
    p.add_argument("--aug-factor", type=int, default=20,
                   help="How many augmented copies per existing sequence (default 20)")
    p.add_argument("--batch-size", type=int, default=500,
                   help="Sequences written per batch (controls memory use)")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    # Late import so module is importable without these at top level
    from hand_gesture_system.training.augment import random_augment, mirror_sequence
    from hand_gesture_system.training.synthetic_gestures import (
        GESTURE_NAMES,
        generate_sequence,
    )

    rng = np.random.default_rng(args.seed)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    bytes_per_seq = args.seq_len * 63 * 4  # float32
    target_bytes  = int(args.target_gb * 1024 ** 3)

    n_gestures = len(GESTURE_NAMES)
    # Distribute target equally across all gesture classes
    seqs_per_class = int(np.ceil(target_bytes / (n_gestures * bytes_per_seq)))
    print(f"Target: {args.target_gb:.1f} GB  |  "
          f"{n_gestures} gestures  |  "
          f"~{seqs_per_class:,} sequences/class  |  "
          f"seq_len={args.seq_len}")

    total_written = 0
    total_bytes   = 0
    t0 = time.time()
    class_summary: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 1. Generate synthetic sequences
    # ------------------------------------------------------------------
    for gesture_name in GESTURE_NAMES:
        class_dir = out_root / gesture_name
        class_dir.mkdir(exist_ok=True)

        # Check how many already exist (allows resuming)
        existing = sorted(class_dir.glob("seq_*.npy"))
        start_idx = len(existing)
        needed    = max(0, seqs_per_class - start_idx)

        if needed == 0:
            print(f"  {gesture_name:15s}: already has {start_idx:,} seqs — skipping")
            class_summary[gesture_name] = start_idx
            total_bytes += start_idx * bytes_per_seq
            continue

        print(f"  {gesture_name:15s}: generating {needed:,} sequences ...", end="", flush=True)

        written = 0
        while written < needed:
            batch = min(args.batch_size, needed - written)
            for i in range(batch):
                seq = generate_sequence(gesture_name, seq_len=args.seq_len, rng=rng)
                # 50% chance of an extra augmentation pass (adds diversity)
                if rng.uniform() < 0.5:
                    seq = random_augment(seq, rng)
                idx = start_idx + written + i
                np.save(class_dir / f"seq_{idx:07d}.npy", seq)
            written += batch
            total_written += batch

        class_summary[gesture_name] = start_idx + written
        total_bytes += (start_idx + written) * bytes_per_seq
        elapsed = time.time() - t0
        rate = total_written / max(elapsed, 1)
        print(f" done  ({start_idx + written:,} total, {rate:.0f} seq/s)")

    # ------------------------------------------------------------------
    # 2. Augment existing sequences (if supplied)
    # ------------------------------------------------------------------
    if args.existing_dir:
        existing_root = Path(args.existing_dir)
        if not existing_root.exists():
            print(f"Warning: --existing-dir '{existing_root}' not found, skipping.")
        else:
            for class_dir in sorted(existing_root.iterdir()):
                if not class_dir.is_dir():
                    continue
                npy_files = sorted(class_dir.glob("*.npy"))
                if not npy_files:
                    continue
                gesture_name = class_dir.name
                out_dir_aug = out_root / gesture_name
                out_dir_aug.mkdir(exist_ok=True)
                existing_out = sorted(out_dir_aug.glob("aug_*.npy"))
                start_idx = len(existing_out)
                print(f"  Augmenting {gesture_name:15s}: "
                      f"{len(npy_files)} real seqs × {args.aug_factor} …", end="", flush=True)
                idx = start_idx
                for npy_path in npy_files:
                    orig = np.load(npy_path)
                    if orig.ndim != 2 or orig.shape[1] < 63:
                        continue
                    orig = orig[:, :63].astype(np.float32)
                    if len(orig) != args.seq_len:
                        # Interpolate to target seq_len
                        t_src = np.linspace(0, 1, len(orig))
                        t_tgt = np.linspace(0, 1, args.seq_len)
                        orig = np.stack([np.interp(t_tgt, t_src, orig[:, f])
                                         for f in range(orig.shape[1])], axis=1).astype(np.float32)
                    # Mirror copy
                    np.save(out_dir_aug / f"aug_{idx:07d}.npy", mirror_sequence(orig))
                    idx += 1
                    for _ in range(args.aug_factor - 1):
                        aug = random_augment(orig, rng)
                        np.save(out_dir_aug / f"aug_{idx:07d}.npy", aug)
                        idx += 1
                        total_written += 1
                        total_bytes += bytes_per_seq
                added = idx - start_idx
                class_summary[gesture_name] = class_summary.get(gesture_name, 0) + added
                print(f" +{added:,}")

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    total_gb = _bytes_to_gb(total_bytes)

    summary = {
        "out_dir":      str(out_root.resolve()),
        "total_sequences": sum(class_summary.values()),
        "total_size_gb":   round(total_gb, 3),
        "seq_len":      args.seq_len,
        "feature_dim":  63,
        "classes":      class_summary,
        "elapsed_s":    round(elapsed, 1),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDataset built in {elapsed:.0f}s")
    print(f"  Total sequences : {summary['total_sequences']:,}")
    print(f"  Total size      : {total_gb:.2f} GB")
    print(f"  Output dir      : {out_root.resolve()}")

    if total_gb < args.target_gb * 0.95:
        print(f"WARNING: reached only {total_gb:.2f} GB (target {args.target_gb:.1f} GB). "
              "Re-run to generate more sequences.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
