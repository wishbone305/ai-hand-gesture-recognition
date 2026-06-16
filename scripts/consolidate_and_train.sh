#!/usr/bin/env bash
# Consolidate .npy → fast .npz then launch training for all 4 model sizes.
set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Consolidate dataset ==="
PYTHONPATH=src python3 - <<'PY'
import numpy as np
from pathlib import Path
import time, json

seq_root = Path("data/large_sequences")
summary = json.loads((seq_root / "summary.json").read_text())
print(f"  {summary['total_sequences']:,} sequences, {summary['total_size_gb']:.2f} GB")

classes = sorted(d.name for d in seq_root.iterdir() if d.is_dir())
all_X, all_y = [], []
for cls in classes:
    files = sorted((seq_root / cls).glob("seq_*.npy"))
    print(f"  {cls:15s}: {len(files):,} files", end="", flush=True)
    chunk = np.stack([np.load(f) for f in files])
    all_X.append(chunk)
    all_y.extend([cls] * len(files))   # string class name, not int index
    print(" OK")

X = np.concatenate(all_X, axis=0)
# Save y as string gesture names (not integers) so dataio.py encodes them
# correctly and checkpoints store real class names like "fist", not "0","1"
y_str = np.array(all_y)          # already strings from class-dir names
print(f"\nTotal: X={X.shape}, y={y_str.shape}, sample y: {y_str[:3]}")

t0 = time.time()
np.savez("data/large_sequences_fast.npz", X_seq=X, y=y_str,
         labels=np.array(classes))
print(f"Saved data/large_sequences_fast.npz in {time.time()-t0:.1f}s")
PY

echo ""
echo "=== Step 2: Train all 4 model sizes sequentially (one at a time to fit in 16 GB RAM) ==="
mkdir -p artifacts/logs

for SIZE in small medium large jumbo; do
  LOG="artifacts/logs/train_${SIZE}.log"
  echo "  Starting $SIZE ..."
  # Run foreground — wait for it to finish before starting the next size.
  # Each job uses ~3-4 GB; running sequentially keeps total RAM under 6 GB.
  PYTHONPATH=src python3 -u -m hand_gesture_system.training.train_stgcn \
    --dataset data/large_sequences_fast.npz \
    --model-size "$SIZE" \
    --epochs 60 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --label-smoothing 0.1 \
    --lr-scheduler cosine \
    --warmup-epochs 5 \
    --sequence-length 48 \
    --device mps \
    --out-dir artifacts/stgcn_v2 \
    --run-name "stgcn_${SIZE}" \
    --export-torchscript \
    2>&1 | tee "$LOG"
  echo "  $SIZE done -> artifacts/stgcn_v2/stgcn_${SIZE}/best_model.pt"
  echo ""
done

echo "All 4 models trained."
