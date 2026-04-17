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
for label_id, cls in enumerate(classes):
    files = sorted((seq_root / cls).glob("seq_*.npy"))
    print(f"  {cls:15s}: {len(files):,} files", end="", flush=True)
    chunk = np.stack([np.load(f) for f in files])
    all_X.append(chunk)
    all_y.extend([label_id] * len(files))
    print(" OK")

X = np.concatenate(all_X, axis=0)
y = np.array(all_y, dtype=np.int32)
print(f"\nTotal: X={X.shape}, y={y.shape}")

t0 = time.time()
np.savez("data/large_sequences_fast.npz", X_seq=X, y=y,
         labels=np.array(classes))
print(f"Saved data/large_sequences_fast.npz in {time.time()-t0:.1f}s")
PY

echo ""
echo "=== Step 2: Launch training for all 4 model sizes ==="
mkdir -p artifacts/logs

for SIZE in small medium large jumbo; do
  LOG="artifacts/logs/train_${SIZE}.log"
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
    > "$LOG" 2>&1 &
  PID=$!
  echo "  $SIZE -> PID $PID, log: $LOG"
done

echo ""
echo "All training jobs launched. Monitor with:"
echo "  tail -f artifacts/logs/train_small.log"
echo "  tail -f artifacts/logs/train_medium.log"
echo "  tail -f artifacts/logs/train_large.log"
echo "  tail -f artifacts/logs/train_jumbo.log"
