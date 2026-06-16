#!/usr/bin/env python3
"""Build real-landmark dataset from the HaGRID image dataset on HuggingFace.

Streams images from Jayabalambika/hagrid-classification-512p-dataset,
runs MediaPipe HandLandmarker on each, and writes coord caches to disk.
Then generates temporal sequences and saves large_sequences_fast.npz.

Usage
-----
    PYTHONPATH=src python3 scripts/build_hagrid_dataset.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# HaGRID label → our gesture class mapping
# Multiple HaGRID labels can map to the same gesture.
# ---------------------------------------------------------------------------
HAGRID_TO_GESTURE: dict[str, str] = {
    "call":            "call_me",
    "fist":            "fist",
    "four":            "four",
    "like":            "thumbs_up",
    "ok":              "ok",
    "one":             "pointing",
    "palm":            "open_palm",
    "peace":           "peace",
    "rock":            "rock",
    "three":           "three",
    "three2":          "three",        # alt three
    "stop":            "open_palm",    # open flat hand = extra open_palm
    "mute":            "open_palm",    # similar open hand
    "two_up":          "peace",        # 2 fingers up = peace variant
}

# Classes we can't get from HaGRID — use base-class borrowing
DYNAMIC_BASE: dict[str, str] = {
    "swipe_left":  "open_palm",
    "swipe_right": "open_palm",
    "grab":        "fist",
    "release":     "open_palm",
    "pinch_zoom":  "pinch",
    # These have no HaGRID equivalent; keep whatever Wikimedia gave us:
    "gun":         None,
    "hand_heart":  None,
    "pinch":       None,
    "spiderman":   None,
    "vulcan":      None,
}

ALL_CLASSES = sorted([
    "call_me", "fist", "four", "grab", "gun", "hand_heart",
    "ok", "open_palm", "peace", "pinch", "pinch_zoom", "pointing",
    "release", "rock", "spiderman", "swipe_left", "swipe_right",
    "three", "thumbs_up", "vulcan",
])


def _build_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mpv
    from mediapipe.tasks.python.core import base_options as mpb
    task = Path.home() / ".cache" / "hand_gesture_system" / "models" / "hand_landmarker.task"
    if not task.exists():
        import urllib.request
        task.parent.mkdir(parents=True, exist_ok=True)
        print("[hagrid] Downloading MediaPipe model…")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            task,
        )
    opts = mpv.HandLandmarkerOptions(
        base_options=mpb.BaseOptions(model_asset_path=str(task)),
        running_mode=mpv.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    return mpv.HandLandmarker.create_from_options(opts), mp


def _extract_coords(landmarker, mp_mod, pil_image):
    """Extract wrist-relative normalised coords from a PIL image."""
    import numpy as np
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb)
    try:
        result = landmarker.detect(mp_img)
    except Exception:
        return None
    if not result.hand_landmarks:
        return None
    lms = result.hand_landmarks[0]
    wx, wy, wz = lms[0].x, lms[0].y, lms[0].z
    scale = max(np.sqrt((lms[9].x-wx)**2+(lms[9].y-wy)**2+(lms[9].z-wz)**2), 1e-6)
    coords = np.array(
        [[(lm.x-wx)/scale, (lm.y-wy)/scale, (lm.z-wz)/scale] for lm in lms],
        dtype=np.float32,
    ).flatten()
    if np.any(~np.isfinite(coords)):
        return None
    return coords


# ---- sequence builder -----------------------------------------------------

def _ou_noise(shape, rng, theta=0.25, sigma=0.009):
    x = np.zeros(shape, dtype=np.float32)
    for t in range(1, shape[0]):
        x[t] = x[t-1]*(1-theta) + rng.standard_normal(shape[1:]).astype(np.float32)*sigma
    return x

def _rot(ax, ay, az):
    cx,sx=np.cos(ax),np.sin(ax); cy,sy=np.cos(ay),np.sin(ay); cz,sz=np.cos(az),np.sin(az)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]],dtype=np.float32)
    Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]],dtype=np.float32)
    Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]],dtype=np.float32)
    return Rx@Ry@Rz

def _make_sequence(base: np.ndarray, gesture: str, seq_len: int, rng) -> np.ndarray:
    pose = base.reshape(21, 3).copy()
    r = np.deg2rad(25.0)
    pose = pose @ _rot(rng.uniform(-r,r), rng.uniform(-r,r), rng.uniform(-r*.4,r*.4)).T
    pose *= rng.uniform(0.9, 1.10)
    frames = np.tile(pose[np.newaxis], (seq_len, 1, 1))
    if gesture in ("swipe_left", "swipe_right"):
        d = -1.0 if gesture == "swipe_left" else 1.0
        frames[:, 0, 0] = np.linspace(0, d * rng.uniform(0.25, 0.70), seq_len)
    elif gesture == "grab":
        t = np.linspace(0, 1, seq_len)
        for j in range(1, 21):
            frames[:, j, :] += (rng.uniform(0.03, 0.12, 3) * t[:, None]).astype(np.float32)
    elif gesture == "release":
        t = np.linspace(0, 1, seq_len)
        for j in range(1, 21):
            frames[:, j, :] -= (rng.uniform(0.03, 0.10, 3) * t[:, None]).astype(np.float32)
    elif gesture == "pinch_zoom":
        t = np.linspace(0, 1, seq_len)
        for j in [6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]:
            frames[:, j, :] += (rng.uniform(0.04, 0.11, 3) * t[:, None]).astype(np.float32)
    tremor = _ou_noise((seq_len, 21, 3), rng, sigma=0.009)
    tremor[:, 0, :] = 0.0
    frames += tremor
    amp = rng.uniform(0.0, 0.025)
    axis = rng.uniform(-1, 1, 3); axis /= np.linalg.norm(axis) + 1e-8
    for t in range(seq_len):
        ang = amp * np.sin(np.pi * t / seq_len)
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]], dtype=np.float32)
        Rd = np.eye(3, dtype=np.float32) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)
        frames[t] = frames[t] @ Rd.T
    return frames.reshape(seq_len, 63).astype(np.float32)


# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target-frames",   type=int, default=300,
                   help="Target real-landmark frames per primary class (default 300)")
    p.add_argument("--cache-dir",  default="data/real_sequences/coords_cache")
    p.add_argument("--seqs-per-image",  type=int, default=20)
    p.add_argument("--seq-len",         type=int, default=48)
    p.add_argument("--npz-out",   default="data/large_sequences_fast.npz")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--no-cache",        action="store_true",
                   help="Ignore existing caches and rebuild from scratch")
    p.add_argument("--min-frames",      type=int, default=20,
                   help="Skip classes with fewer than this many real frames (default 20)")
    return p


def main():
    args = _build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[hagrid] Loading dataset (streaming)…")
    from datasets import load_dataset
    ds = load_dataset(
        "Jayabalambika/hagrid-classification-512p-dataset",
        split="train",
        streaming=True,
    )

    print("[hagrid] Initialising MediaPipe…")
    landmarker, mp_mod = _build_landmarker()

    # Accumulate coords per gesture class from HaGRID
    coords_map: dict[str, list[np.ndarray]] = {g: [] for g in ALL_CLASSES}

    # Load existing caches (unless --no-cache)
    if not args.no_cache:
        for cls in ALL_CLASSES:
            cp = cache_dir / f"{cls}.npy"
            if cp.exists():
                coords_map[cls] = list(np.load(cp, allow_pickle=False))
                if coords_map[cls]:
                    print(f"  [{cls}] loaded {len(coords_map[cls])} from cache")

    # Collect enough images from HaGRID for each gesture we can map
    # Build per-gesture need map
    gesture_needs: dict[str, int] = {}
    for hagrid_lbl, gesture in HAGRID_TO_GESTURE.items():
        need = args.target_frames - len(coords_map[gesture])
        if need > 0:
            gesture_needs[gesture] = max(gesture_needs.get(gesture, 0), need)

    if not gesture_needs:
        print("[hagrid] All classes already at target — skipping download.")
    else:
        print(f"\n[hagrid] Need frames for: {', '.join(f'{g}={n}' for g,n in sorted(gesture_needs.items()))}")
        print("[hagrid] Streaming images…\n")

        # Get label names from dataset features
        label_names: list[str] = []
        for item in ds.take(1):
            feat = ds.features if hasattr(ds, 'features') else None
            if feat and hasattr(feat.get('label'), 'names'):
                label_names = feat['label'].names
            break

        # Reload fresh iterator
        ds_iter = iter(ds.shuffle(seed=42, buffer_size=10000))

        processed = detected = 0
        start = time.time()

        for item in ds_iter:
            # Check if we still need this label
            lbl_idx = item["label"]
            if label_names:
                hagrid_label = label_names[lbl_idx]
            else:
                hagrid_label = str(lbl_idx)

            gesture = HAGRID_TO_GESTURE.get(hagrid_label)
            if gesture is None:
                continue  # not a class we use
            if len(coords_map[gesture]) >= args.target_frames:
                continue  # already full

            processed += 1
            pil_img = item["image"]
            c = _extract_coords(landmarker, mp_mod, pil_img)
            if c is not None:
                coords_map[gesture].append(c)
                detected += 1
                if detected % 50 == 0:
                    elapsed = time.time() - start
                    summary = "  ".join(
                        f"{g}={len(coords_map[g])}"
                        for g in sorted(gesture_needs)
                    )
                    print(f"  [{detected:4d} det / {processed:5d} proc | {elapsed:.0f}s]  {summary}")

            # Save partial caches every 500 detections
            if detected % 500 == 0 and detected > 0:
                _save_caches(coords_map, cache_dir, ALL_CLASSES)

            # Done when all needed gestures are at target
            still_need = any(
                len(coords_map[g]) < args.target_frames
                for g in gesture_needs
            )
            if not still_need:
                break

        elapsed = time.time() - start
        print(f"\n[hagrid] Done: {detected} detections from {processed} images in {elapsed:.0f}s")

    landmarker.close()

    # ------------------------------------------------------------------
    # Propagate dynamic classes from their base
    # ------------------------------------------------------------------
    for cls, base in DYNAMIC_BASE.items():
        if base and not coords_map[cls]:
            coords_map[cls] = coords_map[base].copy()
            print(f"  [{cls}] borrowed {len(coords_map[cls])} frames from {base}")

    # Save all caches
    _save_caches(coords_map, cache_dir, ALL_CLASSES)

    # ------------------------------------------------------------------
    # Rebuild temporal sequences
    # ------------------------------------------------------------------
    print("\n[hagrid] Building temporal sequences…")
    all_X: list[np.ndarray] = []
    all_y: list[str] = []

    for cls in sorted(ALL_CLASSES):
        frames = coords_map[cls]
        if not frames:
            print(f"  WARNING: no frames for {cls} — skipping")
            continue
        if len(frames) < args.min_frames:
            print(f"  WARNING: {cls} only has {len(frames)} frames (< {args.min_frames}) — skipping")
            continue
        cls_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        seqs = [
            _make_sequence(b, cls, args.seq_len, cls_rng)
            for b in frames
            for _ in range(args.seqs_per_image)
        ]
        print(f"  {cls:15s}: {len(frames):4d} frames → {len(seqs):5d} seqs")
        all_X.append(np.stack(seqs))
        all_y.extend([cls] * len(seqs))

    # Balance
    Xall = np.concatenate(all_X)
    yall = np.array(all_y)
    counts = {c: int((yall == c).sum()) for c in np.unique(yall)}
    min_n = min(counts.values())
    bal_rng = np.random.default_rng(0)
    bX, by = [], []
    for c in sorted(counts):
        idx = np.where(yall == c)[0]
        if len(idx) > min_n:
            idx = bal_rng.choice(idx, min_n, replace=False)
        bX.append(Xall[idx])
        by.extend([c] * len(idx))
    X = np.concatenate(bX)
    y = np.array(by)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]

    classes_out = sorted(np.unique(y))
    print(f"\nFinal: {len(X):,} sequences × {len(classes_out)} classes ({min_n} each)")

    npz_out = Path(args.npz_out)
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_out, X_seq=X, y=y, labels=np.array(classes_out))
    print(f"Saved {npz_out}  ✓")


def _save_caches(coords_map: dict, cache_dir: Path, classes: list[str]) -> None:
    for cls in classes:
        if coords_map[cls]:
            np.save(cache_dir / f"{cls}.npy", np.stack(coords_map[cls]))


if __name__ == "__main__":
    main()
