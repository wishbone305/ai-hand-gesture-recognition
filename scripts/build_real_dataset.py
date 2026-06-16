#!/usr/bin/env python3
"""Build a real-landmark gesture dataset.

Pipeline
--------
1. For each gesture class, collect real images from:
   a. data/web_images  (already downloaded)
   b. Wikimedia Commons (sequential download, rate-limited)
2. Run MediaPipe HandLandmarker on every image (static mode).
3. Wrist-relative, palm-scale-normalised 21-joint coords — same as demo.py.
4. Build temporal sequences: tile the real frame + correlated tremor + drift.
5. Dynamic gestures (swipe, grab, release, pinch_zoom) inherit real poses from
   related static classes and add the appropriate motion overlay.
6. Save data/large_sequences_fast.npz  ready for train_stgcn.py.

Usage
-----
    cd "<project root>"
    PYTHONPATH=src python3 scripts/build_real_dataset.py

Flags
-----
    --target-per-class  Real images to collect per class  (default 200)
    --seqs-per-image    Augmented sequences per image      (default 20)
    --seq-len           Frames per sequence               (default 48)
    --out-dir           Image + coord cache root          (default data/real_sequences)
    --npz-out           Output npz                        (default data/large_sequences_fast.npz)
    --skip-download     Use cached coords only, no web
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Gesture → Wikimedia search queries  (multiple per class → more results)
# ---------------------------------------------------------------------------
GESTURE_QUERIES: dict[str, list[str]] = {
    "call_me":    ["shaka sign hand gesture",
                   "call me hand gesture hang loose pinky thumb",
                   "hang loose surfing hand sign"],
    "fist":       ["clenched fist raised power",
                   "closed fist hand gesture knuckles",
                   "fist bump gesture closed hand"],
    "four":       ["four fingers hand sign counting",
                   "four hand gesture extended fingers",
                   "hand showing four fingers"],
    "grab":       ["hand grasping object gesture",
                   "claw hand gesture fingers curved",
                   "grabbing hand gesture"],
    "gun":        ["finger gun hand gesture",
                   "hand gun sign pointing index thumb",
                   "pistol finger gesture thumb up index pointing"],
    "hand_heart": ["finger heart gesture",
                   "mini heart Korean finger sign",
                   "heart hand gesture fingers"],
    "ok":         ["okay sign hand gesture circle",
                   "OK hand sign index thumb ring",
                   "ok hand gesture approval circle fingers"],
    "open_palm":  ["open palm hand five fingers",
                   "hand stop gesture flat palm facing",
                   "open hand gesture spread fingers"],
    "peace":      ["peace sign hand gesture V victory",
                   "two fingers V sign peace",
                   "peace fingers hand sign raised"],
    "pinch":      ["pinch hand gesture fingertips touching",
                   "thumb index finger pinch gesture",
                   "precision pinch hand gesture"],
    "pinch_zoom": ["fingers spread apart gesture",
                   "zoom in gesture fingers spread",
                   "hand pinch spread open gesture"],
    "pointing":   ["pointing index finger gesture",
                   "hand pointing one finger extended",
                   "index finger pointing direction"],
    "release":    ["open hand release gesture",
                   "let go hand gesture open fingers",
                   "hand dropping releasing gesture"],
    "rock":       ["rock sign hand gesture devil horns",
                   "rock on hand gesture index pinky extended",
                   "metal horns hand sign concert"],
    "spiderman":  ["spiderman web shooting hand gesture",
                   "spider-man hand sign web slinging",
                   "spidey sense hand gesture"],
    "swipe_left": ["hand swipe gesture flat open",
                   "swiping hand gesture screen",
                   "hand flat gesture horizontal swipe"],
    "swipe_right":["hand swipe right flat open",
                   "swiping right gesture",
                   "hand gesture horizontal wave"],
    "three":      ["three fingers hand gesture counting",
                   "three hand sign fingers up",
                   "hand showing three fingers"],
    "thumbs_up":  ["thumbs up hand gesture approval like",
                   "thumb up fist approval sign",
                   "thumbs up hand like positive"],
    "vulcan":     ["vulcan salute hand sign Spock",
                   "live long prosper hand gesture",
                   "Star Trek Vulcan hand sign split fingers"],
}

# Dynamic gestures borrow real base poses from related static gestures
DYNAMIC_BASE: dict[str, str] = {
    "swipe_left":  "open_palm",
    "swipe_right": "open_palm",
    "grab":        "fist",
    "release":     "open_palm",
    "pinch_zoom":  "pinch",
}

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"


# ---------------------------------------------------------------------------
# Wikimedia search
# ---------------------------------------------------------------------------

def _wikimedia_image_urls(
    query: str,
    session: requests.Session,
    max_pages: int = 8,
    thumb_width: int = 640,
    timeout: float = 15.0,
) -> list[str]:
    urls: list[str] = []
    continue_params: dict = {}
    for _ in range(max_pages):
        params = {
            "action": "query", "format": "json",
            "generator": "search", "gsrnamespace": 6,
            "gsrsearch": query, "gsrlimit": 50,
            "prop": "imageinfo", "iiprop": "url|size|mime",
            "iiurlwidth": thumb_width,
        }
        params.update(continue_params)
        try:
            r = session.get(WIKIMEDIA_API, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            data = r.json()
        except Exception:
            break
        for page in data.get("query", {}).get("pages", {}).values():
            for info in page.get("imageinfo") or []:
                url = info.get("thumburl") or info.get("url") or ""
                mime = info.get("mime", "")
                w = info.get("thumbwidth") or info.get("width") or 0
                h = info.get("thumbheight") or info.get("height") or 0
                if url and mime.startswith("image/") and w >= 200 and h >= 200:
                    urls.append(url)
        if "continue" not in data:
            break
        continue_params = data["continue"]
        time.sleep(0.15)          # be polite; avoid 429
    return urls


def _download_image(url: str, session: requests.Session, timeout: float = 20.0):
    """Download URL → BGR numpy array.  Returns None on any failure."""
    for attempt in range(3):
        try:
            r = session.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            ct = r.headers.get("Content-Type", "")
            if "image" not in ct and "octet-stream" not in ct:
                return None
            arr = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img if img is not None and img.size > 0 else None
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# MediaPipe extraction
# ---------------------------------------------------------------------------

def _build_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mpv
    from mediapipe.tasks.python.core import base_options as mpb

    task_path = Path.home() / ".cache" / "hand_gesture_system" / "models" / "hand_landmarker.task"
    if not task_path.exists():
        print("[dataset] Downloading MediaPipe model…")
        task_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            timeout=120,
        )
        r.raise_for_status()
        task_path.write_bytes(r.content)

    opts = mpv.HandLandmarkerOptions(
        base_options=mpb.BaseOptions(model_asset_path=str(task_path)),
        running_mode=mpv.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return mpv.HandLandmarker.create_from_options(opts), mp


def _extract_coords(landmarker, mp_module, bgr: np.ndarray) -> np.ndarray | None:
    """BGR image → [63] wrist-relative normalised coords, or None if no hand."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
    try:
        result = landmarker.detect(mp_img)
    except Exception:
        return None
    if not result.hand_landmarks:
        return None
    lms = result.hand_landmarks[0]
    wx, wy, wz = lms[0].x, lms[0].y, lms[0].z
    scale = max(
        np.sqrt((lms[9].x-wx)**2 + (lms[9].y-wy)**2 + (lms[9].z-wz)**2), 1e-6
    )
    coords = np.array(
        [[(lm.x-wx)/scale, (lm.y-wy)/scale, (lm.z-wz)/scale] for lm in lms],
        dtype=np.float32,
    ).flatten()
    # Sanity check: wrist should be at origin
    if np.any(np.abs(coords[:3]) > 0.05) or np.any(~np.isfinite(coords)):
        return None
    return coords


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def _ou_noise(shape, rng, theta=0.25, sigma=0.009):
    x = np.zeros(shape, dtype=np.float32)
    for t in range(1, shape[0]):
        x[t] = x[t-1]*(1-theta) + rng.standard_normal(shape[1:]).astype(np.float32)*sigma
    return x


def _rot(ax, ay, az):
    cx,sx = np.cos(ax),np.sin(ax)
    cy,sy = np.cos(ay),np.sin(ay)
    cz,sz = np.cos(az),np.sin(az)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]],dtype=np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]],dtype=np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]],dtype=np.float32)
    return Rx@Ry@Rz


def _make_sequence(
    base: np.ndarray,   # [63] real MediaPipe frame
    gesture: str,
    seq_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pose = base.reshape(21, 3).copy()

    # Random orientation (moderate — real images already have variation)
    r = np.deg2rad(25.0)
    R = _rot(rng.uniform(-r,r), rng.uniform(-r,r), rng.uniform(-r*.4,r*.4))
    pose = pose @ R.T
    pose *= rng.uniform(0.9, 1.10)

    frames = np.tile(pose[np.newaxis], (seq_len, 1, 1))   # [T, 21, 3]

    # Motion overlays for dynamic gestures
    if gesture in ("swipe_left", "swipe_right"):
        direction = -1.0 if gesture == "swipe_left" else 1.0
        travel = rng.uniform(0.25, 0.70)
        frames[:, 0, 0] = np.linspace(0, direction * travel, seq_len)

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
        for j in [6,7,8,10,11,12,14,15,16,18,19,20]:
            frames[:, j, :] += (rng.uniform(0.04, 0.11, 3) * t[:, None]).astype(np.float32)

    # Correlated tremor (wrist stays at origin / displacement value)
    tremor = _ou_noise((seq_len, 21, 3), rng, sigma=0.009)
    tremor[:, 0, :] = 0.0
    frames += tremor

    # Slow drift
    amp = rng.uniform(0.0, 0.025)
    axis = rng.uniform(-1, 1, 3); axis /= np.linalg.norm(axis)+1e-8
    for t in range(seq_len):
        ang = amp * np.sin(np.pi * t / seq_len)
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]],dtype=np.float32)
        Rd = np.eye(3,dtype=np.float32) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)
        frames[t] = frames[t] @ Rd.T

    return frames.reshape(seq_len, 63).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target-per-class", type=int, default=200)
    p.add_argument("--seqs-per-image",   type=int, default=20)
    p.add_argument("--seq-len",          type=int, default=48)
    p.add_argument("--out-dir",   default="data/real_sequences")
    p.add_argument("--npz-out",   default="data/large_sequences_fast.npz")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args  = build_parser().parse_args()
    rng   = np.random.default_rng(args.seed)
    root  = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root/"coords_cache").mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "hand-gesture-research/1.0 (academic; non-commercial)",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    })

    print("[dataset] Initialising MediaPipe…")
    landmarker, mp_mod = _build_landmarker()

    all_classes = sorted(GESTURE_QUERIES.keys())
    real_coords: dict[str, list[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Phase 1 — collect real MediaPipe coords for all non-dynamic classes
    # ------------------------------------------------------------------
    primary_classes = [c for c in all_classes if c not in DYNAMIC_BASE]

    for cls in primary_classes:
        cache_path = root / "coords_cache" / f"{cls}.npy"
        coords_list: list[np.ndarray] = []

        # Load cache
        if cache_path.exists():
            cached = np.load(cache_path)
            coords_list = list(cached)
            print(f"  [{cls}] loaded {len(coords_list)} cached")

        # Process any existing web_images for this class first
        existing_dir = Path("data/web_images") / cls
        if existing_dir.is_dir():
            for img_path in sorted(existing_dir.glob("*.*")):
                if img_path.suffix.lower() not in {".jpg",".jpeg",".png",".webp"}:
                    continue
                img = cv2.imread(str(img_path))
                if img is None: continue
                c = _extract_coords(landmarker, mp_mod, img)
                if c is not None:
                    coords_list.append(c)
            if coords_list:
                print(f"  [{cls}] +{len(coords_list)} from existing images")

        if args.skip_download or len(coords_list) >= args.target_per_class:
            real_coords[cls] = coords_list
            continue

        # Download from Wikimedia — sequential, rate-limited
        needed = args.target_per_class - len(coords_list)
        print(f"\n[{cls}] Downloading for {needed} more coords…")

        # Collect candidate URLs from all queries for this class
        candidate_urls: list[str] = []
        seen_urls: set[str] = set()
        for q in GESTURE_QUERIES[cls]:
            new_urls = _wikimedia_image_urls(q, session, max_pages=6)
            for u in new_urls:
                if u not in seen_urls:
                    seen_urls.add(u)
                    candidate_urls.append(u)
            time.sleep(0.3)   # pause between queries

        print(f"  {len(candidate_urls)} unique candidate URLs")
        dl_ok = det = 0

        for url in candidate_urls:
            if len(coords_list) >= args.target_per_class:
                break
            img = _download_image(url, session)
            time.sleep(0.08)   # ~12 req/s max — avoids 429
            if img is None:
                continue
            dl_ok += 1
            c = _extract_coords(landmarker, mp_mod, img)
            if c is not None:
                coords_list.append(c)
                det += 1
                if det % 20 == 0:
                    print(f"    {cls}: {len(coords_list)}/{args.target_per_class} "
                          f"(dl={dl_ok}, det={det})")

        print(f"  [{cls}] final: {len(coords_list)} real frames "
              f"(downloaded {dl_ok}, detected {det})")

        # Save cache
        if coords_list:
            np.save(cache_path, np.stack(coords_list))

        real_coords[cls] = coords_list

    # ------------------------------------------------------------------
    # Phase 2 — dynamic classes inherit from their base
    # ------------------------------------------------------------------
    for cls, base_cls in DYNAMIC_BASE.items():
        real_coords[cls] = real_coords.get(base_cls, []).copy()
        print(f"  [{cls}] using {len(real_coords[cls])} frames from {base_cls}")

    # ------------------------------------------------------------------
    # Phase 3 — build temporal sequences
    # ------------------------------------------------------------------
    print("\n[dataset] Building temporal sequences…")
    all_X: list[np.ndarray] = []
    all_y: list[str]        = []
    min_seqs = None

    for cls in all_classes:
        frames = real_coords.get(cls, [])
        if not frames:
            print(f"  WARNING: no frames for {cls} — skipping")
            continue
        cls_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        seqs = [
            _make_sequence(base, cls, args.seq_len, cls_rng)
            for base in frames
            for _ in range(args.seqs_per_image)
        ]
        n = len(seqs)
        print(f"  {cls:15s}: {len(frames):4d} frames × {args.seqs_per_image} = {n:5d} seqs")
        all_X.append(np.stack(seqs))
        all_y.extend([cls] * n)
        min_seqs = n if min_seqs is None else min(min_seqs, n)

    # ------------------------------------------------------------------
    # Phase 4 — balance classes by downsampling to min count
    # ------------------------------------------------------------------
    if min_seqs is not None and min_seqs > 0:
        balanced_X, balanced_y = [], []
        classes_present = sorted(set(all_y))
        Xall = np.concatenate(all_X)
        yall = np.array(all_y)
        bal_rng = np.random.default_rng(0)
        for cls in classes_present:
            idx = np.where(yall == cls)[0]
            if len(idx) > min_seqs:
                idx = bal_rng.choice(idx, min_seqs, replace=False)
            balanced_X.append(Xall[idx])
            balanced_y.extend([cls] * len(idx))
        X = np.concatenate(balanced_X)
        y = np.array(balanced_y)
    else:
        X = np.concatenate(all_X) if all_X else np.empty((0,))
        y = np.array(all_y)

    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]

    classes_out = sorted(np.unique(y))
    print(f"\nFinal dataset: {len(X):,} sequences, {len(classes_out)} classes")
    for c in classes_out:
        n = int((y==c).sum())
        print(f"  {c:15s}: {n}")

    t0 = time.time()
    np.savez(args.npz_out, X_seq=X, y=y, labels=np.array(classes_out))
    print(f"\nSaved {args.npz_out} in {time.time()-t0:.1f}s  ✓")
    landmarker.close()


if __name__ == "__main__":
    main()
