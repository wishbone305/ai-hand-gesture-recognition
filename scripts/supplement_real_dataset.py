#!/usr/bin/env python3
"""Second-pass download: supplement coord caches for classes below --min-frames.

Runs AFTER build_real_dataset.py.  It reads existing caches, identifies
classes that are still short, hits alternative Wikimedia queries, and
appends newly detected coords.  Then rebuilds large_sequences_fast.npz.

Usage
-----
    PYTHONPATH=src python3 scripts/supplement_real_dataset.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Alternative queries targeting CLOSE-UP HAND PHOTOS
# Kept intentionally different from the first-pass queries.
# ---------------------------------------------------------------------------
SUPPLEMENT_QUERIES: dict[str, list[str]] = {
    "call_me":    ["shaka sign close-up hand",
                   "hang loose hand gesture photo close up",
                   "call hand gesture pinky thumb extended close"],
    "fist":       ["fist raised hand closeup photo",
                   "closed fist knuckles hand photograph",
                   "fist gesture hand closeup front view"],
    "four":       ["hand gesture four fingers extended close-up",
                   "counting four hand sign fingers photo",
                   "four finger hand gesture number counting"],
    "grab":       ["hand grasping claw gesture close photo",
                   "grab hand gesture fingers bent curl",
                   "claw hand gesture close up photo"],
    "gun":        ["pistol hand sign index thumb gesture close",
                   "finger gun gesture hand photo",
                   "gun hand sign close-up thumb index"],
    "hand_heart": ["finger heart hand close up photo",
                   "Korean heart finger gesture close-up",
                   "mini heart hand sign fingers photo"],
    "ok":         ["OK hand sign close-up photo gesture",
                   "okay gesture circle index thumb hand close",
                   "hand sign OK ring finger gesture photo"],
    "open_palm":  ["open hand flat fingers spread gesture close up",
                   "palm hand five fingers extended front",
                   "spread fingers hand gesture photo close-up"],
    "peace":      ["peace V sign fingers close up photo",
                   "victory sign hand two fingers close-up",
                   "peace fingers hand gesture photograph"],
    "pinch":      ["pinch gesture index thumb touching close-up",
                   "finger pinch gesture close up photo",
                   "thumb index finger pinch hand photo"],
    "pinch_zoom": ["fingers spread open gesture photo close-up",
                   "zoom gesture hand spread fingers close up",
                   "two hand pinch spread apart gesture"],
    "pointing":   ["index finger pointing close-up hand photo",
                   "pointing hand gesture one finger front",
                   "hand pointing direction index finger photo"],
    "release":    ["open hand release drop gesture photo",
                   "fingers open spread release hand close up",
                   "hand open gesture releasing fingers photo"],
    "rock":       ["rock devil horns hand sign close up photo",
                   "metal horns hand gesture index pinky photo",
                   "rock music hand sign fingers close up"],
    "spiderman":  ["spiderman web gesture hand photo close up",
                   "spider man hand sign shooting web",
                   "spidey gesture hand fingers photo"],
    "swipe_left": ["flat open hand horizontal gesture photo",
                   "hand gesture swiping sideways photo",
                   "open palm hand gesture side horizontal"],
    "swipe_right":["open palm hand facing camera photo",
                   "flat hand gesture horizontal front view",
                   "hand open gesture horizontal swipe photo"],
    "three":      ["three fingers hand gesture close-up photo",
                   "counting three hand sign fingers photograph",
                   "three finger hand gesture front view"],
    "thumbs_up":  ["thumbs up close-up hand photo gesture",
                   "thumb up fist gesture photograph close",
                   "like hand gesture thumb extended close up"],
    "vulcan":     ["Vulcan salute hand gesture close-up photo",
                   "spock live long prosper hand sign close",
                   "star trek hand gesture fingers split close up"],
}

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

DYNAMIC_BASE: dict[str, str] = {
    "swipe_left":  "open_palm",
    "swipe_right": "open_palm",
    "grab":        "fist",
    "release":     "open_palm",
    "pinch_zoom":  "pinch",
}


def _wikimedia_urls(query: str, session: requests.Session,
                    max_pages: int = 10, thumb_width: int = 640) -> list[str]:
    urls: list[str] = []
    cont: dict = {}
    for _ in range(max_pages):
        params = {
            "action": "query", "format": "json",
            "generator": "search", "gsrnamespace": 6,
            "gsrsearch": query, "gsrlimit": 50,
            "prop": "imageinfo", "iiprop": "url|size|mime",
            "iiurlwidth": thumb_width,
        }
        params.update(cont)
        try:
            r = session.get(WIKIMEDIA_API, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(5); continue
            r.raise_for_status()
            data = r.json()
        except Exception:
            break
        for page in data.get("query", {}).get("pages", {}).values():
            for info in (page.get("imageinfo") or []):
                url = info.get("thumburl") or info.get("url") or ""
                mime = info.get("mime", "")
                w = info.get("thumbwidth") or info.get("width") or 0
                h = info.get("thumbheight") or info.get("height") or 0
                if url and mime.startswith("image/") and w >= 200 and h >= 200:
                    urls.append(url)
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(0.15)
    return urls


def _download_image(url: str, session: requests.Session):
    for attempt in range(3):
        try:
            r = session.get(url, timeout=20, allow_redirects=True)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1)); continue
            if r.status_code != 200: return None
            ct = r.headers.get("Content-Type", "")
            if "image" not in ct and "octet-stream" not in ct: return None
            arr = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img if img is not None and img.size > 0 else None
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


def _build_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mpv
    from mediapipe.tasks.python.core import base_options as mpb
    task = Path.home() / ".cache" / "hand_gesture_system" / "models" / "hand_landmarker.task"
    opts = mpv.HandLandmarkerOptions(
        base_options=mpb.BaseOptions(model_asset_path=str(task)),
        running_mode=mpv.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.30,
        min_hand_presence_confidence=0.30,
        min_tracking_confidence=0.30,
    )
    return mpv.HandLandmarker.create_from_options(opts), mp


def _extract_coords(landmarker, mp_mod, bgr: np.ndarray):
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
    if np.any(np.abs(coords[:3]) > 0.05) or np.any(~np.isfinite(coords)):
        return None
    return coords


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--min-frames",      type=int, default=80,
                   help="Supplement any class below this frame count (default 80)")
    p.add_argument("--target-frames",   type=int, default=200,
                   help="Target frame count per class (default 200)")
    p.add_argument("--cache-dir",  default="data/real_sequences/coords_cache")
    p.add_argument("--seqs-per-image",  type=int, default=20)
    p.add_argument("--seq-len",         type=int, default=48)
    p.add_argument("--npz-out",   default="data/large_sequences_fast.npz")
    p.add_argument("--seed",            type=int, default=99)
    return p


# ---- sequence builder (identical to build_real_dataset.py) ----------------

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

def _make_sequence(base, gesture, seq_len, rng):
    pose = base.reshape(21,3).copy()
    r = np.deg2rad(25.0)
    pose = pose @ _rot(rng.uniform(-r,r),rng.uniform(-r,r),rng.uniform(-r*.4,r*.4)).T
    pose *= rng.uniform(0.9, 1.10)
    frames = np.tile(pose[np.newaxis], (seq_len,1,1))
    if gesture in ("swipe_left","swipe_right"):
        d = -1.0 if gesture=="swipe_left" else 1.0
        frames[:,0,0] = np.linspace(0, d*rng.uniform(0.25,0.70), seq_len)
    elif gesture=="grab":
        t=np.linspace(0,1,seq_len)
        for j in range(1,21): frames[:,j,:]+=(rng.uniform(0.03,0.12,3)*t[:,None]).astype(np.float32)
    elif gesture=="release":
        t=np.linspace(0,1,seq_len)
        for j in range(1,21): frames[:,j,:]-=(rng.uniform(0.03,0.10,3)*t[:,None]).astype(np.float32)
    elif gesture=="pinch_zoom":
        t=np.linspace(0,1,seq_len)
        for j in [6,7,8,10,11,12,14,15,16,18,19,20]:
            frames[:,j,:]+=(rng.uniform(0.04,0.11,3)*t[:,None]).astype(np.float32)
    tremor = _ou_noise((seq_len,21,3), rng, sigma=0.009)
    tremor[:,0,:]=0.0
    frames+=tremor
    amp=rng.uniform(0.0,0.025)
    axis=rng.uniform(-1,1,3); axis/=np.linalg.norm(axis)+1e-8
    for t in range(seq_len):
        ang=amp*np.sin(np.pi*t/seq_len)
        K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]],dtype=np.float32)
        Rd=np.eye(3,dtype=np.float32)+np.sin(ang)*K+(1-np.cos(ang))*(K@K)
        frames[t]=frames[t]@Rd.T
    return frames.reshape(seq_len,63).astype(np.float32)


# ---------------------------------------------------------------------------

def main():
    args = _build_parser().parse_args()
    rng  = np.random.default_rng(args.seed)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "hand-gesture-research/1.0 (academic; non-commercial)",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    })

    print("[supplement] Initialising MediaPipe…")
    landmarker, mp_mod = _build_landmarker()

    primary_classes = [c for c in sorted(SUPPLEMENT_QUERIES) if c not in DYNAMIC_BASE]

    # ------------------------------------------------------------------
    # Download + detect for under-represented primary classes
    # ------------------------------------------------------------------
    for cls in primary_classes:
        cache_path = cache_dir / f"{cls}.npy"
        coords: list[np.ndarray] = []
        if cache_path.exists():
            coords = list(np.load(cache_path))

        if len(coords) >= args.target_frames:
            print(f"  [{cls}] {len(coords)} frames — already at target, skipping")
            continue

        needed = args.target_frames - len(coords)
        print(f"\n[{cls}] have {len(coords)}, need {needed} more…")

        candidate_urls: list[str] = []
        seen: set[str] = set()
        for q in SUPPLEMENT_QUERIES[cls]:
            for u in _wikimedia_urls(q, session, max_pages=8):
                if u not in seen:
                    seen.add(u); candidate_urls.append(u)
            time.sleep(0.3)
        print(f"  {len(candidate_urls)} unique URLs")

        dl_ok = det = 0
        for url in candidate_urls:
            if len(coords) >= args.target_frames:
                break
            img = _download_image(url, session)
            time.sleep(0.08)
            if img is None: continue
            dl_ok += 1
            c = _extract_coords(landmarker, mp_mod, img)
            if c is not None:
                coords.append(c); det += 1
                if det % 20 == 0:
                    print(f"    {cls}: {len(coords)}/{args.target_frames} (dl={dl_ok} det={det})")

        print(f"  [{cls}] done: {len(coords)} frames (added {det})")
        if coords:
            np.save(cache_path, np.stack(coords))

    landmarker.close()

    # ------------------------------------------------------------------
    # Propagate dynamic classes from their base
    # ------------------------------------------------------------------
    all_coords: dict[str, list[np.ndarray]] = {}
    for cls in sorted(SUPPLEMENT_QUERIES):
        cache_path = cache_dir / f"{cls}.npy"
        if cache_path.exists():
            all_coords[cls] = list(np.load(cache_path))
        else:
            all_coords[cls] = []

    for cls, base in DYNAMIC_BASE.items():
        if not all_coords.get(cls):
            all_coords[cls] = all_coords.get(base, []).copy()
            print(f"  [{cls}] borrowed {len(all_coords[cls])} from {base}")

    # ------------------------------------------------------------------
    # Rebuild sequences + npz
    # ------------------------------------------------------------------
    print("\n[supplement] Rebuilding temporal sequences…")
    all_X: list[np.ndarray] = []
    all_y: list[str]        = []

    for cls in sorted(all_coords):
        frames = all_coords[cls]
        if not frames:
            print(f"  WARNING: still no frames for {cls} — skipping")
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

    # Balance classes
    Xall = np.concatenate(all_X)
    yall = np.array(all_y)
    counts = {c: int((yall==c).sum()) for c in np.unique(yall)}
    min_n  = min(counts.values())
    bal_rng = np.random.default_rng(0)
    bX, by = [], []
    for c in sorted(counts):
        idx = np.where(yall==c)[0]
        if len(idx) > min_n:
            idx = bal_rng.choice(idx, min_n, replace=False)
        bX.append(Xall[idx]); by.extend([c]*len(idx))
    X = np.concatenate(bX); y = np.array(by)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]

    classes_out = sorted(np.unique(y))
    print(f"\nFinal: {len(X):,} sequences × {len(classes_out)} classes  ({min_n} each)")

    np.savez(args.npz_out, X_seq=X, y=y, labels=np.array(classes_out))
    print(f"Saved {args.npz_out}  ✓")


if __name__ == "__main__":
    main()
