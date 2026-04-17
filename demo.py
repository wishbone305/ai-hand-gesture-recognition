"""Hand Gesture Recognition — standalone demo.

Runs the webcam with full hand mesh visualization and ST-GCN gesture
recognition.  Designed to be packaged as a single executable:

    pip install pyinstaller
    pyinstaller demo.spec

Controls
--------
  Q / ESC   quit
  S         save a screenshot to ~/Desktop
  H         toggle landmark index numbers
"""
from __future__ import annotations

import sys
import os
import time
import argparse
from pathlib import Path
from collections import deque

# ---------------------------------------------------------------------------
# Resolve base directory (script vs PyInstaller bundle)
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    _BASE_DIR = Path(sys._MEIPASS)          # type: ignore[attr-defined]
else:
    _BASE_DIR = Path(__file__).parent

_SRC = _BASE_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The bundled small model lives next to the executable / in the bundle
_MODEL_PATH  = _BASE_DIR / "artifacts" / "stgcn_v2" / "stgcn_small" / "best_model.pt"
_LABELS_PATH = _BASE_DIR / "artifacts" / "stgcn_v2" / "stgcn_small" / "labels.txt"


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[demo] Missing: {pkg}  →  pip install {pkg}")
        sys.exit(1)

for _p in ("cv2", "numpy", "mediapipe", "torch"):
    _require(_p)

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path):
    from hand_gesture_system.training.stgcn_model import build_stgcn
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    labels: list[str] = list(ckpt["labels"])
    model_size: str   = ckpt.get("model_size", "small")
    in_channels: int  = ckpt.get("in_channels", 3)
    use_velocity: bool = ckpt.get("use_velocity", False)
    model = build_stgcn(num_classes=len(labels), in_channels=in_channels, model_size=model_size)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, labels, use_velocity, in_channels


# ---------------------------------------------------------------------------
# Coordinate → tensor
# ---------------------------------------------------------------------------
_SEQ_LEN = 48

def _to_tensor(seq: np.ndarray, use_velocity: bool, in_channels: int) -> torch.Tensor:
    T = len(seq)
    if T < _SEQ_LEN:
        pad = np.zeros((_SEQ_LEN - T, 63), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)
    else:
        seq = seq[-_SEQ_LEN:]

    coords = seq.reshape(_SEQ_LEN, 21, 3)   # [T, 21, 3]

    if use_velocity and in_channels == 6:
        vel = np.zeros_like(coords)
        vel[1:] = coords[1:] - coords[:-1]
        x = np.concatenate([coords, vel], axis=-1).transpose(2, 0, 1)[np.newaxis]  # [1,6,T,21]
    else:
        x = coords.transpose(2, 0, 1)[np.newaxis]                                  # [1,3,T,21]

    return torch.from_numpy(x.astype(np.float32))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17),
]
_TRIANGLES = [
    (0,1,5),(0,5,9),(0,9,13),(0,13,17),
    (1,2,5),(2,5,6),(5,6,9),(6,9,10),(9,10,13),(10,13,14),(13,14,17),(14,17,18),
]
_BONES = [
    (1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
]


def _lm_px(lm, w: int, h: int) -> tuple[int, int]:
    return int(np.clip(lm.x * w, 0, w-1)), int(np.clip(lm.y * h, 0, h-1))


def _draw_mesh(frame: np.ndarray, lms, color=(60, 130, 240), alpha=0.45) -> None:
    h, w = frame.shape[:2]
    pts = [_lm_px(lm, w, h) for lm in lms]
    palm_sz = float(np.linalg.norm(np.array(pts[9]) - np.array(pts[0])))
    thick = max(6, int(palm_sz * 0.12))
    r = max(3, thick // 2)

    ov = frame.copy()
    for a, b, c in _TRIANGLES:
        cv2.fillPoly(ov, [np.array([pts[a], pts[b], pts[c]], dtype=np.int32)], color)
    for s, e in _BONES:
        cv2.line(ov, pts[s], pts[e], color, thick, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(ov, (x, y), r, color, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    for s, e in _CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (200, 230, 255), 1, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (80, 255, 150), -1, cv2.LINE_AA)


def _draw_gesture(frame: np.ndarray, label: str, score: float, hand_idx: int,
                  wrist_px: tuple[int, int]) -> None:
    """Draw gesture name prominently above the wrist."""
    h, w = frame.shape[:2]
    text = label.replace("_", " ").upper()
    pct  = f"{score*100:.0f}%"

    # Choose font scale based on frame width
    fs_big  = max(0.9, w / 900)
    fs_small = fs_big * 0.55

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs_big, 2)
    (pw, ph), _ = cv2.getTextSize(pct,  cv2.FONT_HERSHEY_SIMPLEX, fs_small, 1)

    # Anchor above wrist, clamped to frame
    wx, wy = wrist_px
    x0 = int(np.clip(wx - tw // 2, 4, w - tw - 4))
    y0 = int(np.clip(wy - 30, th + 12, h - ph - 12))

    # Shadow
    cv2.putText(frame, text, (x0+2, y0+2),
                cv2.FONT_HERSHEY_DUPLEX, fs_big, (0, 0, 0), 3, cv2.LINE_AA)
    # Main label — colour shifts green→yellow→red with confidence
    r = int(255 * (1 - score))
    g = int(255 * score)
    cv2.putText(frame, text, (x0, y0),
                cv2.FONT_HERSHEY_DUPLEX, fs_big, (r, g, 60), 2, cv2.LINE_AA)

    # Confidence percentage just below
    px0 = int(np.clip(wx - pw // 2, 4, w - pw - 4))
    cv2.putText(frame, pct, (px0, y0 + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, fs_small, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, fps: float) -> None:
    h, w = frame.shape[:2]
    bar_h = 30
    cv2.rectangle(frame, (0, 0), (w, bar_h), (15, 15, 15), -1)
    cv2.putText(frame, f"ST-GCN 1M  |  FPS {fps:.0f}  |  Q=quit  S=screenshot  H=indices",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(camera_index: int = 0) -> None:
    # ---- MediaPipe ---------------------------------------------------------
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mpv
    from mediapipe.tasks.python.core import base_options as mpb

    task_path = Path.home() / ".cache" / "hand_gesture_system" / "models" / "hand_landmarker.task"
    if not task_path.exists():
        print("[demo] Downloading MediaPipe model (~8 MB)…")
        import urllib.request
        task_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            task_path,
        )

    landmarker = mpv.HandLandmarker.create_from_options(
        mpv.HandLandmarkerOptions(
            base_options=mpb.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mpv.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    # ---- Gesture model -----------------------------------------------------
    gesture_model = labels = None
    use_velocity = False
    in_channels  = 3

    if _MODEL_PATH.exists():
        print(f"[demo] Loading model from {_MODEL_PATH} …")
        gesture_model, labels, use_velocity, in_channels = _load_model(_MODEL_PATH)
        print(f"[demo] Ready — {len(labels)} gestures: {', '.join(labels)}")
    else:
        print(f"[demo] No model found at {_MODEL_PATH} — running mesh-only mode")

    # ---- Camera ------------------------------------------------------------
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[demo] Cannot open camera {camera_index}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    coord_hist: dict[int, deque] = {}
    show_idx = False
    fps_q: deque = deque(maxlen=30)

    cv2.namedWindow("Hand Gesture Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Gesture Demo", 1280, 720)
    print("[demo] Running. Press Q or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        for i, lms in enumerate(result.hand_landmarks):
            # Mesh
            _draw_mesh(frame, lms)

            # Optional joint indices
            if show_idx:
                for j, lm in enumerate(lms):
                    px, py = _lm_px(lm, w, h)
                    cv2.putText(frame, str(j), (px+3, py-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 0), 1)

            # Coordinate extraction (wrist-relative, palm-size normalised)
            if i not in coord_hist:
                coord_hist[i] = deque(maxlen=_SEQ_LEN)
            wx, wy, wz = lms[0].x, lms[0].y, lms[0].z
            scale = max(np.sqrt((lms[9].x-wx)**2+(lms[9].y-wy)**2+(lms[9].z-wz)**2), 1e-6)
            coords = np.array([
                [(lm.x-wx)/scale, (lm.y-wy)/scale, (lm.z-wz)/scale] for lm in lms
            ], dtype=np.float32).flatten()
            coord_hist[i].append(coords)

            # Gesture prediction
            if gesture_model is not None and len(coord_hist[i]) >= 8:
                seq = np.stack(coord_hist[i])
                with torch.no_grad():
                    logits = gesture_model(_to_tensor(seq, use_velocity, in_channels))
                    probs  = torch.softmax(logits, dim=-1)[0]
                    top    = int(probs.argmax())
                    score  = float(probs[top])

                wrist_px = _lm_px(lms[0], w, h)
                _draw_gesture(frame, labels[top], score, i, wrist_px)

        # Clean up disappeared hands
        for k in list(coord_hist):
            if k >= len(result.hand_landmarks):
                del coord_hist[k]

        # FPS
        fps_q.append(time.time())
        fps = (len(fps_q)-1) / max(fps_q[-1]-fps_q[0], 1e-3) if len(fps_q) > 1 else 0
        _draw_hud(frame, fps)

        cv2.imshow("Hand Gesture Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            p = Path.home() / "Desktop" / f"gesture_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(p), frame)
            print(f"[demo] Saved → {p}")
        elif key == ord("h"):
            show_idx = not show_idx

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand Gesture Recognition Demo")
    ap.add_argument("--camera", type=int, default=0)
    run(camera_index=ap.parse_args().camera)


if __name__ == "__main__":
    main()
