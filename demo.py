"""Hand Gesture Recognition — standalone demo.

Runs the webcam with full hand mesh visualization and ST-GCN gesture
recognition.  Designed to be packaged as a single executable:

    pip install pyinstaller
    pyinstaller --onefile --noconsole demo.py

The script auto-discovers the best trained model from the standard
artifacts directory.  If none is found it falls back to mesh-only mode
(no gesture labels).

Controls
--------
  Q / ESC   quit
  S         save a screenshot to ~/Desktop
  M         cycle through available trained models (if multiple exist)
  H         toggle landmark index numbers
"""
from __future__ import annotations

import sys
import os
import time
import argparse
from pathlib import Path
from collections import deque
from typing import Optional

# ---------------------------------------------------------------------------
# Resolve the base directory whether running as script or PyInstaller bundle
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    # Running inside a PyInstaller bundle
    _BASE_DIR = Path(sys._MEIPASS)          # type: ignore[attr-defined]
    _ARTIFACTS_DIR = Path(sys.executable).parent / "artifacts"
else:
    _BASE_DIR = Path(__file__).parent
    _ARTIFACTS_DIR = _BASE_DIR / "artifacts"

# Add src/ to path so hand_gesture_system is importable
_SRC = _BASE_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Lazy imports — give a clean error if dependencies are missing
# ---------------------------------------------------------------------------
def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[demo] Missing dependency: {pkg}")
        print(f"       Install with: pip install {pkg}")
        sys.exit(1)


for _pkg in ("cv2", "numpy", "mediapipe", "torch"):
    _require(_pkg)

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _find_models() -> list[Path]:
    """Return best_model.pt paths sorted by preference (jumbo > large > medium > small)."""
    order = {"jumbo": 0, "large": 1, "medium": 2, "small": 3}
    candidates = sorted(
        _ARTIFACTS_DIR.rglob("best_model.pt"),
        key=lambda p: order.get(p.parent.name.split("_")[-1], 99),
    )
    return candidates


def _load_model(ckpt_path: Path):
    """Load STGCNGestureNet from checkpoint, return (model, labels, use_velocity)."""
    from hand_gesture_system.training.stgcn_model import build_stgcn

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    labels: list[str] = ckpt["labels"]
    model_size: str = ckpt.get("model_size", "medium")
    in_channels: int = ckpt.get("in_channels", 3)
    use_velocity: bool = ckpt.get("use_velocity", False)

    model = build_stgcn(
        num_classes=len(labels),
        in_channels=in_channels,
        model_size=model_size,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, labels, use_velocity, model_size


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

_SEQ_LEN = 48   # frames buffered before making a prediction


def _seq_to_tensor(
    seq: np.ndarray,           # [T, 63]  xyz coords
    use_velocity: bool,
    in_channels: int,
    seq_len: int,
) -> torch.Tensor:
    """Convert coordinate history to [1, C, T, 21] tensor."""
    T = len(seq)
    # Pad / truncate to seq_len
    if T < seq_len:
        pad = np.zeros((seq_len - T, seq.shape[1]), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)
    else:
        seq = seq[-seq_len:]

    coords = seq[:, :63].reshape(seq_len, 21, 3)   # [T, 21, 3]

    if use_velocity and in_channels == 6:
        vel = np.zeros_like(coords)
        vel[1:] = coords[1:] - coords[:-1]
        combined = np.concatenate([coords, vel], axis=-1)  # [T, 21, 6]
        x = combined.transpose(2, 0, 1)[np.newaxis]        # [1, 6, T, 21]
    else:
        x = coords.transpose(2, 0, 1)[np.newaxis]          # [1, 3, T, 21]

    return torch.from_numpy(x.astype(np.float32))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

# Hand graph connections (MediaPipe 21-landmark layout)
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

_TRIANGLES = [
    (0,1,5),(0,5,9),(0,9,13),(0,13,17),
    (1,2,5),(2,5,6),
    (5,6,9),(6,9,10),
    (9,10,13),(10,13,14),
    (13,14,17),(14,17,18),
]

_BONES = [
    (1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
]


def _lm_to_px(lm, w: int, h: int) -> tuple[int, int]:
    return int(np.clip(lm.x * w, 0, w - 1)), int(np.clip(lm.y * h, 0, h - 1))


def _draw_mesh(frame: np.ndarray, landmarks, color=(60, 130, 240), alpha=0.45) -> None:
    h, w = frame.shape[:2]
    pts = [_lm_to_px(lm, w, h) for lm in landmarks]

    wrist = np.array(pts[0], dtype=float)
    palm_size = float(np.linalg.norm(np.array(pts[9], dtype=float) - wrist))
    thick = max(6, int(palm_size * 0.12))
    r = thick // 2

    overlay = frame.copy()
    for a, b, c in _TRIANGLES:
        cv2.fillPoly(overlay, [np.array([pts[a], pts[b], pts[c]], dtype=np.int32)], color)
    for s, e in _BONES:
        cv2.line(overlay, pts[s], pts[e], color, thick, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(overlay, (x, y), r, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for s, e in _CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (200, 230, 255), 1, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (80, 255, 150), -1, cv2.LINE_AA)


def _draw_label(frame: np.ndarray, label: str, score: float, hand_idx: int) -> None:
    h, w = frame.shape[:2]
    bar_w = int(w * 0.30)
    x0 = 20 + hand_idx * (bar_w + 15)
    y0 = h - 90
    # Background pill
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 70), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 70), (80, 80, 80), 1)
    # Confidence bar
    bar_fill = int(bar_w * score)
    cv2.rectangle(frame, (x0 + 4, y0 + 44), (x0 + 4 + bar_fill - 8, y0 + 60),
                  (60, 200, 80), -1)
    # Text
    cv2.putText(frame, label.replace("_", " ").title(),
                (x0 + 8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{score * 100:.0f}%",
                (x0 + 8, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 255, 180), 1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, model_name: str, fps: float, show_idx: bool) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(frame, f"Model: {model_name}   FPS: {fps:.1f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    hint = "Q=quit  S=screenshot  M=next model  H=toggle indices"
    cv2.putText(frame, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def run(camera_index: int = 0) -> None:
    # ---- MediaPipe setup ---------------------------------------------------
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base

        task_path = Path.home() / ".cache" / "hand_gesture_system" / "models" / "hand_landmarker.task"
        if not task_path.exists():
            print("[demo] Downloading MediaPipe hand landmarker model (~8 MB)...")
            import urllib.request
            task_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                task_path,
            )
            print("[demo] Download complete.")

        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(opts)
    except Exception as exc:
        print(f"[demo] Failed to initialise MediaPipe: {exc}")
        sys.exit(1)

    # ---- Model setup -------------------------------------------------------
    models = _find_models()
    model_idx = 0
    gesture_model = None
    labels: list[str] = []
    use_velocity = False
    in_channels = 3
    model_name = "none"

    def _load_current_model():
        nonlocal gesture_model, labels, use_velocity, in_channels, model_name
        if not models:
            model_name = "no model found"
            return
        ckpt = models[model_idx]
        print(f"[demo] Loading {ckpt} ...")
        gesture_model, labels, use_velocity, size = _load_model(ckpt)
        in_channels = 6 if use_velocity else 3
        model_name = f"ST-GCN {size} ({len(labels)} gestures)"
        print(f"[demo] {model_name} ready")

    _load_current_model()

    # ---- Camera ------------------------------------------------------------
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[demo] Cannot open camera {camera_index}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Per-hand coordinate history (keyed by hand index in current frame)
    coord_history: dict[int, deque] = {}
    show_indices = False
    fps_times: deque = deque(maxlen=30)
    frame_ts = 0

    print("[demo] Running. Press Q or ESC to quit.")
    cv2.namedWindow("Hand Gesture Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Gesture Demo", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        t_ms = int(time.time() * 1000)

        # ---- MediaPipe inference -------------------------------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, t_ms)

        # ---- Per-hand processing -------------------------------------------
        for hand_idx, (hand_landmarks, handedness) in enumerate(
            zip(result.hand_landmarks, result.handedness)
        ):
            lms = hand_landmarks

            # Draw full mesh
            _draw_mesh(frame, lms)

            # Optional landmark indices
            if show_indices:
                for i, lm in enumerate(lms):
                    px, py = _lm_to_px(lm, w, h)
                    cv2.putText(frame, str(i), (px + 4, py - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 0), 1)

            # Build coordinate vector (wrist-relative, palm-size normalised)
            if hand_idx not in coord_history:
                coord_history[hand_idx] = deque(maxlen=_SEQ_LEN)

            # Wrist-relative normalisation
            wx, wy, wz = lms[0].x, lms[0].y, lms[0].z
            mid_dist = np.sqrt((lms[9].x - wx)**2 + (lms[9].y - wy)**2 + (lms[9].z - wz)**2)
            scale = max(mid_dist, 1e-6)
            coords = np.array(
                [[(lm.x - wx) / scale, (lm.y - wy) / scale, (lm.z - wz) / scale]
                 for lm in lms],
                dtype=np.float32,
            ).flatten()   # [63]
            coord_history[hand_idx].append(coords)

            # Gesture prediction
            if gesture_model is not None and len(coord_history[hand_idx]) >= 8:
                seq = np.stack(coord_history[hand_idx])  # [T, 63]
                with torch.no_grad():
                    tensor = _seq_to_tensor(seq, use_velocity, in_channels, _SEQ_LEN)
                    logits = gesture_model(tensor)
                    probs = torch.softmax(logits, dim=-1)[0]
                    top_idx = int(probs.argmax())
                    top_score = float(probs[top_idx])

                label = labels[top_idx] if top_idx < len(labels) else "unknown"
                _draw_label(frame, label, top_score, hand_idx)

        # Clear history for hands that disappeared
        active = set(range(len(result.hand_landmarks)))
        for k in list(coord_history.keys()):
            if k not in active:
                del coord_history[k]

        # ---- FPS / HUD -----------------------------------------------------
        fps_times.append(time.time())
        fps = len(fps_times) / max(fps_times[-1] - fps_times[0], 1e-3) if len(fps_times) > 1 else 0
        _draw_hud(frame, model_name, fps, show_indices)

        cv2.imshow("Hand Gesture Demo", frame)

        # ---- Key handling --------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # Q or ESC
            break
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            dest = Path.home() / "Desktop" / f"gesture_demo_{ts}.png"
            cv2.imwrite(str(dest), frame)
            print(f"[demo] Screenshot saved → {dest}")
        elif key == ord("m") and models:
            model_idx = (model_idx + 1) % len(models)
            coord_history.clear()
            _load_current_model()
        elif key == ord("h"):
            show_indices = not show_indices

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition Demo")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    args = parser.parse_args()
    run(camera_index=args.camera)


if __name__ == "__main__":
    main()
