from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hand_gesture_system.config import TemporalConfig
from hand_gesture_system.features.extractor import FeatureExtractor
from hand_gesture_system.features.temporal import TemporalFeatureExtractor
from hand_gesture_system.tracking.mediapipe_tracker import MediaPipeHandTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect temporal gesture samples from webcam and save as .npy sequences"
    )
    parser.add_argument("--label", required=True, help="Class label for this collection run")
    parser.add_argument("--output-dir", default="data/sequences", help="Output dataset root")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument("--window-name", default="Collect Sequence Samples")
    parser.add_argument("--no-mirror", action="store_true")
    return parser


def main() -> int:
    import cv2

    args = build_parser().parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be >= 1")
    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be >= 2")

    output_root = Path(args.output_dir)
    output_label_dir = output_root / args.label
    output_label_dir.mkdir(parents=True, exist_ok=True)

    tracker = MediaPipeHandTracker(max_hands=args.max_hands)
    feature_extractor = FeatureExtractor()
    temporal_extractor = TemporalFeatureExtractor(
        TemporalConfig(sequence_length=args.sequence_length, min_sequence_frames=2)
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    sample_count = 0
    recording = False
    buffer: list[np.ndarray] = []

    try:
        while sample_count < args.samples:
            ok, frame = cap.read()
            if not ok:
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
            if key == ord(" ") and not recording:
                recording = True
                buffer = []

            meshes = tracker.detect(frame)
            if recording and meshes:
                mesh = meshes[0]
                try:
                    coords = feature_extractor.extract_coordinates(mesh)
                    buffer.append(coords)
                except Exception:
                    pass

                if len(buffer) >= args.sequence_length:
                    coordinate_sequence = np.stack(buffer, axis=0)
                    temporal_features = temporal_extractor.extract(
                        coordinate_sequence,
                        force_fixed_length=True,
                    )

                    timestamp = int(time.time() * 1000)
                    sample_path = output_label_dir / f"{args.label}_{timestamp}_{sample_count:04d}.npy"
                    np.save(sample_path, temporal_features.astype(np.float32))

                    sample_count += 1
                    recording = False
                    buffer = []

            status = "RECORDING" if recording else "IDLE"
            hint = "Space:start  q/Esc:quit"
            progress = f"label={args.label}  sample={sample_count}/{args.samples}  status={status}"

            cv2.putText(frame, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, hint, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

            if recording:
                cv2.putText(
                    frame,
                    f"captured frames: {len(buffer)}/{args.sequence_length}",
                    (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (80, 200, 255),
                    2,
                )

            cv2.imshow(args.window_name, frame)

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    print(f"Saved {sample_count} samples to {output_label_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
