from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hand_gesture_system.training.dataio import encode_labels, load_sequence_dataset, save_labels_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export sequence dataset directory to compact .npz format"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--labels-out", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    x, y_text = load_sequence_dataset(
        path=args.dataset_dir,
        sequence_length=args.sequence_length,
    )
    y = np.array(y_text, dtype="<U64")

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X_seq=x.astype(np.float32), y=y)

    _, labels, _ = encode_labels(y_text)
    if args.labels_out:
        save_labels_file(labels, args.labels_out)

    print(f"Exported dataset: {output_path}")
    print(f"samples={x.shape[0]} frames={x.shape[1]} features={x.shape[2]} classes={len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
