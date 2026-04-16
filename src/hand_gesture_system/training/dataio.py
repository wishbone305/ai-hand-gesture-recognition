from __future__ import annotations

from pathlib import Path

import numpy as np

from hand_gesture_system.features.temporal import to_fixed_length


def load_sequence_dataset(
    path: str | Path,
    x_key: str = "X_seq",
    y_key: str = "y",
    sequence_length: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".npz":
        return _load_from_npz(dataset_path, x_key=x_key, y_key=y_key)

    if dataset_path.is_dir():
        return _load_from_directory(dataset_path, sequence_length=sequence_length)

    raise ValueError(
        f"Unsupported dataset path '{dataset_path}'. Use .npz file or directory."
    )


def _load_from_npz(path: Path, x_key: str, y_key: str) -> tuple[np.ndarray, list[str]]:
    dataset = np.load(path, allow_pickle=False)
    if x_key not in dataset or y_key not in dataset:
        raise KeyError(
            f"Dataset must contain keys '{x_key}' and '{y_key}'. "
            f"Available keys: {list(dataset.keys())}"
        )

    x = np.asarray(dataset[x_key], dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected X tensor as [samples, frames, features], got {x.shape}")

    y = [str(item) for item in dataset[y_key]]
    if len(y) != x.shape[0]:
        raise ValueError(f"X/Y sample mismatch: {x.shape[0]} vs {len(y)}")

    return x, y


def _load_from_directory(
    root: Path,
    sequence_length: int | None,
) -> tuple[np.ndarray, list[str]]:
    label_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
    if not label_dirs:
        raise ValueError(
            f"No class subdirectories found in dataset directory: {root}"
        )

    sequences: list[np.ndarray] = []
    labels: list[str] = []

    for label_dir in label_dirs:
        label = label_dir.name
        files = sorted(label_dir.glob("*.npy"))
        if not files:
            continue

        for file_path in files:
            sequence = np.load(file_path)
            if sequence.ndim != 2:
                raise ValueError(
                    f"Expected sequence array [frames, features] at {file_path}, got {sequence.shape}"
                )

            sequence = sequence.astype(np.float32)
            if sequence_length is not None:
                sequence = to_fixed_length(sequence, sequence_length)

            sequences.append(sequence)
            labels.append(label)

    if not sequences:
        raise ValueError(f"No .npy sequence files found under {root}")

    feature_dim = sequences[0].shape[1]
    frame_count = sequences[0].shape[0]

    for index, sequence in enumerate(sequences):
        if sequence.shape[1] != feature_dim:
            raise ValueError(
                f"Feature dimension mismatch in sample {index}: expected {feature_dim}, got {sequence.shape[1]}"
            )
        if sequence.shape[0] != frame_count:
            raise ValueError(
                "Frame length mismatch detected. "
                "Pass --sequence-length during training/evaluation to auto-pad/truncate."
            )

    x = np.stack(sequences, axis=0)
    return x.astype(np.float32), labels


def encode_labels(labels: list[str]) -> tuple[np.ndarray, list[str], dict[str, int]]:
    classes = sorted(set(labels))
    mapping = {label: idx for idx, label in enumerate(classes)}
    encoded = np.array([mapping[label] for label in labels], dtype=np.int64)
    return encoded, classes, mapping


def stratified_split_indices(
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_id in sorted(set(int(item) for item in y.tolist())):
        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)

        val_count = max(1, int(round(len(class_indices) * val_ratio)))
        if val_count >= len(class_indices):
            val_count = max(1, len(class_indices) - 1)

        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    if not train_indices or not val_indices:
        raise ValueError(
            "Could not create non-empty train/val split. Add more samples per class."
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return np.array(train_indices, dtype=np.int64), np.array(val_indices, dtype=np.int64)


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[str],
) -> dict[str, object]:
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_pred_array = np.asarray(y_pred, dtype=np.int64)

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            f"y_true/y_pred shape mismatch: {y_true_array.shape} vs {y_pred_array.shape}"
        )

    per_class: list[dict[str, float | int | str]] = []
    for class_index, class_label in enumerate(labels):
        tp = int(np.sum((y_true_array == class_index) & (y_pred_array == class_index)))
        fp = int(np.sum((y_true_array != class_index) & (y_pred_array == class_index)))
        fn = int(np.sum((y_true_array == class_index) & (y_pred_array != class_index)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class.append(
            {
                "label": class_label,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(np.sum(y_true_array == class_index)),
            }
        )

    accuracy = float(np.mean(y_true_array == y_pred_array))
    macro_precision = float(np.mean([row["precision"] for row in per_class]))
    macro_recall = float(np.mean([row["recall"] for row in per_class]))
    macro_f1 = float(np.mean([row["f1"] for row in per_class]))

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def save_labels_file(labels: list[str], path: str | Path) -> None:
    Path(path).write_text("\n".join(labels) + "\n")
