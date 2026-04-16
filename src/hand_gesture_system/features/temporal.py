from __future__ import annotations

import numpy as np

from hand_gesture_system.config import TemporalConfig


def _ensure_2d(sequence: np.ndarray) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"Expected sequence with shape [frames, features], got {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError("Temporal sequence is empty")
    return array


def to_fixed_length(sequence: np.ndarray, length: int) -> np.ndarray:
    sequence = _ensure_2d(sequence)

    if sequence.shape[0] >= length:
        return sequence[-length:]

    pad_count = length - sequence.shape[0]
    pad = np.repeat(sequence[:1], repeats=pad_count, axis=0)
    return np.concatenate([pad, sequence], axis=0)


def temporal_channels(
    sequence: np.ndarray,
    include_velocity: bool,
    include_acceleration: bool,
) -> np.ndarray:
    sequence = _ensure_2d(sequence)
    channels = [sequence]

    if include_velocity:
        velocity = np.diff(sequence, axis=0, prepend=sequence[:1])
        channels.append(velocity)
    if include_acceleration:
        acceleration = np.diff(sequence, axis=0, n=2, prepend=sequence[:1], append=sequence[-1:])
        channels.append(acceleration)

    return np.concatenate(channels, axis=1).astype(np.float32)


class TemporalFeatureExtractor:
    def __init__(self, cfg: TemporalConfig | None = None) -> None:
        self.cfg = cfg or TemporalConfig()

    def extract(
        self,
        coordinate_sequence: np.ndarray,
        force_fixed_length: bool = False,
    ) -> np.ndarray:
        sequence = _ensure_2d(coordinate_sequence)
        if force_fixed_length:
            sequence = to_fixed_length(sequence, self.cfg.sequence_length)

        return temporal_channels(
            sequence=sequence,
            include_velocity=self.cfg.include_velocity,
            include_acceleration=self.cfg.include_acceleration,
        )
