"""Data augmentation transforms for hand landmark sequences.

All functions operate on arrays of shape [T, 63]  (frames × flattened xyz)
and return a new array of the same shape.  The coordinates are assumed to be
in the wrist-relative, palm-size-normalised format produced by
FeatureExtractor.extract_coordinates().

Functions
---------
rotate_sequence       — random 3-D rotation of all joints
scale_sequence        — uniform scale (palm-size perturbation)
translate_sequence    — constant x/y/z offset on every frame
noise_sequence        — i.i.d. Gaussian noise per joint per frame
time_warp             — non-uniform time interpolation (speed variation)
mirror_sequence       — flip x-axis (left↔right hand)
joint_dropout         — zero out random joints for random spans
gaussian_blur_time    — smooth along time axis (slow-motion effect)
random_augment        — randomly combine several transforms

combine
-------
augment_dataset       — expand an (X, y) dataset by N augmented copies
"""
from __future__ import annotations

import numpy as np

_NUM_JOINTS = 21
_COORD_DIM  = 63   # 21 × 3


# ---------------------------------------------------------------------------
# Low-level rotation helpers
# ---------------------------------------------------------------------------

def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)


def _reshape_in(seq: np.ndarray) -> np.ndarray:
    """[T, 63] → [T, 21, 3]"""
    return seq.reshape(len(seq), _NUM_JOINTS, 3)

def _reshape_out(joints: np.ndarray) -> np.ndarray:
    """[T, 21, 3] → [T, 63]"""
    return joints.reshape(len(joints), _COORD_DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def rotate_sequence(
    seq: np.ndarray,
    rng: np.random.Generator,
    max_deg: float = 40.0,
) -> np.ndarray:
    """Rotate the entire sequence by a random amount on each axis."""
    r = np.deg2rad(max_deg)
    ax, ay, az = rng.uniform(-r, r), rng.uniform(-r, r), rng.uniform(-r*0.4, r*0.4)
    R = _rot_x(ax) @ _rot_y(ay) @ _rot_z(az)
    joints = _reshape_in(seq) @ R.T
    return _reshape_out(joints)


def scale_sequence(
    seq: np.ndarray,
    rng: np.random.Generator,
    lo: float = 0.80,
    hi: float = 1.20,
) -> np.ndarray:
    """Uniformly scale all coordinates (simulates different hand sizes)."""
    factor = rng.uniform(lo, hi)
    return (seq * factor).astype(np.float32)


def translate_sequence(
    seq: np.ndarray,
    rng: np.random.Generator,
    max_delta: float = 0.08,
) -> np.ndarray:
    """Add a random constant offset to all joints on every frame."""
    delta = rng.uniform(-max_delta, max_delta, size=3).astype(np.float32)
    joints = _reshape_in(seq) + delta[np.newaxis, np.newaxis, :]
    return _reshape_out(joints)


def noise_sequence(
    seq: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.018,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise to all coordinates."""
    noise = rng.standard_normal(seq.shape).astype(np.float32) * sigma
    return (seq + noise).astype(np.float32)


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Flip x-axis — converts right-hand gesture to left-hand and vice versa."""
    joints = _reshape_in(seq).copy()
    joints[:, :, 0] *= -1.0
    return _reshape_out(joints)


def time_warp(
    seq: np.ndarray,
    rng: np.random.Generator,
    max_speed: float = 0.35,
) -> np.ndarray:
    """Non-uniform time stretching: randomly compress or expand regions."""
    T = len(seq)
    # Generate a smooth random time map via cumsum of perturbed uniform steps
    steps = np.ones(T, dtype=np.float32)
    n_bumps = rng.integers(1, 4)
    for _ in range(n_bumps):
        center = rng.uniform(0.1, 0.9) * T
        width  = rng.uniform(0.05, 0.25) * T
        amp    = rng.uniform(-max_speed, max_speed)
        t_idx  = np.arange(T, dtype=np.float32)
        bump   = amp * np.exp(-0.5 * ((t_idx - center) / width) ** 2)
        steps  += bump
    steps = np.clip(steps, 0.2, 2.5)
    t_src = np.cumsum(steps)
    t_src = (t_src - t_src[0]) / (t_src[-1] - t_src[0]) * (T - 1)   # remap to [0, T-1]
    t_tgt = np.arange(T, dtype=np.float32)
    warped = np.zeros_like(seq)
    for feat in range(seq.shape[1]):
        warped[:, feat] = np.interp(t_tgt, t_src, seq[:, feat])
    return warped.astype(np.float32)


def joint_dropout(
    seq: np.ndarray,
    rng: np.random.Generator,
    p_joint: float = 0.05,
    p_frame: float = 0.10,
) -> np.ndarray:
    """Zero out random non-wrist joints on random frames (simulates occlusion).

    Joint 0 (wrist) is always excluded from dropout: it is the coordinate
    origin in wrist-relative data (always 0,0,0), so dropping it is a no-op
    and including it in the mask distorts frame-level statistics.
    """
    T = len(seq)
    result = seq.copy()
    joints = _reshape_in(result)  # [T, 21, 3]

    # Per-frame dropout: zero all non-wrist joints on selected frames
    frame_mask = rng.random(T) < p_frame
    if frame_mask.any():
        joints[frame_mask, 1:, :] = 0.0  # keep wrist (joint 0) intact

    # Per-joint dropout across all frames (joints 1-20 only)
    joint_candidates = np.arange(1, _NUM_JOINTS)  # exclude wrist
    joint_mask = rng.random(len(joint_candidates)) < p_joint
    if joint_mask.any():
        drop_indices = joint_candidates[joint_mask]
        joints[:, drop_indices, :] = 0.0

    return _reshape_out(joints)


def gaussian_blur_time(
    seq: np.ndarray,
    sigma: float = 1.2,
) -> np.ndarray:
    """Gaussian smooth along the time axis — simulates slow/blurred motion."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(seq, sigma=sigma, axis=0).astype(np.float32)


def random_perspective(
    seq: np.ndarray,
    rng: np.random.Generator,
    max_deg: float = 15.0,
) -> np.ndarray:
    """Apply a small additional random rotation to simulate camera viewpoint change."""
    return rotate_sequence(seq, rng, max_deg=max_deg)


# ---------------------------------------------------------------------------
# Combined random augmentation
# ---------------------------------------------------------------------------

def random_augment(
    seq: np.ndarray,
    rng: np.random.Generator,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply a random combination of augmentations.

    strength: 0=very mild, 1=default, >1=aggressive
    """
    out = seq.copy()

    # Always rotate and scale
    out = rotate_sequence(out, rng, max_deg=40.0 * strength)
    out = scale_sequence(out, rng, lo=max(0.6, 1.0 - 0.2 * strength), hi=min(1.4, 1.0 + 0.2 * strength))

    # 80% chance of noise
    if rng.uniform() < 0.8:
        sigma = rng.uniform(0.005, 0.025) * strength
        out = noise_sequence(out, rng, sigma=sigma)

    # 70% chance of time warp
    if rng.uniform() < 0.7:
        out = time_warp(out, rng, max_speed=0.3 * strength)

    # 50% chance of translation
    if rng.uniform() < 0.5:
        out = translate_sequence(out, rng, max_delta=0.07 * strength)

    # 30% chance of mirror
    if rng.uniform() < 0.3:
        out = mirror_sequence(out)

    # 20% chance of joint dropout (low strength to not destroy too much info)
    if rng.uniform() < 0.2:
        out = joint_dropout(out, rng, p_joint=0.04, p_frame=0.06)

    # 30% chance of temporal smoothing
    if rng.uniform() < 0.3:
        sigma = rng.uniform(0.5, 1.5)
        try:
            out = gaussian_blur_time(out, sigma=sigma)
        except ImportError:
            pass  # scipy not available, skip

    return out


# ---------------------------------------------------------------------------
# Dataset-level augmentation
# ---------------------------------------------------------------------------

def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_augments: int,
    rng: np.random.Generator | None = None,
    strength: float = 1.0,
    include_mirror: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand (X, y) by generating n_augments copies of every sample.

    Args:
        X: [N, T, 63] float32 sequences
        y: [N] int labels
        n_augments: number of augmented copies per original sample
        rng: random generator (created if None)
        strength: augmentation intensity (1.0 = default)
        include_mirror: always add a mirrored copy of every sample

    Returns:
        (X_aug, y_aug): concatenated originals + augmented copies
    """
    if rng is None:
        rng = np.random.default_rng()

    aug_seqs = [X]
    aug_labels = [y]

    if include_mirror:
        mirrored = np.stack([mirror_sequence(X[i]) for i in range(len(X))])
        aug_seqs.append(mirrored)
        aug_labels.append(y)

    for _ in range(n_augments):
        batch = np.stack([random_augment(X[i], rng, strength=strength) for i in range(len(X))])
        aug_seqs.append(batch)
        aug_labels.append(y)

    return np.concatenate(aug_seqs, axis=0), np.concatenate(aug_labels, axis=0)
