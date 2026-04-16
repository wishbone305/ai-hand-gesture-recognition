from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass
class HandMesh:
    landmarks: list[Landmark]
    handedness: str = "unknown"
    confidence: float = 0.0


@dataclass
class GesturePrediction:
    label: str
    score: float
    source: str


@dataclass
class HandResult:
    hand_id: str
    mesh: HandMesh
    predictions: list[GesturePrediction] = field(default_factory=list)
    feature_vector: np.ndarray | None = None
    sequence_feature_vector: np.ndarray | None = None


@dataclass
class FrameResult:
    hands: list[HandResult] = field(default_factory=list)
