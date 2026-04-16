from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuleConfig:
    thumb_margin: float = 0.04
    extension_ratio: float = 1.08
    ok_pinch_threshold: float = 0.45


@dataclass
class TemporalConfig:
    sequence_length: int = 32
    min_sequence_frames: int = 8
    include_velocity: bool = True
    include_acceleration: bool = True


@dataclass
class SmoothingConfig:
    enabled: bool = True
    window_size: int = 8
    min_consecutive_frames: int = 3
    stable_score: float = 0.9


@dataclass
class ConfusionFixConfig:
    enabled: bool = True


@dataclass
class DTWConfig:
    distance_threshold: float = 0.35
    score_scale: float = 6.0
    top_k: int = 3


@dataclass
class PipelineConfig:
    max_hands: int = 2
    top_k_predictions: int = 3
    include_rules: bool = True
    hand_assignment_distance_threshold: float = 0.25
    max_missed_frames: int = 12
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    confusion_fix: ConfusionFixConfig = field(default_factory=ConfusionFixConfig)
    dtw: DTWConfig = field(default_factory=DTWConfig)
