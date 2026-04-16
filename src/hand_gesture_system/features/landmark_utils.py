from __future__ import annotations

import math

import numpy as np

from hand_gesture_system.config import RuleConfig
from hand_gesture_system.types import HandMesh, Landmark

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGER_TIPS = {
    "thumb": THUMB_TIP,
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "ring": RING_TIP,
    "pinky": PINKY_TIP,
}


def distance(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def palm_size(mesh: HandMesh) -> float:
    wrist = mesh.landmarks[WRIST]
    middle_mcp = mesh.landmarks[MIDDLE_MCP]
    size = distance(wrist, middle_mcp)
    return max(size, 1e-6)


def _is_finger_extended(mesh: HandMesh, pip_idx: int, tip_idx: int, cfg: RuleConfig) -> bool:
    wrist = mesh.landmarks[WRIST]
    pip = mesh.landmarks[pip_idx]
    tip = mesh.landmarks[tip_idx]
    return distance(tip, wrist) > (distance(pip, wrist) * cfg.extension_ratio)


def _is_thumb_extended(mesh: HandMesh, cfg: RuleConfig) -> bool:
    thumb_tip = mesh.landmarks[THUMB_TIP]
    thumb_ip = mesh.landmarks[THUMB_IP]
    index_mcp = mesh.landmarks[INDEX_MCP]
    margin = palm_size(mesh) * cfg.thumb_margin

    if mesh.handedness == "right":
        return thumb_tip.x > (thumb_ip.x + margin)
    if mesh.handedness == "left":
        return thumb_tip.x < (thumb_ip.x - margin)

    # Fallback for unknown handedness: compare separation from index base.
    return distance(thumb_tip, index_mcp) > distance(thumb_ip, index_mcp)


def finger_states(mesh: HandMesh, cfg: RuleConfig | None = None) -> dict[str, bool]:
    cfg = cfg or RuleConfig()
    return {
        "thumb": _is_thumb_extended(mesh, cfg),
        "index": _is_finger_extended(mesh, INDEX_PIP, INDEX_TIP, cfg),
        "middle": _is_finger_extended(mesh, MIDDLE_PIP, MIDDLE_TIP, cfg),
        "ring": _is_finger_extended(mesh, RING_PIP, RING_TIP, cfg),
        "pinky": _is_finger_extended(mesh, PINKY_PIP, PINKY_TIP, cfg),
    }


def finger_count(states: dict[str, bool]) -> int:
    return sum(1 for extended in states.values() if extended)


def pinch_distance(mesh: HandMesh) -> float:
    thumb_tip = mesh.landmarks[THUMB_TIP]
    index_tip = mesh.landmarks[INDEX_TIP]
    return distance(thumb_tip, index_tip) / palm_size(mesh)


def normalized_landmark_vector(mesh: HandMesh) -> np.ndarray:
    wrist = mesh.landmarks[WRIST]
    scale = palm_size(mesh)
    vector: list[float] = []

    for point in mesh.landmarks:
        vector.extend(
            [
                (point.x - wrist.x) / scale,
                (point.y - wrist.y) / scale,
                (point.z - wrist.z) / scale,
            ]
        )

    return np.array(vector, dtype=np.float32)
