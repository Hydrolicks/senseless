"""Stick-figure geometry for the Speech-mode signing figure (pure, no Tk).

One 153-value frame (normalized: shoulder-midpoint origin, shoulder-width units,
y down) becomes line segments and dots in canvas coordinates. A fixed view
window (``UI.figure_extent``) is fitted into the box with one uniform scale and
centred, so the figure doesn't jump in size between frames or words. Hands that
are absent (all-zero blocks) are not drawn; the face is not in the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN, UI

# Pose subset order: nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip.
POSE_EDGES = ((1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8))
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)  # fmt: skip
HEAD_RADIUS = 0.22  # shoulder widths


@dataclass(frozen=True)
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str  # "body" | "left" | "right"


@dataclass(frozen=True)
class Dot:
    x: float
    y: float
    r: float
    kind: str  # "head" | "joint"


def figure_geometry(
    vec: np.ndarray,
    box: tuple[float, float, float, float],
    extent: tuple[float, float, float, float] = UI.figure_extent,
) -> tuple[list[Segment], list[Dot]]:
    pose = np.asarray(vec[ls.POSE_START : ls.POSE_END], dtype=np.float64).reshape(-1, 3)
    if not np.any(pose):
        return [], []
    bx, by, bw, bh = box
    x_min, x_max, y_min, y_max = extent
    scale = min(bw / (x_max - x_min), bh / (y_max - y_min))
    ox = bx + bw / 2 - scale * (x_min + x_max) / 2
    oy = by + bh / 2 - scale * (y_min + y_max) / 2

    def to_canvas(p: np.ndarray) -> tuple[float, float]:
        x = float(np.clip(p[0], x_min, x_max))
        y = float(np.clip(p[1], y_min, y_max))
        return ox + scale * x, oy + scale * y

    segments = [Segment(*to_canvas(pose[a]), *to_canvas(pose[b]), "body") for a, b in POSE_EDGES]
    dots = [Dot(*to_canvas(pose[0]), HEAD_RADIUS * scale, "head")]
    dots += [Dot(*to_canvas(pose[i]), 3.0, "joint") for i in (1, 2, 3, 4)]
    for start, end, kind in (
        (ls.LEFT_HAND_START, ls.LEFT_HAND_END, "left"),
        (ls.RIGHT_HAND_START, ls.RIGHT_HAND_END, "right"),
    ):
        hand = np.asarray(vec[start:end], dtype=np.float64).reshape(-1, 3)
        if np.any(hand):
            segments += [
                Segment(*to_canvas(hand[a]), *to_canvas(hand[b]), kind) for a, b in HAND_EDGES
            ]
    return segments, dots


def frame_index(
    elapsed_s: float, n_frames: int = SIGN.window_length, fps: float = SIGN.reference_fps
) -> int | None:
    """Frame to show ``elapsed_s`` into a take, or None once the take is over."""
    index = int(elapsed_s * fps + 1e-9)
    return index if index < n_frames else None


def rest_frame(library: dict[str, np.ndarray]) -> np.ndarray | None:
    """A neutral pose for the idle figure: the first word's first frame, hands removed."""
    if not library:
        return None
    frame = np.array(library[sorted(library)[0]][0], dtype=np.float32, copy=True)
    frame[ls.LEFT_HAND_START : ls.RIGHT_HAND_END] = 0.0
    return frame
