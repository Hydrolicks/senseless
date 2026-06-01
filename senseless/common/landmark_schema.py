"""Per-frame landmark feature-vector schema for Senseless.

Defined ONCE here so capture (Pi), training (Colab), and on-Pi inference all
agree on the exact layout and dimensionality. The FACE MESH IS EXCLUDED:
features are the two hands (MediaPipe Hand landmarks, 21 each) plus a curated
upper-body subset of the MediaPipe Pose landmarks.

Coordinate convention
---------------------
Each landmark contributes ``(x, y, z)`` in MediaPipe's normalized image
coordinates: ``x`` and ``y`` in ``[0, 1]`` relative to image width/height, and
``z`` on roughly the same scale as ``x`` (smaller = closer to the camera). Pose
``visibility`` is intentionally dropped so every landmark is uniformly 3-D.
Missing landmarks (e.g. a hand not detected in a frame) are filled with zeros.

Flat feature vector layout (one frame)
--------------------------------------
::

    [  0 :  63)  left hand   : 21 landmarks x (x, y, z)
    [ 63 : 126)  right hand  : 21 landmarks x (x, y, z)
    [126 : 153)  pose subset :  9 landmarks x (x, y, z)
    --------------------------------------------------------
    FEATURE_DIM = 153

Within each block, landmark ``i`` occupies ``[i*3 : i*3 + 3] = (x, y, z)``.
A model input window is therefore shaped ``(window_length, FEATURE_DIM)`` --
see ``config.SIGN.window_length``.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Hands (MediaPipe Hand landmark model: 21 points per hand) -------------
NUM_HAND_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # (x, y, z)
HAND_DIM = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK  # 63 per hand

# --- Pose subset (MediaPipe Pose has 33 landmarks; we keep upper body) -----
# Indices below refer to the MediaPipe Pose landmark numbering. Excluded on
# purpose: face points 1-10 (eyes/ears/mouth -- face mesh is excluded), the
# hand-on-pose points 17-22 (redundant with the dedicated hand model), and the
# lower body 25-32. Hips (23, 24) are kept as a stable torso reference useful
# for translation/scale normalization. This subset is a tunable design choice.
POSE_LANDMARKS: tuple[tuple[int, str], ...] = (
    (0, "nose"),
    (11, "left_shoulder"),
    (12, "right_shoulder"),
    (13, "left_elbow"),
    (14, "right_elbow"),
    (15, "left_wrist"),
    (16, "right_wrist"),
    (23, "left_hip"),
    (24, "right_hip"),
)
POSE_LANDMARK_INDICES: tuple[int, ...] = tuple(idx for idx, _ in POSE_LANDMARKS)
POSE_LANDMARK_NAMES: tuple[str, ...] = tuple(name for _, name in POSE_LANDMARKS)
NUM_POSE_LANDMARKS = len(POSE_LANDMARKS)  # 9
POSE_DIM = NUM_POSE_LANDMARKS * COORDS_PER_LANDMARK  # 27

# --- Flat feature-vector block offsets -------------------------------------
LEFT_HAND_START = 0
LEFT_HAND_END = LEFT_HAND_START + HAND_DIM  # 63
RIGHT_HAND_START = LEFT_HAND_END  # 63
RIGHT_HAND_END = RIGHT_HAND_START + HAND_DIM  # 126
POSE_START = RIGHT_HAND_END  # 126
POSE_END = POSE_START + POSE_DIM  # 153

#: Length of one per-frame feature vector.
FEATURE_DIM = POSE_END  # 153


@dataclass(frozen=True)
class FeatureBlock:
    """A named contiguous slice of the flat per-frame feature vector."""

    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


#: The feature vector in order; concatenating these blocks yields FEATURE_DIM.
FEATURE_BLOCKS: tuple[FeatureBlock, ...] = (
    FeatureBlock("left_hand", LEFT_HAND_START, LEFT_HAND_END),
    FeatureBlock("right_hand", RIGHT_HAND_START, RIGHT_HAND_END),
    FeatureBlock("pose", POSE_START, POSE_END),
)


def describe() -> str:
    """Return a human-readable summary of the schema (for logging/debugging)."""
    lines = [f"FEATURE_DIM = {FEATURE_DIM}"]
    for block in FEATURE_BLOCKS:
        lines.append(f"  [{block.start:>3}:{block.end:>3}) {block.name:<11} ({block.size} dims)")
    lines.append(f"  pose indices (MediaPipe Pose): {list(POSE_LANDMARK_INDICES)}")
    return "\n".join(lines)
