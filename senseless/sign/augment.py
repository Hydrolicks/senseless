"""Hand-landmark perturbation for training: tolerate a different hand model.

The Pi runs MediaPipe's lite hand model (95 ms/frame on a Pi 4 vs 265 ms for the
full model), but the training data was recorded with the full Tasks model. The lite
model's finger positions differ by a few percent of the hand's size, which the
classifier was never shown. The recordings store landmarks, not video, so they
can't be re-run through the lite model; instead, training copies of each window
get a plausible "different model" version of both hands:

* one small rotation (in the image plane) and scale per hand per window, pivoting
  on the wrist, so the hand's position relative to the body is kept;
* one fixed offset per landmark per window (a systematic per-model bias);
* small per-frame jitter.

Units are the normalized feature space (shoulder widths); a hand spans ~0.55 there,
so the defaults stay within a few percent of the hand's size. Absent hands (all
zero) stay absent, and the pose block is untouched.
"""

from __future__ import annotations

import numpy as np

from senseless.common import landmark_schema as ls

_HANDS = (
    slice(ls.LEFT_HAND_START, ls.LEFT_HAND_END),
    slice(ls.RIGHT_HAND_START, ls.RIGHT_HAND_END),
)


def perturb_hands(
    window: np.ndarray,
    rng: np.random.Generator,
    bias_sigma: float = 0.012,
    jitter_sigma: float = 0.006,
    max_rot_deg: float = 10.0,
    max_scale: float = 0.1,
) -> np.ndarray:
    """Return a copy of ``window`` (T, FEATURE_DIM) with both hands perturbed."""
    out = np.array(window, dtype=np.float32, copy=True)
    t = len(out)
    for block in _HANDS:
        hand = out[:, block].reshape(t, ls.NUM_HAND_LANDMARKS, ls.COORDS_PER_LANDMARK)
        present = np.any(hand != 0.0, axis=(1, 2))
        if not present.any():
            continue
        theta = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
        scale = 1.0 + rng.uniform(-max_scale, max_scale)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        bias = rng.normal(0.0, bias_sigma, (ls.NUM_HAND_LANDMARKS, ls.COORDS_PER_LANDMARK))
        jitter = rng.normal(0.0, jitter_sigma, hand.shape)
        bias[0] = 0.0  # the wrist is the pivot: keep it where the pose model agrees
        jitter[:, 0] = 0.0

        wrist = hand[:, :1, :]
        rel = hand - wrist
        rel[..., :2] = scale * (rel[..., :2] @ rot.T)
        rel[..., 2] *= scale
        new = wrist + rel + bias + jitter
        hand[present] = new[present].astype(np.float32)
        out[:, block] = hand.reshape(t, -1)
    return out


def drop_hand_run(window: np.ndarray, rng: np.random.Generator, prob: float = 0.5) -> np.ndarray:
    """Return a copy with, per moving hand and with probability ``prob``, one short run blanked.

    The Pi's lite hand model loses a hand for a few frames during fast movements, which
    recordings made with the full model never show. The run is 4-12 of the window's
    steps (~0.1-0.4 s) and is centred where the hand moves fastest. A hand that is
    absent or never moves is left alone; the pose block is untouched.
    """
    out = np.array(window, dtype=np.float32, copy=True)
    n = len(out)
    for block in _HANDS:
        present = np.any(out[:, block] != 0.0, axis=1)
        speed = np.r_[0.0, np.abs(np.diff(out[:, block], axis=0)).sum(axis=1)]
        speed *= present & np.r_[False, present[:-1]]
        if speed.sum() <= 0.0 or rng.random() >= prob:
            continue
        length = int(rng.integers(4, 13))
        if length > n:
            continue
        centre = int(rng.choice(n, p=speed / speed.sum()))
        start = min(max(centre - length // 2, 0), n - length)
        out[start : start + length, block] = 0.0
    return out
