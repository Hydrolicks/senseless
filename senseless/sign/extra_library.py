"""MS-ASL takes for the Speech-mode figure: clean, rank, review, add to the library.

Words we did not record get a stick-figure take from MS-ASL clips (lite-tracker
landmark sequences from ``collect.msasl``). Each clip is trimmed to the annotated
sign, dropped if the hands are missing in most frames, has short tracker gaps
interpolated, is mirrored to right-hand dominance like our own takes, and is
resampled to 30 FPS at its natural speed (capped at 3 s). Per word, the most
typical clip is the automatic pick; a review page lets a person reject words or
pick alternates. Our own recorded takes always win (``sign/library.py``).

    python -m senseless.sign.extra_library review   # C:/Senseless_anim -> review.html
    python -m senseless.sign.extra_library build [--choices choices.json]
"""

from __future__ import annotations

import math

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.sign.window import resample_window

FPS = 30.0  # playback rate of the figure (UI.figure_tick_ms ~ 33 ms)
MAX_TAKE_S = 3.0
MIN_HAND_COVERAGE = 0.6
MAX_GAP_S = 0.3
COMPARE_STEPS = 45  # takes are resampled to this many steps only to compare them

_LEFT = slice(ls.LEFT_HAND_START, ls.LEFT_HAND_END)
_RIGHT = slice(ls.RIGHT_HAND_START, ls.RIGHT_HAND_END)
_POSE = slice(ls.POSE_START, ls.POSE_END)
_BLOCKS = (_LEFT, _RIGHT, _POSE)
_C = ls.COORDS_PER_LANDMARK
_POSE_PAIRS = tuple(
    (i, ls.POSE_LANDMARK_NAMES.index("right_" + name[len("left_") :]))
    for i, name in enumerate(ls.POSE_LANDMARK_NAMES)
    if name.startswith("left_")
)


def _present(vecs: np.ndarray, block: slice) -> np.ndarray:
    return np.any(vecs[:, block] != 0.0, axis=1)


def hand_coverage(vecs: np.ndarray) -> float:
    """Share of frames with at least one hand."""
    if len(vecs) == 0:
        return 0.0
    return float(np.mean(_present(vecs, _LEFT) | _present(vecs, _RIGHT)))


def fill_gaps(times: np.ndarray, vecs: np.ndarray, max_gap_s: float = MAX_GAP_S) -> np.ndarray:
    """Linearly fill a block missing between two frames that have it, if they are <= max_gap_s apart."""  # noqa: E501
    out = np.array(vecs, dtype=np.float32, copy=True)
    for block in _BLOCKS:
        idx = np.flatnonzero(_present(out, block))
        for a, b in zip(idx[:-1], idx[1:], strict=False):
            if b - a > 1 and times[b] - times[a] <= max_gap_s + 1e-9:
                w = ((times[a + 1 : b] - times[a]) / (times[b] - times[a]))[:, None]
                out[a + 1 : b, block] = out[a, block] * (1.0 - w) + out[b, block] * w
    return out


def motion(vecs: np.ndarray, block: slice) -> float:
    """Summed frame-to-frame change of a block, over consecutive frames that both have it."""
    present = _present(vecs, block)
    both = present[1:] & present[:-1]
    return float(np.abs(np.diff(vecs[:, block], axis=0))[both].sum())


def mirror(vecs: np.ndarray) -> np.ndarray:
    """Left-right mirror: negate x, swap the hands and the left/right pose points."""
    out = np.array(vecs, dtype=np.float32, copy=True)
    out[:, 0::_C] *= -1.0  # every point starts on a multiple of 3: x of every point
    out[:, _LEFT], out[:, _RIGHT] = out[:, _RIGHT].copy(), out[:, _LEFT].copy()
    for a, b in _POSE_PAIRS:
        pa = slice(ls.POSE_START + a * _C, ls.POSE_START + (a + 1) * _C)
        pb = slice(ls.POSE_START + b * _C, ls.POSE_START + (b + 1) * _C)
        out[:, pa], out[:, pb] = out[:, pb].copy(), out[:, pa].copy()
    return out


def natural_resample(times: np.ndarray, vecs: np.ndarray, duration: float) -> np.ndarray:
    """Resample [0, min(duration, MAX_TAKE_S)] to FPS frames per second."""
    span = min(float(duration), MAX_TAKE_S)
    if span <= 0:
        raise ValueError("duration must be positive")
    n = int(math.ceil(span * FPS - 1e-9)) + 1
    return resample_window(
        np.asarray(times), np.asarray(vecs), end_time=span, length=n, span_s=span
    )


def clean_clip(times: np.ndarray, vecs: np.ndarray, duration: float) -> np.ndarray | None:
    """One playable take from a stored MS-ASL sequence, or None if the hands are mostly missing."""
    times, vecs = np.asarray(times, dtype=np.float64), np.asarray(vecs, dtype=np.float32)
    keep = (times >= 0.0) & (times <= duration + 1e-9)
    t, v = times[keep], vecs[keep]
    if len(t) == 0 or hand_coverage(v) < MIN_HAND_COVERAGE:
        return None
    v = fill_gaps(t, v)
    if motion(v, _LEFT) > motion(v, _RIGHT):
        v = mirror(v)
    return natural_resample(t, v, duration)


def _compare_form(take: np.ndarray) -> np.ndarray:
    n = len(take)
    if n < 2:
        return np.repeat(take, COMPARE_STEPS, axis=0)
    span = (n - 1) / FPS
    return resample_window(
        np.arange(n) / FPS, take, end_time=span, length=COMPARE_STEPS, span_s=span
    )


def rank(takes: list[np.ndarray]) -> list[int]:
    """Indices from most to least typical (smallest summed distance to the others first)."""
    if len(takes) <= 1:
        return list(range(len(takes)))
    flat = np.stack([_compare_form(t) for t in takes]).reshape(len(takes), -1).astype(np.float64)
    sq = (flat**2).sum(axis=1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T, 0.0))
    return [int(i) for i in np.argsort(dist.sum(axis=1), kind="stable")]
