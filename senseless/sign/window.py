"""Time-based windowing: make "45 frames" mean the same span of time at any FPS.

The classifier learned each sign as ``SIGN.window_length`` (45) consecutive frames
recorded at ~``SIGN.reference_fps`` (30), i.e. a span of 44/30 s. Counting frames
instead of time breaks when the frame rate changes: at the Pi's ~10 FPS, "the last
45 frames" is 4.5 s of motion and every sign looks three times slower. Here frames
are kept with their capture timestamps and resampled to the model's 45 evenly
spaced steps over that same span, whatever rate they arrived at.

Interpolation is per block (left hand, right hand, pose). Between two frames a
block is interpolated linearly only if it is present (non-zero) in both;
otherwise the nearer frame's block is used, so a hand is never blended halfway
with the all-zero "absent" block of the missing-hand policy. Samples before the
first frame or after the last hold the edge frame.

``resample_window`` is pure and is also used to simulate low frame rates during
training, so training and inference resample identically.
"""

from __future__ import annotations

import collections

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN

_BLOCKS = (
    slice(ls.LEFT_HAND_START, ls.LEFT_HAND_END),
    slice(ls.RIGHT_HAND_START, ls.RIGHT_HAND_END),
    slice(ls.POSE_START, ls.POSE_END),
)


def default_span_s() -> float:
    """Time covered by one model window: (window_length - 1) reference-rate frames."""
    return (SIGN.window_length - 1) / SIGN.reference_fps


def resample_window(
    times: np.ndarray, vecs: np.ndarray, end_time: float, length: int, span_s: float
) -> np.ndarray:
    """Resample timestamped feature vectors onto ``length`` evenly spaced steps.

    ``times`` (N,) must be increasing; ``vecs`` is (N, FEATURE_DIM). The steps run
    from ``end_time - span_s`` to ``end_time``. Returns (length, FEATURE_DIM) float32.
    """
    times = np.asarray(times, dtype=np.float64)
    vecs = np.asarray(vecs, dtype=np.float32)
    n = len(times)
    if n == 0:
        raise ValueError("need at least one frame to resample")
    steps = end_time - span_s + np.arange(length) * (span_s / max(length - 1, 1))

    # Left neighbour j and right neighbour k for every step, with weight w toward k.
    j = np.searchsorted(times, steps, side="right") - 1
    before = j < 0
    after = j >= n - 1
    j = np.clip(j, 0, n - 1)
    k = np.minimum(j + 1, n - 1)
    gap = times[k] - times[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(gap > 0, (steps - times[j]) / gap, 0.0)
    w[before] = 0.0  # hold the first frame
    w[after] = 0.0  # hold the last frame (j == k == n - 1)
    w = w[:, None].astype(np.float32)

    out = np.empty((length, vecs.shape[1]), dtype=np.float32)
    for block in _BLOCKS:
        vj, vk = vecs[j, block], vecs[k, block]
        present_j = np.any(vj != 0.0, axis=1, keepdims=True)
        present_k = np.any(vk != 0.0, axis=1, keepdims=True)
        lerp = vj * (1.0 - w) + vk * w
        nearest = np.where(w < 0.5, vj, vk)
        out[:, block] = np.where(present_j & present_k, lerp, nearest)
    return out


def simulate_capture(
    window: np.ndarray,
    fps: float,
    rng: np.random.Generator,
    jitter: float = 0.15,
    reference_fps: float | None = None,
) -> np.ndarray:
    """What the live time window would produce had ``window`` been captured at ``fps``.

    ``window`` is (length, FEATURE_DIM) recorded at ``reference_fps``. A slower camera
    catches only every ``reference_fps / fps``-th frame, at a random phase and with
    timing jitter (as a fraction of its frame interval); those frames are resampled
    back onto the window's steps with ``resample_window``, exactly as at inference.
    Used as training augmentation so the model learns signs at the Pi's frame rate.
    """
    ref = SIGN.reference_fps if reference_fps is None else reference_fps
    n = len(window)
    times = np.arange(n) / ref
    if fps >= ref:  # a camera at the reference rate catches every frame
        return resample_window(times, window, end_time=times[-1], length=n, span_s=times[-1])
    step = ref / fps
    picks = np.arange(rng.uniform(0.0, step), n, step)
    picks = picks + rng.normal(0.0, jitter * step, len(picks))
    idx = np.unique(np.clip(np.round(picks), 0, n - 1).astype(int))
    return resample_window(times[idx], window[idx], end_time=times[-1], length=n, span_s=times[-1])


class TimeWindow:
    """Rolling buffer of timestamped feature vectors, sampled as one model window.

    ``append(t, vec)`` with increasing ``t`` (seconds). ``ready()`` is true once the
    buffer covers ``span_s``; ``sample()`` then returns the newest span resampled to
    ``length`` steps. Only frames still needed are kept (one older than the span, for
    interpolating its start), so the buffer stays bounded at any frame rate.
    """

    def __init__(self, length: int | None = None, span_s: float | None = None) -> None:
        self.length = SIGN.window_length if length is None else length
        self.span_s = default_span_s() if span_s is None else span_s
        self._times: collections.deque[float] = collections.deque()
        self._vecs: collections.deque[np.ndarray] = collections.deque()

    def append(self, t: float, vec: np.ndarray) -> None:
        if self._times and t <= self._times[-1]:
            raise ValueError(f"timestamps must increase ({t} after {self._times[-1]})")
        self._times.append(float(t))
        self._vecs.append(vec)
        start = t - self.span_s
        while len(self._times) >= 2 and self._times[1] <= start:
            self._times.popleft()
            self._vecs.popleft()

    def ready(self) -> bool:
        return bool(self._times) and self._times[0] <= self._times[-1] - self.span_s + 1e-9

    def sample(self) -> np.ndarray:
        return resample_window(
            np.fromiter(self._times, dtype=np.float64),
            np.stack(self._vecs),
            end_time=self._times[-1],
            length=self.length,
            span_s=self.span_s,
        )

    def clear(self) -> None:
        self._times.clear()
        self._vecs.clear()

    def __len__(self) -> int:
        return len(self._times)
