"""Onset-triggered sign segmentation for the live demo ("onset" mode).

Every training window starts the moment a hand enters the frame and covers one
sign (see collect/recorder.py). Classifying a sliding window continuously also
shows the model windows it never saw in training (the end of one sign plus the
start of the next, a hand on its way up), so wrong words flicker between the
right ones. Onset mode feeds the classifier training-shaped windows only:

    idle        rest with the hands out of view
    pending     a hand appeared; wait for ``confirm_frames`` frames in a row with a
                hand, so a one-frame false detection doesn't start a capture
    capturing   collect one span from the first hand frame (a brief detection
                dropout doesn't abort it)
    holding     one window was emitted; wait until the hands have been gone for
                ``release_s`` before arming again, so each sign gives one word

The emitted window is resampled with ``window.resample_window``, exactly like the
recorder's takes. Pure logic: timestamps and vectors in, windows out.
"""

from __future__ import annotations

import numpy as np

from senseless.common.config import SIGN
from senseless.sign.window import default_span_s, resample_window


class OnsetSegmenter:
    """Turns a stream of (time, feature vector, hands present) into one window per sign."""

    def __init__(
        self,
        length: int | None = None,
        span_s: float | None = None,
        release_s: float | None = None,
        confirm_frames: int | None = None,
    ) -> None:
        self.length = SIGN.window_length if length is None else length
        self.span_s = default_span_s() if span_s is None else span_s
        self.release_s = SIGN.release_s if release_s is None else release_s
        self.confirm_frames = max(
            1, SIGN.onset_confirm_frames if confirm_frames is None else confirm_frames
        )
        self.state = "idle"
        self._onset = 0.0
        self._times: list[float] = []
        self._vecs: list[np.ndarray] = []
        self._seen = 0
        self._absent_since: float | None = None

    def progress(self, t: float) -> float:
        """Fraction of the current capture completed (0 unless capturing)."""
        if self.state not in ("pending", "capturing"):
            return 0.0
        return float(min(max((t - self._onset) / self.span_s, 0.0), 1.0))

    def update(self, t: float, vec: np.ndarray, hands_present: bool) -> np.ndarray | None:
        """Feed one frame; return a (length, FEATURE_DIM) window when a sign completes."""
        if self.state == "idle":
            if hands_present:
                self._onset, self._times, self._vecs, self._seen = t, [t], [vec], 1
                self.state = "capturing" if self.confirm_frames <= 1 else "pending"
            return None

        if self.state == "pending":
            if not hands_present:  # a blip: forget it
                self.state = "idle"
                return None
            self._times.append(t)
            self._vecs.append(vec)
            self._seen += 1
            if self._seen >= self.confirm_frames:
                self.state = "capturing"
            return self._maybe_emit(t)

        if self.state == "capturing":
            self._times.append(t)
            self._vecs.append(vec)
            return self._maybe_emit(t)

        # holding: re-arm once the hands have been gone for release_s
        if hands_present:
            self._absent_since = None
        else:
            if self._absent_since is None:
                self._absent_since = t
            if t - self._absent_since >= self.release_s - 1e-9:
                self.state = "idle"
                self._absent_since = None
        return None

    def _maybe_emit(self, t: float) -> np.ndarray | None:
        if self.state != "capturing" or t - self._onset < self.span_s - 1e-9:
            return None
        window = resample_window(
            np.array(self._times),
            np.stack(self._vecs),
            end_time=self._onset + self.span_s,
            length=self.length,
            span_s=self.span_s,
        )
        self.state = "holding"
        self._absent_since = None
        self._times, self._vecs = [], []
        return window
