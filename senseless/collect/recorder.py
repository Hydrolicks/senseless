"""Interactive sign data-collection recorder (DEV TOOL, PC).

Keypress-per-take recorder with a live monitoring window. Press SPACE to *arm* a
take; capture then starts automatically on the first frame a hand appears, runs
for one window span (~1.5 s, see ``sign/window.py``) and is resampled to the
model's ``window_length`` steps, so a take means the same stretch of time at any
frame rate (30 FPS on the PC reproduces the frames almost exactly).
Arming-then-onset means the sign fills the window from the moment your hands enter
frame -- no wasted empty lead-in and no clipping of the start (so an idle position
with hands out of frame is fine).
The overlay reports which hands are actually detected, so you can tell an empty
window from a real one before it saves.

Windows are saved under ``data/<label>/`` via ``collect.dataset``, using the same
normalized landmark pipeline as inference.

Controls: SPACE = arm a take | q or Esc = quit.

Run via ``python -m senseless.collect`` (see instructions.md, step 5).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from senseless.collect import dataset
from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN
from senseless.sign import capture, landmarks
from senseless.sign.window import default_span_s, resample_window

# Landmark dot colors (BGR).
_LEFT_HAND = (0, 255, 0)
_RIGHT_HAND = (255, 128, 0)
_POSE = (0, 0, 255)
_SPACE_KEY = 32


class TakeMachine:
    """Take logic of the recorder, with no camera and no GUI, so it can be tested.

    Manual mode: ``idle`` -> ``arm()`` -> ``armed`` -> first hand frame -> ``recording``
    -> one span later the take is returned -> ``idle``.
    Auto mode starts ``armed``. After a take it goes to ``waiting`` and re-arms once no
    hand has been seen for ``clear_s`` without a break. ``toggle_pause()`` switches to
    ``paused`` (dropping a take in progress) and back to ``waiting``.
    """

    def __init__(
        self,
        span_s: float,
        auto: bool = False,
        clear_s: float = SIGN.collect_clear_s,
        length: int = SIGN.window_length,
    ) -> None:
        self.span_s = span_s
        self.auto = auto
        self.clear_s = clear_s
        self.length = length
        self.state = "armed" if auto else "idle"
        self.onset = 0.0
        self.frames = 0
        self._times: list[float] = []
        self._vecs: list[np.ndarray] = []
        self._clear_since: float | None = None

    def arm(self) -> None:
        """Manual mode: arm one take (only from idle)."""
        if not self.auto and self.state == "idle":
            self.state = "armed"

    def toggle_pause(self) -> None:
        """Auto mode: pause (dropping any take in progress) or resume into waiting."""
        if not self.auto:
            return
        if self.state == "paused":
            self.state, self._clear_since = "waiting", None
        else:
            self.state, self._times, self._vecs = "paused", [], []

    def step(self, stamp: float, vec: np.ndarray, hands_present: bool) -> np.ndarray | None:
        """Advance by one frame; return a finished ``(length, FEATURE_DIM)`` take or None."""
        if self.state == "armed" and hands_present:
            self.state, self.onset = "recording", stamp
            self._times, self._vecs = [stamp], [vec]
        elif self.state == "recording":
            self._times.append(stamp)
            self._vecs.append(vec)
            if stamp - self.onset >= self.span_s:
                take = resample_window(
                    np.array(self._times),
                    np.stack(self._vecs),
                    end_time=self.onset + self.span_s,
                    length=self.length,
                    span_s=self.span_s,
                )
                self.frames = len(self._times)
                self._times, self._vecs = [], []
                self.state = "waiting" if self.auto else "idle"
                self._clear_since = None
                return take
        elif self.state == "waiting":
            if hands_present:
                self._clear_since = None
            elif self._clear_since is None:
                self._clear_since = stamp
            elif stamp - self._clear_since >= self.clear_s:
                self.state = "armed"
        return None


def undo_last(saved: list[Path]) -> Path | None:
    """Delete the newest take saved this session; return its path, or None if none are left."""
    if not saved:
        return None
    path = saved.pop()
    path.unlink(missing_ok=True)
    return path


def _draw(frame_rgb: np.ndarray, raw: landmarks.RawLandmarks, lines: list[str], recording: bool):
    """Draw landmark dots + status lines; return a BGR image for cv2.imshow."""
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    def _dots(points: np.ndarray | None, color: tuple[int, int, int]) -> None:
        if points is None:
            return
        for x, y, _z in points:
            cv2.circle(bgr, (int(x * w), int(y * h)), 4, color, -1)

    _dots(raw.left_hand, _LEFT_HAND)
    _dots(raw.right_hand, _RIGHT_HAND)
    if raw.pose is not None:
        _dots(np.asarray(raw.pose)[list(ls.POSE_LANDMARK_INDICES)], _POSE)
    if recording:
        cv2.circle(bgr, (22, 22), 10, (0, 0, 255), -1)  # red dot = recording

    y = 24
    for text in lines:
        cv2.putText(bgr, text, (44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            bgr, text, (44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 28
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(description="Record sign samples (arm, capture at onset).")
    parser.add_argument("--label", required=True, help="Vocabulary word to record.")
    parser.add_argument("--samples", type=int, default=30, help="Target number of takes.")
    parser.add_argument("--backend", choices=landmarks.BACKEND_NAMES, default=None)
    parser.add_argument("--source", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--parallel", action="store_true", help="Run pose and hands in separate processes."
    )
    args = parser.parse_args()

    import cv2

    backend = landmarks.create_backend(args.backend, parallel=True if args.parallel else None)
    src = capture.LatestFrameGrabber(capture.open_frame_source(source=args.source))
    count = dataset.count_samples(args.label)
    state = "idle"  # idle -> armed -> recording -> idle
    span = default_span_s()
    times: list[float] = []
    vecs: list[np.ndarray] = []
    start = onset = time.perf_counter()
    window_name = f"Senseless collect - {args.label}"
    print(f"Recording '{args.label}'. SPACE = arm a take, q = quit. Already have {count}.")

    try:
        with backend, src:
            while (item := src.read_stamped()) is not None:
                frame_rgb, stamp = item
                raw = backend.extract(frame_rgb, int((stamp - start) * 1000))
                vec = landmarks.frame_landmarks_to_vector(raw)
                hands_present = raw.left_hand is not None or raw.right_hand is not None

                if state == "armed" and hands_present:
                    state, onset, times, vecs = "recording", stamp, [stamp], [vec]
                elif state == "recording":
                    times.append(stamp)
                    vecs.append(vec)
                    if stamp - onset >= span:
                        take = resample_window(
                            np.array(times),
                            np.stack(vecs),
                            end_time=onset + span,
                            length=SIGN.window_length,
                            span_s=span,
                        )
                        path = dataset.save_window(take, args.label)
                        count += 1
                        state = "idle"
                        print(f"saved {path.name}  ({count}/{args.samples}, {len(times)} frames)")

                left = "L" if raw.left_hand is not None else "-"
                right = "R" if raw.right_hand is not None else "-"
                status = {
                    "idle": "SPACE = arm a take",
                    "armed": "armed - waiting for a hand...",
                    "recording": f"REC {stamp - onset:.1f}/{span:.1f} s",
                }[state]
                lines = [
                    f"{args.label}    saved {count}/{args.samples}",
                    f"hands: {left}{right}    {status}",
                    "SPACE = arm    |    q = quit",
                ]
                cv2.imshow(window_name, _draw(frame_rgb, raw, lines, state == "recording"))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == _SPACE_KEY and state == "idle":
                    state = "armed"
    finally:
        cv2.destroyAllWindows()
    print(f"Done. {count} samples for '{args.label}' in {dataset.label_dir(args.label)}.")
