"""Interactive sign data-collection recorder (DEV TOOL, PC).

Live monitoring window plus a take recorder. A take starts automatically on the
first frame a hand appears, runs for one window span (~1.5 s, see
``sign/window.py``) and is resampled to the model's ``window_length`` steps, so a
take means the same stretch of time at any frame rate. Starting at the onset means
the sign fills the window from the moment your hands enter frame: no wasted empty
lead-in, and no clipping of the start.

Two ways to arm a take (the logic lives in ``TakeMachine``):

- default: press SPACE before each take.
- ``--auto``: hands-free. After a take, hide your hands (e.g. behind your back); once
  no hand has been seen for ``SIGN.collect_clear_s`` (0.5 s) the next take is armed.
  The first take is also armed only after the hands have been hidden, so start with
  your hands out of view. SPACE pauses and resumes.

BACKSPACE deletes the last take saved in this session (repeatable). The overlay
reports which hands are detected, so you can tell an empty window from a real one.

Windows are saved under ``data/<label>/`` via ``collect.dataset``, using the same
normalized landmark pipeline as inference.

Run via ``python -m senseless.collect --label WORD [--auto]`` (see instructions.md).
On the Pi with the USB webcam (stop the app first; it holds the camera)::

    python -m senseless.collect --label WORD --auto --backend lite --parallel --camera opencv
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
_BACKSPACE_KEY = 8
_STATUS = {
    "idle": "SPACE = arm a take",
    "armed": "armed - waiting for a hand...",
    "waiting": "hide hands to re-arm",
    "paused": "PAUSED - SPACE to resume",
}


class TakeMachine:
    """Take logic of the recorder, with no camera and no GUI, so it can be tested.

    Manual mode: ``idle`` -> ``arm()`` -> ``armed`` -> first hand frame -> ``recording``
    -> one span later the take is returned -> ``idle``.
    Auto mode starts ``waiting`` (the hands are in view when the recorder is launched).
    It re-arms once no hand has been seen for ``clear_s`` without a break, and does so
    again after every take. While ``armed`` it can sit for a long time, so a first hand
    frame only moves it to ``pending``; ``confirm_frames`` hand frames in a row (like the
    live ``OnsetSegmenter``) make it ``recording`` with the onset at the first of them,
    and a no-hand frame in ``pending`` drops the blip and goes back to ``armed``.
    ``toggle_pause()`` switches to ``paused`` (dropping a take in progress) and back to
    ``waiting``.

    ``state`` is one of ``"idle"``, ``"armed"``, ``"pending"`` (auto only),
    ``"recording"``, ``"waiting"`` (auto only) or ``"paused"`` (auto only).
    """

    def __init__(
        self,
        span_s: float,
        auto: bool = False,
        clear_s: float = SIGN.collect_clear_s,
        confirm_frames: int = SIGN.onset_confirm_frames,
        length: int = SIGN.window_length,
    ) -> None:
        self.span_s = span_s
        self.auto = auto
        self.clear_s = clear_s
        self.confirm_frames = confirm_frames
        self.length = length
        self.state = "waiting" if auto else "idle"
        self.onset = 0.0
        self.frames = 0
        self._times: list[float] = []
        self._vecs: list[np.ndarray] = []
        self._clear_since: float | None = None
        self._seen = 0

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
            self.onset = stamp
            self._times, self._vecs, self._seen = [stamp], [vec], 1
            self.state = "pending" if self.auto and self.confirm_frames > 1 else "recording"
        elif self.state == "pending":
            if not hands_present:  # a blip: forget it
                self._times, self._vecs = [], []
                self.state = "armed"
            else:
                self._times.append(stamp)
                self._vecs.append(vec)
                self._seen += 1
                if self._seen >= self.confirm_frames:
                    self.state = "recording"
        elif self.state == "recording":
            self._times.append(stamp)
            self._vecs.append(vec)
            if stamp - self.onset >= self.span_s - 1e-9:
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
            elif stamp - self._clear_since >= self.clear_s - 1e-9:
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


def camera_preference(choice: str) -> str | None:
    """``--camera`` value -> ``capture.open_frame_source`` preference (None = auto)."""
    return None if choice == "auto" else choice


def main() -> None:
    parser = argparse.ArgumentParser(description="Record sign samples (capture at hand onset).")
    parser.add_argument("--label", required=True, help="Vocabulary word to record.")
    parser.add_argument("--samples", type=int, default=30, help="Target number of takes.")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Hands-free: re-arm after the hands are hidden for 0.5 s; SPACE pauses.",
    )
    parser.add_argument("--backend", choices=landmarks.BACKEND_NAMES, default=None)
    parser.add_argument(
        "--camera",
        choices=["auto", "picamera", "opencv"],
        default="auto",
        help="Frame source: auto = picamera2 on the Pi, else OpenCV; Pi USB webcam = opencv.",
    )
    parser.add_argument("--source", type=int, default=0, help="Webcam index (OpenCV path).")
    parser.add_argument(
        "--parallel", action="store_true", help="Run pose and hands in separate processes."
    )
    args = parser.parse_args()

    import cv2

    backend = landmarks.create_backend(args.backend, parallel=True if args.parallel else None)
    prefer = camera_preference(args.camera)
    src = capture.LatestFrameGrabber(capture.open_frame_source(prefer=prefer, source=args.source))
    count = dataset.count_samples(args.label)
    machine = TakeMachine(default_span_s(), auto=args.auto)
    saved: list[Path] = []
    start = time.perf_counter()
    window_name = f"Senseless collect - {args.label}"
    space = "SPACE = pause" if args.auto else "SPACE = arm"
    controls = f"{space}  |  BACKSPACE = undo  |  q/Esc = quit"
    print(f"Recording '{args.label}'. {controls}. Already have {count}.")

    try:
        with backend, src:
            while (item := src.read_stamped()) is not None:
                frame_rgb, stamp = item
                raw = backend.extract(frame_rgb, int((stamp - start) * 1000))
                vec = landmarks.frame_landmarks_to_vector(raw)
                hands_present = raw.left_hand is not None or raw.right_hand is not None

                take = machine.step(stamp, vec, hands_present)
                if take is not None:
                    path = dataset.save_window(take, args.label)
                    saved.append(path)
                    count += 1
                    print(f"saved {path.name}  ({count}/{args.samples}, {machine.frames} frames)")

                left = "L" if raw.left_hand is not None else "-"
                right = "R" if raw.right_hand is not None else "-"
                recording = machine.state in ("recording", "pending")
                if recording:
                    status = f"REC {stamp - machine.onset:.1f}/{machine.span_s:.1f} s"
                else:
                    status = _STATUS[machine.state]
                lines = [
                    f"{args.label}    saved {count}/{args.samples}",
                    f"hands: {left}{right}    {status}",
                    controls,
                ]
                cv2.imshow(window_name, _draw(frame_rgb, raw, lines, recording))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == _SPACE_KEY:
                    if args.auto:
                        machine.toggle_pause()
                    else:
                        machine.arm()
                elif key == _BACKSPACE_KEY:
                    path = undo_last(saved)
                    if path is not None:
                        count -= 1
                        print(f"deleted {path.name}  ({count}/{args.samples})")
    finally:
        cv2.destroyAllWindows()
    print(f"Done. {count} samples for '{args.label}' in {dataset.label_dir(args.label)}.")
