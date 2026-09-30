"""Sign-mode worker for the touchscreen app (runs in its own process).

Camera (newest frame only) -> perception (lite models in parallel on the Pi) ->
normalized vector -> onset segmenter -> classifier, one result per sign. Sends
a small RGB preview with the hand dots drawn on it to ``frames`` and status /
results / errors to ``events`` (see common/events.py). The parts are injectable
factories so tests can run the real loop with fakes.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN, UI
from senseless.common.events import SignResult, SignStatus, WorkerError, WorkerReady
from senseless.common.process import parent_alive
from senseless.sign import capture, landmarks
from senseless.sign.classifier import interpret
from senseless.sign.segmenter import OnsetSegmenter

_LEFT = (80, 210, 140)  # RGB, as in the GUI legend
_RIGHT = (242, 163, 58)
_POSE = (228, 87, 46)


@dataclass
class SignParts:
    """Factories for the pipeline parts; swapped for fakes in tests."""

    source: Callable[[], Any]  # -> object with read_stamped(), usable as a context manager
    backend: Callable[[], landmarks.PerceptionBackend]
    classifier: Callable[[], Any]  # -> object with .labels and .probabilities(window)


def default_sign_parts() -> SignParts:
    from senseless.sign.classifier import SignClassifier

    return SignParts(
        source=lambda: capture.LatestFrameGrabber(
            capture.open_frame_source(prefer=UI.camera, source=UI.camera_source)
        ),
        backend=lambda: landmarks.create_backend(
            UI.perception_backend, parallel=UI.parallel_perception
        ),
        classifier=SignClassifier,
    )


def make_preview(
    frame_rgb: np.ndarray, raw: landmarks.RawLandmarks, size: tuple[int, int] = UI.preview_size
) -> np.ndarray:
    """Downscale the frame to ``size`` (w, h) and draw the hand and pose dots (RGB)."""
    import cv2

    width, height = size
    img = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)

    def dots(points, color) -> None:
        if points is None:
            return
        for x, y, _z in np.asarray(points):
            cv2.circle(img, (int(x * width), int(y * height)), 3, color, -1)

    dots(raw.left_hand, _LEFT)
    dots(raw.right_hand, _RIGHT)
    if raw.pose is not None:
        dots(np.asarray(raw.pose)[list(ls.POSE_LANDMARK_INDICES)], _POSE)
    return np.ascontiguousarray(img, dtype=np.uint8)


def run_sign_worker(frames, events, stop, parts: SignParts | None = None) -> None:
    """Run the sign pipeline until ``stop`` is set or something fails."""
    try:
        parts = parts if parts is not None else default_sign_parts()
        classifier = parts.classifier()
        backend = parts.backend()
    except BaseException as exc:  # noqa: BLE001 -- shown to the user
        traceback.print_exc()
        events.put(WorkerError(f"Sign model or MediaPipe files could not load: {exc}"))
        return
    try:
        source = parts.source()
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        backend.close()
        events.put(WorkerError(f"Camera not found. Check the USB cable, then tap Retry. ({exc})"))
        return

    segmenter = OnsetSegmenter()
    t0: float | None = None
    prev = time.perf_counter()
    fps = 0.0
    try:
        with backend, source:
            events.put(WorkerReady("sign"))
            while not stop.is_set() and parent_alive():  # parent gone: don't hold the camera
                item = source.read_stamped()
                if item is None:
                    events.put(WorkerError("Camera stream ended."))
                    return
                frame, stamp = item
                if t0 is None:
                    t0 = stamp
                raw = backend.extract(frame, int((stamp - t0) * 1000))
                vec = landmarks.frame_landmarks_to_vector(raw)
                hands = raw.left_hand is not None or raw.right_hand is not None

                now = time.perf_counter()
                if now > prev:
                    inst = 1.0 / (now - prev)
                    fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
                prev = now

                window = segmenter.update(stamp, vec, hands)
                if window is not None:
                    word, best, conf = interpret(
                        classifier.probabilities(window),
                        classifier.labels,
                        SIGN.min_confidence,
                        SIGN.idle_label,
                    )
                    events.put(SignResult(word, best, conf))
                events.put(SignStatus(segmenter.state, segmenter.progress(stamp), hands, fps))
                frames.put(make_preview(frame, raw))
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        events.put(WorkerError(f"Sign engine error: {exc}"))
