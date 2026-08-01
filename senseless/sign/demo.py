"""Live sign-recognition demo (DEV TOOL, PC).

Webcam -> MediaPipe -> normalized rolling window -> INT8 TFLite GRU -> predicted
word, drawn live. Needs the trained model (run ``notebooks/train_gru.py`` first),
MediaPipe, and a webcam.

    python -m senseless.sign.demo
    python -m senseless.sign.demo --source 1

A prediction is only shown while a hand is in frame (so it stays quiet at rest).
Press q or Esc to quit.
"""

from __future__ import annotations

import argparse
import collections
import time

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN
from senseless.sign import capture, landmarks
from senseless.sign.classifier import SignClassifier

_LEFT_HAND = (0, 255, 0)
_RIGHT_HAND = (255, 128, 0)
_POSE = (0, 0, 255)


def _draw(frame_rgb: np.ndarray, raw: landmarks.RawLandmarks, word: str | None, conf: float):
    """Draw landmarks + the current prediction; return a BGR image."""
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

    hands = ("L" if raw.left_hand is not None else "-") + (
        "R" if raw.right_hand is not None else "-"
    )
    label = word if word else "..."
    color = (0, 220, 0) if word else (200, 200, 200)
    cv2.putText(bgr, f"{label}  {conf:.2f}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
    cv2.putText(bgr, f"{label}  {conf:.2f}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
    cv2.putText(
        bgr,
        f"hands: {hands}   q: quit",
        (12, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(description="Live sign recognition demo.")
    parser.add_argument("--backend", choices=["tasks", "holistic"], default=None)
    parser.add_argument("--source", type=int, default=0, help="Webcam index.")
    args = parser.parse_args()

    import cv2

    backend = landmarks.create_backend(args.backend)
    classifier = SignClassifier()
    src = capture.open_frame_source(source=args.source)
    window: collections.deque[np.ndarray] = collections.deque(maxlen=SIGN.window_length)
    word: str | None = None
    conf = 0.0
    frame_i = 0
    start = time.perf_counter()
    try:
        with backend, src:
            for frame_rgb in src.frames():
                ts_ms = int((time.perf_counter() - start) * 1000)
                raw = backend.extract(frame_rgb, ts_ms)
                window.append(landmarks.frame_landmarks_to_vector(raw))
                frame_i += 1

                full = len(window) == SIGN.window_length
                hands_present = raw.left_hand is not None or raw.right_hand is not None
                if full and frame_i % SIGN.inference_stride == 0:
                    if hands_present:
                        word, conf = classifier.predict(np.stack(window))
                    else:
                        word, conf = None, 0.0

                cv2.imshow("Senseless - live sign demo", _draw(frame_rgb, raw, word, conf))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
