"""Live sign-recognition demo (PC or Pi).

Camera -> MediaPipe -> normalized rolling window -> INT8 TFLite GRU -> predicted
word. Needs the trained model (run ``notebooks/train_gru.py`` first) + MediaPipe.

    python -m senseless.sign.demo                 # GUI window (needs a display)
    python -m senseless.sign.demo --headless      # console only (SSH / headless Pi)
    python -m senseless.sign.demo --camera picamera --headless   # on the Pi
    python -m senseless.sign.demo --backend lite --parallel --camera opencv --headless  # Pi 4

The window is time-based (sign/window.py): the last ~1.5 s of frames, whatever the
FPS, resampled to the 45 steps at 30 FPS the model was trained on. The FPS readout
still matters, since fewer frames per sign means less detail; the Pi 4 with
``--backend lite --parallel`` reaches ~10 FPS. A prediction is only shown while a
hand is in frame.
GUI: press q or Esc to quit. Headless: Ctrl+C.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN
from senseless.sign import capture, landmarks
from senseless.sign.classifier import SignClassifier
from senseless.sign.window import TimeWindow

_LEFT_HAND = (0, 255, 0)
_RIGHT_HAND = (255, 128, 0)
_POSE = (0, 0, 255)


def _draw(
    frame_rgb: np.ndarray, raw: landmarks.RawLandmarks, word: str | None, conf: float, fps: float
):
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
        f"hands: {hands}   fps: {fps:4.1f}   q: quit",
        (12, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(description="Live sign recognition demo.")
    parser.add_argument("--backend", choices=landmarks.BACKEND_NAMES, default=None)
    parser.add_argument(
        "--camera",
        choices=["auto", "picamera", "opencv"],
        default="auto",
        help="Frame source; auto picks picamera2 on the Pi, else OpenCV.",
    )
    parser.add_argument("--source", type=int, default=0, help="Webcam index (OpenCV path).")
    parser.add_argument(
        "--headless", action="store_true", help="No window; print predictions + FPS (SSH/Pi)."
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run the pose and hands models in separate processes (use on the Pi).",
    )
    args = parser.parse_args()

    if not args.headless:
        import cv2

    prefer = None if args.camera == "auto" else args.camera
    backend = landmarks.create_backend(args.backend, parallel=True if args.parallel else None)
    classifier = SignClassifier()
    # Newest-frame-only: when processing is slower than the camera (the Pi), skip
    # stale frames instead of lagging behind the camera's buffer.
    src = capture.LatestFrameGrabber(capture.open_frame_source(prefer=prefer, source=args.source))

    window = TimeWindow()
    word: str | None = None
    conf = 0.0
    start = prev = time.perf_counter()
    last_infer = last_print = start
    fps = 0.0
    print(
        f"Running (labels: {len(classifier.labels)}). {'Ctrl+C' if args.headless else 'q'} to quit."
    )
    try:
        with backend, src:
            while (item := src.read_stamped()) is not None:
                frame_rgb, stamp = item
                ts_ms = int((stamp - start) * 1000)  # capture time; strictly increasing
                raw = backend.extract(frame_rgb, ts_ms)
                window.append(stamp, landmarks.frame_landmarks_to_vector(raw))

                now = time.perf_counter()
                dt = now - prev
                prev = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

                hands_present = raw.left_hand is not None or raw.right_hand is not None
                if window.ready() and now - last_infer >= SIGN.inference_interval_s:
                    last_infer = now
                    word, conf = (
                        classifier.predict(window.sample()) if hands_present else (None, 0.0)
                    )

                if args.headless:
                    if now - last_print >= 0.5:
                        last_print = now
                        shown = word if word else "..."
                        print(
                            f"\rfps={fps:4.1f}  hands={'Y' if hands_present else '-'}  "
                            f"pred={shown} ({conf:.2f})      ",
                            end="",
                            flush=True,
                        )
                else:
                    cv2.imshow("Senseless - live sign demo", _draw(frame_rgb, raw, word, conf, fps))
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
