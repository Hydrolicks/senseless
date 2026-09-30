"""Live sign-recognition demo (PC or Pi).

Camera -> MediaPipe -> normalized landmarks -> INT8 TFLite GRU -> words. Needs the
trained model (run ``notebooks/train_gru.py`` first) + MediaPipe.

    python -m senseless.sign.demo                 # GUI window (needs a display)
    python -m senseless.sign.demo --headless      # console only (SSH / headless Pi)
    python -m senseless.sign.demo --backend lite --parallel --camera opencv --headless  # Pi 4

Modes:

* ``onset`` (default): rest with your hands out of view; when a hand appears, one
  ~1.5 s span is captured from that moment and classified once, exactly like the
  training takes (sign/segmenter.py). The word is added to a running sentence, and
  the next sign is armed once your hands have left the view. Most reliable.
* ``continuous``: classify the last ~1.5 s every ``SIGN.inference_interval_s``
  while a hand is visible. Signs can be chained without lowering the hands, but
  windows that straddle two signs make wrong words flicker.

The window is time-based (sign/window.py), so a slower camera doesn't stretch the
signs; the Pi 4 with ``--backend lite --parallel`` runs at ~7 FPS.
GUI: q or Esc quits, c clears the sentence. Headless: Ctrl+C.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN
from senseless.sign import capture, landmarks
from senseless.sign.classifier import SignClassifier, interpret
from senseless.sign.segmenter import OnsetSegmenter
from senseless.sign.window import TimeWindow

_LEFT_HAND = (0, 255, 0)
_RIGHT_HAND = (255, 128, 0)
_POSE = (0, 0, 255)
_MAX_SENTENCE_WORDS = 8

_STATE_TEXT = {
    "idle": "rest hands out of view, then sign",
    "pending": "hand detected...",
    "capturing": "capturing sign",
    "holding": "lower your hands for the next sign",
}


def _put(bgr, text: str, org: tuple[int, int], scale: float, color, thick: int = 2) -> None:
    import cv2

    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4)
    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def _draw(
    frame_rgb: np.ndarray,
    raw: landmarks.RawLandmarks,
    headline: str,
    accepted: bool,
    status: str,
    progress: float,
    sentence: list[str],
    fps: float,
):
    """Draw landmarks, the latest result, capture progress and the sentence; return BGR."""
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
    _put(bgr, headline, (12, 48), 1.3, (0, 220, 0) if accepted else (200, 200, 200))
    _put(bgr, f"{status}   hands: {hands}   fps: {fps:4.1f}", (12, 80), 0.55, (255, 255, 255), 1)
    if progress > 0.0:
        cv2.rectangle(bgr, (12, 92), (12 + int((w - 24) * progress), 102), (0, 200, 255), -1)
        cv2.rectangle(bgr, (12, 92), (w - 12, 102), (255, 255, 255), 1)
    if sentence:
        _put(bgr, " ".join(sentence), (12, h - 20), 0.9, (255, 255, 255))
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
    parser.add_argument(
        "--mode",
        choices=["onset", "continuous"],
        default="onset",
        help="onset: one word per sign, hands out of view between signs (default).",
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

    segmenter = OnsetSegmenter()
    window = TimeWindow()
    sentence: list[str] = []
    headline, accepted = "...", False
    start = prev = time.perf_counter()
    last_infer = last_print = start
    fps = 0.0
    quit_key = "Ctrl+C" if args.headless else "q"
    print(f"Running in {args.mode} mode (labels: {len(classifier.labels)}). {quit_key} to quit.")
    if args.mode == "onset":
        print("Rest your hands out of view; sign when ready; lower your hands after each sign.")

    def report(line: str) -> None:
        if args.headless:
            print("\r" + " " * 100 + "\r" + line, flush=True)

    try:
        with backend, src:
            while (item := src.read_stamped()) is not None:
                frame_rgb, stamp = item
                ts_ms = int((stamp - start) * 1000)  # capture time; strictly increasing
                raw = backend.extract(frame_rgb, ts_ms)
                vec = landmarks.frame_landmarks_to_vector(raw)
                hands_present = raw.left_hand is not None or raw.right_hand is not None

                now = time.perf_counter()
                dt = now - prev
                prev = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

                if args.mode == "onset":
                    done = segmenter.update(stamp, vec, hands_present)
                    if done is not None:
                        word, best, conf = interpret(
                            classifier.probabilities(done),
                            classifier.labels,
                            SIGN.min_confidence,
                            SIGN.idle_label,
                        )
                        accepted = word is not None
                        if accepted:
                            sentence = (sentence + [word])[-_MAX_SENTENCE_WORDS:]
                            headline = f"{word}  {conf:.2f}"
                            report(f"-> {word} ({conf:.2f})    sentence: {' '.join(sentence)}")
                        else:
                            headline = f"?  (best: {best} {conf:.2f})"
                            report(f"-> not recognized (best guess {best}, {conf:.2f})")
                    status = _STATE_TEXT[segmenter.state]
                    progress = segmenter.progress(stamp)
                else:
                    window.append(stamp, vec)
                    progress = 0.0
                    status = "continuous"
                    if window.ready() and now - last_infer >= SIGN.inference_interval_s:
                        last_infer = now
                        word, best, conf = interpret(
                            classifier.probabilities(window.sample()),
                            classifier.labels,
                            SIGN.min_confidence,
                            SIGN.idle_label,
                        )
                        accepted = hands_present and word is not None
                        headline = f"{word}  {conf:.2f}" if accepted else "..."

                if args.headless:
                    if now - last_print >= 0.25:
                        last_print = now
                        bar = f" [{'#' * int(progress * 10):<10}]" if progress > 0 else ""
                        print(
                            f"\rfps={fps:4.1f}  hands={'Y' if hands_present else '-'}  "
                            f"{status}{bar}      ",
                            end="",
                            flush=True,
                        )
                else:
                    image = _draw(
                        frame_rgb, raw, headline, accepted, status, progress, sentence, fps
                    )
                    cv2.imshow("Senseless - live sign demo", image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("c"):
                        sentence = []
    except KeyboardInterrupt:
        pass
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
