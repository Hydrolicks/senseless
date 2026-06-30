"""Interactive sign data-collection recorder (DEV TOOL, PC).

Keypress-per-take recorder with a live monitoring window. For one label, press
SPACE to record a single ``window_length``-frame window of normalized features;
it is saved under ``data/<label>/`` via ``collect.dataset``. Reuses the same
camera + landmark pipeline as inference, so collected data matches what the
model sees at runtime (the windows are already normalized).

Controls: SPACE = record one take | q or Esc = quit.

Run via ``python -m senseless.collect`` (see instructions.md, step 5).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from senseless.collect import dataset
from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN
from senseless.sign import capture, landmarks

# Landmark dot colors (BGR).
_LEFT_HAND = (0, 255, 0)
_RIGHT_HAND = (255, 128, 0)
_POSE = (0, 0, 255)
_SPACE_KEY = 32


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
    parser = argparse.ArgumentParser(description="Record sign samples (keypress per take).")
    parser.add_argument("--label", required=True, help="Vocabulary word to record.")
    parser.add_argument("--samples", type=int, default=30, help="Target number of takes.")
    parser.add_argument("--backend", choices=["tasks", "holistic"], default=None)
    parser.add_argument("--source", type=int, default=0, help="Webcam index.")
    args = parser.parse_args()

    import cv2

    backend = landmarks.create_backend(args.backend)
    src = capture.open_frame_source(source=args.source)
    count = dataset.count_samples(args.label)
    window: list[np.ndarray] = []
    recording = False
    start = time.perf_counter()
    window_name = f"Senseless collect - {args.label}"
    print(f"Recording '{args.label}'. SPACE = take, q = quit. Already have {count}.")

    try:
        with backend, src:
            for frame_rgb in src.frames():
                ts_ms = int((time.perf_counter() - start) * 1000)
                raw = backend.extract(frame_rgb, ts_ms)
                vec = landmarks.frame_landmarks_to_vector(raw)

                if recording:
                    window.append(vec)
                    if len(window) >= SIGN.window_length:
                        path = dataset.save_window(np.stack(window), args.label)
                        count += 1
                        recording = False
                        window = []
                        print(f"saved {path.name}  ({count}/{args.samples})")

                rec_txt = f"   REC {len(window)}/{SIGN.window_length}" if recording else ""
                lines = [
                    f"{args.label}    saved {count}/{args.samples}",
                    f"detected: {'yes' if vec.any() else 'NO'}{rec_txt}",
                    "SPACE = record a take    |    q = quit",
                ]
                cv2.imshow(window_name, _draw(frame_rgb, raw, lines, recording))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == _SPACE_KEY and not recording:
                    recording = True
                    window = []
    finally:
        cv2.destroyAllWindows()
    print(f"Done. {count} samples for '{args.label}' in {dataset.label_dir(args.label)}.")
