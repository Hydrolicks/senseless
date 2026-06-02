"""Live sign-perception preview (DEV TOOL, PC only).

Opens a webcam (or a video file), runs the configured perception backend, and
shows each frame with the detected hand/pose landmarks drawn on it plus a status
overlay: which hands were found, whether the pose is present, and whether the
normalized 153-vector is active or all-zero ("no detection"). Use it to eyeball
tracking quality and check left/right hand assignment before collecting data.

This is a debugging aid, not part of the runtime pipeline. It needs a GUI
(OpenCV highgui) and MediaPipe, so it runs on the dev PC, not the headless Pi.

Examples
--------
    python -m senseless.sign.preview                  # default webcam, tasks backend
    python -m senseless.sign.preview --backend holistic
    python -m senseless.sign.preview --video clip.mp4
    python -m senseless.sign.preview --source 1       # second camera

Press 'q' or Esc to quit.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN
from senseless.sign import capture, landmarks

# Overlay colors are BGR (OpenCV).
_LEFT_HAND_COLOR = (0, 255, 0)
_RIGHT_HAND_COLOR = (255, 128, 0)
_POSE_COLOR = (0, 0, 255)
_ORIGIN_COLOR = (0, 255, 255)


def _annotate(
    frame_rgb: np.ndarray,
    raw: landmarks.RawLandmarks,
    vec: np.ndarray,
    backend_name: str,
    fps: float,
) -> np.ndarray:
    """Draw landmarks + a status overlay; return a BGR image for cv2.imshow."""
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    def _dots(points: np.ndarray | None, color: tuple[int, int, int]) -> None:
        if points is None:
            return
        for x, y, _z in points:
            cv2.circle(bgr, (int(x * w), int(y * h)), 4, color, -1)

    _dots(raw.left_hand, _LEFT_HAND_COLOR)
    _dots(raw.right_hand, _RIGHT_HAND_COLOR)
    if raw.pose is not None:
        pose = np.asarray(raw.pose)
        _dots(pose[list(ls.POSE_LANDMARK_INDICES)], _POSE_COLOR)
        left_sh = pose[ls.LEFT_SHOULDER_INDEX]
        right_sh = pose[ls.RIGHT_SHOULDER_INDEX]
        ox = int((left_sh[0] + right_sh[0]) / 2 * w)
        oy = int((left_sh[1] + right_sh[1]) / 2 * h)
        cv2.drawMarker(bgr, (ox, oy), _ORIGIN_COLOR, cv2.MARKER_CROSS, 18, 2)

    nz = int(np.count_nonzero(vec))
    feat_line = (
        f"feature: ACTIVE  nz={nz}/{ls.FEATURE_DIM}" if nz else "feature: ALL-ZERO (no detection)"
    )
    lines = [
        f"backend={backend_name}  fps={fps:4.1f}  mirror={SIGN.mirror}",
        f"L hand: {'Y' if raw.left_hand is not None else '-'}   "
        f"R hand: {'Y' if raw.right_hand is not None else '-'}   "
        f"pose: {'Y' if raw.pose is not None else '-'}",
        feat_line,
        "L/R swapped? set SIGN.mirror=True in config.py    |    press q to quit",
    ]
    y = 24
    for text in lines:
        cv2.putText(bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 26
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(description="Live sign-perception preview (dev tool).")
    parser.add_argument("--backend", choices=["tasks", "holistic"], default=None)
    parser.add_argument("--source", type=int, default=0, help="Webcam index.")
    parser.add_argument("--video", default=None, help="Video file path (instead of webcam).")
    args = parser.parse_args()

    import cv2

    backend = landmarks.create_backend(args.backend)
    src = (
        capture.OpenCVSource(args.video)
        if args.video
        else capture.open_frame_source(source=args.source)
    )

    window = "Senseless - sign perception preview"
    start = time.perf_counter()
    prev = start
    fps = 0.0
    try:
        with backend, src:
            for frame_rgb in src.frames():
                ts_ms = int((time.perf_counter() - start) * 1000)
                raw = backend.extract(frame_rgb, ts_ms)
                vec = landmarks.frame_landmarks_to_vector(raw)

                now = time.perf_counter()
                dt = now - prev
                prev = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

                cv2.imshow(window, _annotate(frame_rgb, raw, vec, type(backend).__name__, fps))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
