"""Camera frame sources for the sign channel.

One interface (``FrameSource``) with two implementations behind it:

* ``OpenCVSource``  -- ``cv2.VideoCapture`` for laptop/dev (webcam or video file).
* ``PiCameraSource`` -- ``picamera2`` for the Pi Camera Module 3 on the Pi.

Both lazy-import their dependency, so this module imports anywhere (neither cv2
nor picamera2 is needed just to import it). Every source yields a contiguous
RGB ``uint8`` array shaped ``(H, W, 3)`` at ``config.CAMERA`` resolution, which
is exactly what the MediaPipe backends in ``landmarks.py`` expect.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from senseless.common.config import CAMERA


class FrameSource(ABC):
    """A source of RGB frames. Use as a context manager or via ``frames()``."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next RGB frame, or ``None`` when the stream is exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying camera/file handle."""

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the source is exhausted."""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame


class OpenCVSource(FrameSource):
    """Frames from an OpenCV ``VideoCapture`` (webcam index or video file path)."""

    def __init__(self, source: int | str = 0) -> None:
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)
        self._cap.set(cv2.CAP_PROP_FPS, CAMERA.framerate)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open OpenCV video source: {source!r}")

    def read(self) -> np.ndarray | None:
        ok, frame_bgr = self._cap.read()
        if not ok:
            return None
        return np.ascontiguousarray(self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        self._cap.release()


class PiCameraSource(FrameSource):
    """Frames from the Pi Camera Module 3 via picamera2.

    NOTE: picamera2's channel ordering for the "RGB888" format is a known
    gotcha and can come back BGR depending on libcamera/version. If colors look
    swapped on the Pi, flip ``config.CAMERA.pixel_format`` to "BGR888" (or add a
    cvtColor here). Verify on-device.
    """

    def __init__(self) -> None:
        from picamera2 import Picamera2

        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": (CAMERA.width, CAMERA.height), "format": CAMERA.pixel_format}
        )
        self._picam2.configure(config)
        self._picam2.start()

    def read(self) -> np.ndarray | None:
        return np.ascontiguousarray(self._picam2.capture_array())

    def close(self) -> None:
        self._picam2.stop()
        self._picam2.close()


def open_frame_source(prefer: str | None = None, source: int | str = 0) -> FrameSource:
    """Open the best available frame source.

    ``prefer="picamera"`` or ``"opencv"`` forces a backend; ``None`` auto-selects
    picamera2 if it is importable (i.e. on the Pi), otherwise OpenCV (dev laptop).
    """
    if prefer == "picamera":
        return PiCameraSource()
    if prefer == "opencv":
        return OpenCVSource(source)
    try:
        import picamera2  # noqa: F401
    except ImportError:
        return OpenCVSource(source)
    return PiCameraSource()
