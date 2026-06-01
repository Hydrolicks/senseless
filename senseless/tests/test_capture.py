"""Light checks for the camera abstraction.

Real frame I/O needs hardware, so it is not unit-tested. These just confirm the
module imports without cv2/picamera2 installed and that the interface is wired up.
"""

import pytest

from senseless.sign import capture


def test_module_imports_without_camera_deps() -> None:
    # Importing must not require cv2 or picamera2 (they are lazy-imported).
    assert hasattr(capture, "open_frame_source")
    assert hasattr(capture, "FrameSource")


def test_frame_source_is_abstract() -> None:
    with pytest.raises(TypeError):
        capture.FrameSource()  # type: ignore[abstract]
