"""Light import check for the live demo (no GUI/camera/model loaded here).

cv2, MediaPipe, and the TFLite interpreter are all lazy-loaded, so the module
must import without them.
"""

from senseless.sign import demo


def test_demo_imports() -> None:
    assert hasattr(demo, "main")
    assert hasattr(demo, "_draw")
