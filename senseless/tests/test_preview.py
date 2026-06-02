"""Light smoke check for the dev preview tool (no GUI/camera here).

The heavy deps (cv2, MediaPipe) are lazy-imported inside the functions, so the
module must import without them present.
"""

from senseless.sign import preview


def test_preview_module_imports() -> None:
    assert hasattr(preview, "main")
    assert hasattr(preview, "_annotate")
