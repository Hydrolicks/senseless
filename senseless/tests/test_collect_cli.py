"""Light import check for the collection recorder (no camera/GUI here).

cv2 is lazy-imported inside the recorder, so the module must import without it.
"""

from senseless.collect import recorder


def test_recorder_imports_without_cv2() -> None:
    assert hasattr(recorder, "main")
    assert hasattr(recorder, "_draw")
