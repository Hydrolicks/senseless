"""Light import check for the collection recorder (no camera/GUI here).

cv2 is lazy-imported inside the recorder, so the module must import without it.
"""

from senseless.collect import recorder


def test_recorder_imports_without_cv2() -> None:
    assert hasattr(recorder, "main")
    assert hasattr(recorder, "_draw")


def test_camera_choice_maps_to_the_frame_source_preference() -> None:
    assert recorder.camera_preference("auto") is None  # picamera2 on the Pi, else OpenCV
    assert recorder.camera_preference("opencv") == "opencv"  # USB webcam on the Pi (C920)
    assert recorder.camera_preference("picamera") == "picamera"
