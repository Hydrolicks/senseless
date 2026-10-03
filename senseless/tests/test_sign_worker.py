"""Sign worker loop with fake parts: no camera, no MediaPipe, no TFLite.

The worker runs in-process here (queue.Queue backends, threading.Event); in the
app it runs in its own spawned process with multiprocessing queues.
"""

import queue
import threading

import numpy as np

from senseless.common.events import SignResult, SignStatus, WorkerError, WorkerReady
from senseless.common.queue import DropOldestQueue
from senseless.sign.capture import FrameSource
from senseless.sign.landmarks import PerceptionBackend, RawLandmarks
from senseless.sign.worker import SignParts, run_sign_worker


def _pose() -> np.ndarray:
    pose = np.full((33, 3), 0.5, dtype=np.float32)
    pose[11] = (0.40, 0.50, 0.0)  # shoulders set the normalization frame
    pose[12] = (0.60, 0.50, 0.0)
    return pose


class FakeCamera(FrameSource):
    """``n`` frames at 10 FPS (synthetic stamps), then the stream ends; or endless."""

    def __init__(self, n: int | None) -> None:
        self.n, self.i = n, 0

    def read_stamped(self):
        if self.n is not None and self.i >= self.n:
            return None
        stamp = self.i / 10.0
        self.i += 1
        return np.zeros((48, 64, 3), dtype=np.uint8), stamp

    def read(self):
        item = self.read_stamped()
        return None if item is None else item[0]

    def close(self) -> None:
        pass


class FakeBackend(PerceptionBackend):
    """Hands in view from 0.5 s on (frame index 5)."""

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, frame_rgb, timestamp_ms):
        hand = np.full((21, 3), 0.3, dtype=np.float32) if self.calls >= 5 else None
        self.calls += 1
        return RawLandmarks(left_hand=hand, right_hand=None, pose=_pose())

    def close(self) -> None:
        pass


class FakeClassifier:
    labels = ["HELLO", "IDLE"]

    def probabilities(self, window):
        assert window.shape == (45, 153)
        return np.array([0.9, 0.1], dtype=np.float32)


def _queues():
    return DropOldestQueue(2, backend=queue.Queue), DropOldestQueue(10_000, backend=queue.Queue)


def _drain(q) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_one_sign_gives_one_result_and_the_stream_end_is_reported() -> None:
    frames, events = _queues()
    parts = SignParts(lambda: FakeCamera(30), FakeBackend, FakeClassifier)
    run_sign_worker(frames, events, threading.Event(), parts)
    got = _drain(events)
    assert got[0] == WorkerReady("sign")
    results = [e for e in got if isinstance(e, SignResult)]
    assert results == [SignResult("HELLO", "HELLO", results[0].confidence)]
    assert any(isinstance(e, SignStatus) and e.hands for e in got)
    assert got[-1] == WorkerError("Camera stream ended.")
    preview = _drain(frames)[-1]
    assert preview.shape == (240, 320, 3) and preview.dtype == np.uint8


def test_stop_event_ends_the_loop() -> None:
    frames, events = _queues()
    stop = threading.Event()
    parts = SignParts(lambda: FakeCamera(None), FakeBackend, FakeClassifier)
    worker = threading.Thread(target=run_sign_worker, args=(frames, events, stop, parts))
    worker.start()
    stop.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()


def test_a_camera_that_fails_to_open_is_reported() -> None:
    def broken_camera():
        raise RuntimeError("Could not open OpenCV video source: 0")

    frames, events = _queues()
    parts = SignParts(broken_camera, FakeBackend, FakeClassifier)
    run_sign_worker(frames, events, threading.Event(), parts)
    (error,) = _drain(events)
    assert isinstance(error, WorkerError)
    assert error.message.startswith("Camera not found.")


class _DeadParent:
    def is_alive(self) -> bool:
        return False


def test_an_orphaned_worker_exits_even_though_stop_was_never_set(monkeypatch) -> None:
    monkeypatch.setattr("multiprocessing.parent_process", lambda: _DeadParent())
    frames, events = _queues()
    parts = SignParts(lambda: FakeCamera(None), FakeBackend, FakeClassifier)
    worker = threading.Thread(
        target=run_sign_worker, args=(frames, events, threading.Event(), parts), daemon=True
    )
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive()


def test_a_failure_logs_its_traceback_as_well_as_telling_the_gui(capsys) -> None:
    def broken_camera():
        raise RuntimeError("camera exploded")

    frames, events = _queues()
    parts = SignParts(broken_camera, FakeBackend, FakeClassifier)
    run_sign_worker(frames, events, threading.Event(), parts)
    assert "RuntimeError: camera exploded" in capsys.readouterr().err


def test_a_model_load_failure_logs_its_traceback(capsys) -> None:
    def no_classifier():
        raise FileNotFoundError("models/sign_gru_int8.tflite")

    frames, events = _queues()
    parts = SignParts(lambda: FakeCamera(1), FakeBackend, no_classifier)
    run_sign_worker(frames, events, threading.Event(), parts)
    assert "FileNotFoundError" in capsys.readouterr().err


def test_a_crash_in_the_loop_logs_its_traceback(capsys) -> None:
    class BadClassifier(FakeClassifier):
        def probabilities(self, window):
            raise ValueError("bad window")

    frames, events = _queues()
    parts = SignParts(lambda: FakeCamera(30), FakeBackend, BadClassifier)
    run_sign_worker(frames, events, threading.Event(), parts)
    assert "ValueError: bad window" in capsys.readouterr().err


def test_the_app_uses_lite_where_mediapipe_still_has_solutions(monkeypatch) -> None:
    from senseless.sign import landmarks, worker

    monkeypatch.setattr(landmarks, "solutions_available", lambda: True)
    assert worker.app_backend_name("lite") == "lite"


def test_the_app_falls_back_to_tasks_where_solutions_is_gone(monkeypatch) -> None:
    # MediaPipe 0.10.35 on the dev PC has no mp.solutions, so "lite" cannot start there.
    from senseless.sign import landmarks, worker

    monkeypatch.setattr(landmarks, "solutions_available", lambda: False)
    assert worker.app_backend_name("lite") == "tasks"
    assert worker.app_backend_name("tasks") == "tasks"


def test_solutions_available_checks_the_installed_mediapipe(monkeypatch) -> None:
    import sys
    import types

    from senseless.sign import landmarks

    old = types.SimpleNamespace(solutions=types.SimpleNamespace(hands=object()))
    monkeypatch.setitem(sys.modules, "mediapipe", old)
    assert landmarks.solutions_available()
    monkeypatch.setitem(sys.modules, "mediapipe", types.SimpleNamespace(tasks=object()))
    assert not landmarks.solutions_available()
