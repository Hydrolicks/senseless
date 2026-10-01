"""TDD specs for ParallelBackend (sign/parallel.py): pose and hands in separate
worker processes, frames passed through shared memory.

Real MediaPipe is replaced by the fakes in ``parallel_fakes`` (built inside the
spawned workers), so these tests cover the plumbing: frame transfer, joining the
two results, pose striding, failure reporting and shutdown.
"""

import numpy as np
import parallel_fakes as fakes
import pytest

from senseless.sign.parallel import ParallelBackend


def _frame(red: int, green: int) -> np.ndarray:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[..., 0] = red
    frame[..., 1] = green
    return frame


@pytest.fixture(scope="module")
def backend():
    with ParallelBackend(fakes.make_fake_pose, fakes.make_fake_hands) as be:
        yield be


def test_extract_joins_pose_and_hands_from_the_workers(backend) -> None:
    raw = backend.extract(_frame(7, 9), timestamp_ms=0)
    assert np.all(raw.pose == 7.0)
    assert np.all(raw.left_hand == 9.0)
    assert raw.right_hand is None


def test_every_frame_reaches_the_workers_fresh(backend) -> None:
    for i, value in enumerate((11, 22, 33, 44)):
        raw = backend.extract(_frame(value, value + 1), timestamp_ms=100 + 33 * i)
        assert raw.pose[0, 0] == value
        assert raw.left_hand[0, 0] == value + 1


def test_frame_shape_change_is_rejected(backend) -> None:
    with pytest.raises(ValueError, match="shape"):
        backend.extract(np.zeros((10, 10, 3), dtype=np.uint8), timestamp_ms=10_000)


def test_pose_stride_reuses_the_previous_pose() -> None:
    with ParallelBackend(fakes.make_fake_pose, fakes.make_fake_hands, pose_stride=2) as be:
        poses = [
            be.extract(_frame(v, 0), timestamp_ms=33 * i).pose for i, v in enumerate((1, 2, 3, 4))
        ]
    # frames 0 and 2 run pose; frames 1 and 3 reuse the pose from the frame before.
    assert [float(p[0, 0]) for p in poses] == [1.0, 1.0, 3.0, 3.0]


def test_worker_error_surfaces_as_runtime_error() -> None:
    with ParallelBackend(fakes.make_crashing_pose, fakes.make_fake_hands) as be:
        with pytest.raises(RuntimeError, match="pose model exploded"):
            be.extract(_frame(1, 1), timestamp_ms=0)


def test_close_stops_the_worker_processes() -> None:
    be = ParallelBackend(fakes.make_fake_pose, fakes.make_fake_hands)
    be.extract(_frame(1, 1), timestamp_ms=0)
    procs = list(be._procs.values())
    assert all(p.is_alive() for p in procs)
    be.close()
    assert not any(p.is_alive() for p in procs)


def test_an_orphaned_worker_loop_exits_when_its_parent_dies(monkeypatch) -> None:
    import queue
    import threading
    from multiprocessing import shared_memory

    from senseless.common.queue import DropOldestQueue
    from senseless.sign import parallel

    class DeadParent:
        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr("multiprocessing.parent_process", lambda: DeadParent())
    monkeypatch.setattr(parallel, "_PARENT_POLL_S", 0.05)
    shm = shared_memory.SharedMemory(create=True, size=48 * 64 * 3)
    try:
        requests = DropOldestQueue(1, backend=queue.Queue)  # nothing is ever sent
        results = DropOldestQueue(2, backend=queue.Queue)
        worker = threading.Thread(
            target=parallel._worker_main,
            args=(fakes.make_fake_hands, shm.name, (48, 64, 3), requests, results),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
    finally:
        shm.close()
        shm.unlink()
