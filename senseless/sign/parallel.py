"""Parallel perception: pose and hands models in separate worker processes.

On the Pi 4 every MediaPipe model uses about one core (measured: hands lite 95
ms/frame, pose lite ~100 ms), so running them one after the other wastes the
other cores. ``ParallelBackend`` runs each in its own process (CLAUDE.md: use
multiprocessing, not threads, for CPU-bound stages), so a frame costs
max(pose, hands) instead of their sum.

Data flow per frame (synchronous, one frame in flight):

    main: copy frame -> shared memory -> request (seq, ts) to each worker
    pose worker  / hands worker: read shared memory, run model, reply (seq, result)
    main: wait for both replies -> slot hands -> RawLandmarks

The frame travels through shared memory, so the ~0.9 MB image is not pickled
twice per frame; only small landmark arrays go through the queues, which are the
project's bounded ``DropOldestQueue``. With one frame in flight the queues never
build a backlog, and pairing with ``LatestFrameGrabber`` keeps the frames fresh.

Workers are spawned (not forked) so MediaPipe is only ever initialised inside
them, and they start on the first frame (the shared-memory size is the frame's).
"""

from __future__ import annotations

import functools
import multiprocessing
import queue
import time
import traceback
from collections.abc import Callable
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from senseless.common.config import SIGN
from senseless.common.process import parent_alive
from senseless.common.queue import DropOldestQueue
from senseless.sign.landmarks import (
    PerceptionBackend,
    RawLandmarks,
    make_hands_estimator,
    make_pose_estimator,
    slot_hands,
)

_READY = "ready"
_OK = "ok"
_ERROR = "error"
_STARTUP_TIMEOUT_S = 120.0  # loading MediaPipe models on a Pi takes a few seconds
_FRAME_TIMEOUT_S = 30.0
_POLL_S = 0.25  # how often a waiting main process checks that a worker is alive
_PARENT_POLL_S = 1.0  # how often an idle worker checks that its parent is still alive


def _worker_main(
    factory: Callable[[], Any],
    shm_name: str,
    shape: tuple[int, ...],
    requests: DropOldestQueue,
    results: DropOldestQueue,
) -> None:
    """Worker loop: build one estimator, then process frames from shared memory."""
    try:
        estimator = factory()
    except BaseException:  # noqa: BLE001 -- reported to the main process
        results.put((_ERROR, None, traceback.format_exc()))
        return
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    results.put((_READY, None, None))
    try:
        while True:
            try:
                msg = requests.get(timeout=_PARENT_POLL_S)
            except queue.Empty:
                if parent_alive():
                    continue
                break  # the sign worker died without closing us: exit instead of lingering
            if msg is None:
                break
            seq, timestamp_ms = msg
            try:
                results.put((_OK, seq, estimator.process(frame, timestamp_ms)))
            except BaseException:  # noqa: BLE001 -- reported, worker keeps serving
                results.put((_ERROR, seq, traceback.format_exc()))
    finally:
        estimator.close()
        del frame  # release the buffer before closing the segment
        shm.close()


class ParallelBackend(PerceptionBackend):
    """Runs a pose estimator and a hands estimator in two worker processes.

    ``pose_factory`` / ``hands_factory`` are picklable zero-argument callables that
    build the estimators inside the workers (see ``for_family``). ``pose_stride``
    (default ``SIGN.pose_stride``) runs pose on every Nth frame only and reuses the
    last pose in between; the shoulders it anchors on barely move.
    """

    def __init__(
        self,
        pose_factory: Callable[[], Any],
        hands_factory: Callable[[], Any],
        pose_stride: int | None = None,
        mirror: bool | None = None,
    ) -> None:
        self._factories = {"pose": pose_factory, "hands": hands_factory}
        self._pose_stride = max(1, SIGN.pose_stride if pose_stride is None else pose_stride)
        self._mirror = SIGN.mirror if mirror is None else mirror
        self._ctx = multiprocessing.get_context("spawn")
        self._shape: tuple[int, ...] | None = None
        self._shm: shared_memory.SharedMemory | None = None
        self._view: np.ndarray | None = None
        self._procs: dict[str, Any] = {}
        self._requests: dict[str, DropOldestQueue] = {}
        self._results: dict[str, DropOldestQueue] = {}
        self._seq = 0
        self._last_pose: np.ndarray | None = None

    @classmethod
    def for_family(cls, family: str) -> ParallelBackend:
        """Parallel version of the "tasks" or "lite" backend."""
        return cls(
            functools.partial(make_pose_estimator, family),
            functools.partial(make_hands_estimator, family),
        )

    def _start(self, shape: tuple[int, ...]) -> None:
        self._shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
        self._view = np.ndarray(shape, dtype=np.uint8, buffer=self._shm.buf)
        self._shape = shape
        for name, factory in self._factories.items():
            self._requests[name] = DropOldestQueue(1, backend=self._ctx.Queue)
            self._results[name] = DropOldestQueue(2, backend=self._ctx.Queue)
            proc = self._ctx.Process(
                target=_worker_main,
                args=(factory, self._shm.name, shape, self._requests[name], self._results[name]),
                name=f"senseless-{name}",
                daemon=True,
            )
            proc.start()
            self._procs[name] = proc
        for name in self._factories:
            self._await(name, _READY, _STARTUP_TIMEOUT_S)

    def _await(self, name: str, seq: int | str, timeout_s: float) -> Any:
        """Wait for worker ``name``'s reply to ``seq`` (or its ready signal)."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                kind, got_seq, payload = self._results[name].get(timeout=_POLL_S)
            except queue.Empty:
                proc = self._procs[name]
                if not proc.is_alive():
                    raise RuntimeError(f"{name} worker exited (code {proc.exitcode})") from None
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"{name} worker gave no reply in {timeout_s:.0f} s"
                    ) from None
                continue
            if kind == _ERROR:
                raise RuntimeError(f"{name} worker failed:\n{payload}")
            if kind == _READY:
                if seq == _READY:
                    return None
                continue
            if got_seq == seq:
                return payload
            # A stale reply (e.g. left over after an error on the other worker): skip it.

    def extract(self, frame_rgb: np.ndarray, timestamp_ms: int) -> RawLandmarks:
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        if self._shm is None:
            self._start(frame.shape)
        elif frame.shape != self._shape:
            raise ValueError(
                f"frame shape {frame.shape} differs from the first frame's {self._shape}"
            )
        self._view[...] = frame

        seq = self._seq
        self._seq += 1
        run_pose = seq % self._pose_stride == 0
        if run_pose:
            self._requests["pose"].put((seq, timestamp_ms))
        self._requests["hands"].put((seq, timestamp_ms))
        if run_pose:
            self._last_pose = self._await("pose", seq, _FRAME_TIMEOUT_S)
        hands = self._await("hands", seq, _FRAME_TIMEOUT_S)
        left, right = slot_hands(hands, self._mirror)
        return RawLandmarks(left_hand=left, right_hand=right, pose=self._last_pose)

    def close(self) -> None:
        for req in self._requests.values():
            req.put(None)  # never blocks (drop-oldest)
        for proc in self._procs.values():
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        for q in (*self._requests.values(), *self._results.values()):
            q.close()
        self._view = None
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None
