"""A worker whose GUI is killed must exit, even with an unread preview frame queued.

Real processes: this test -> fake GUI -> worker. The fake GUI starts the worker,
waits until it has queued a frame larger than an OS pipe buffer, then dies without
cleaning up. Without ``release_if_orphaned`` the worker's interpreter blocks at exit
joining the queue feeder thread, which can never flush into a pipe nobody reads.
"""

import multiprocessing
import os
import signal
import time

import numpy as np

from senseless.common.process import parent_alive
from senseless.common.queue import DropOldestQueue
from senseless.ui.controller import release_if_orphaned


def _worker(frames, events, stop, started, report) -> None:
    report.put(os.getpid())  # straight to the test, so it can always clean up
    try:
        frames.put(np.zeros((240, 320, 3), dtype=np.uint8))  # 230 KB > a 64 KB pipe buffer
        started.set()
        while not stop.is_set() and parent_alive():
            time.sleep(0.05)
    finally:
        release_if_orphaned(frames, events)


def _fake_gui(report) -> None:
    ctx = multiprocessing.get_context("spawn")
    frames = DropOldestQueue(2, backend=ctx.Queue)
    events = DropOldestQueue(32, backend=ctx.Queue)
    stop, started = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_worker, args=(frames, events, stop, started, report), daemon=False)
    proc.start()
    started.wait(60)
    time.sleep(0.5)  # let the feeder thread start writing the frame
    os._exit(0)  # a crash: no stop event, no queue draining, no join


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_a_worker_exits_after_its_gui_dies_with_a_frame_still_queued() -> None:
    ctx = multiprocessing.get_context("spawn")
    report = ctx.Queue()
    gui = ctx.Process(target=_fake_gui, args=(report,))
    gui.start()
    pid = report.get(timeout=90)
    gui.join(10)
    try:
        deadline = time.monotonic() + 15
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _pid_alive(pid), "orphaned worker is stuck at exit"
    finally:
        if _pid_alive(pid):
            os.kill(pid, signal.SIGTERM)
