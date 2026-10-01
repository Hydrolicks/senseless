"""Fake mode workers for the ModeController tests.

They live in their own module (not a ``test_`` file) because the controller
spawns them in child processes, which must import them by name.
"""

import os
import time

import numpy as np

from senseless.common.events import SignStatus, WorkerError, WorkerReady


def ready_worker(frames, events, stop) -> None:
    events.put(WorkerReady("fake"))
    while not stop.is_set():
        frames.put(np.zeros((240, 320, 3), dtype=np.uint8))
        events.put(SignStatus("idle", 0.0, False, 10.0))
        time.sleep(0.05)


def stubborn_worker(frames, events, stop) -> None:
    events.put(WorkerReady("fake"))
    while True:  # ignores the stop request
        time.sleep(0.1)


def crashing_worker(frames, events, stop) -> None:
    raise SystemExit(3)


def one_shot_worker(frames, events, stop) -> None:
    events.put(WorkerReady("fake"))
    events.put(WorkerError("boom"))


def dies_with_a_frame_queued_worker(frames, events, stop) -> None:
    events.put(WorkerReady("fake"))
    frames.put(np.zeros((4, 4, 3), dtype=np.uint8))
    time.sleep(0.3)  # the small frame is now in the pipe
    os._exit(1)  # a native crash: no clean shutdown
