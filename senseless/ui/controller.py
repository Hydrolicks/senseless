"""Runs one mode worker at a time for the touchscreen app.

Each worker is a spawned process with two queues back to the GUI: ``frames``
(preview images, max 2, newest wins) and ``events`` (status, words, speech
text, errors; max RUNTIME.transcript_queue_maxsize). Stopping is non-blocking
so the Tk loop never freezes: ``request_stop()`` then ``poll_stopped()`` on each
GUI tick; a worker still alive after the timeout is terminated. Events drained
while stopping are kept and returned by the next ``drain_events()``.
"""

from __future__ import annotations

import multiprocessing
import queue
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from senseless.common.config import RUNTIME, UI
from senseless.common.queue import DropOldestQueue


def sign_entry(frames, events, stop) -> None:
    """Process target for Sign mode."""
    from senseless.sign.worker import run_sign_worker

    run_sign_worker(frames, events, stop)


def speech_entry(frames, events, stop) -> None:
    """Process target for Speech mode (no preview frames)."""
    from senseless.asr.worker import run_speech_worker

    run_speech_worker(events, stop)


class ModeController:
    """Starts, stops and watches the single worker process of the active mode."""

    def __init__(
        self,
        targets: dict[str, Callable] | None = None,
        switch_timeout_s: float | None = None,
    ) -> None:
        self._targets = (
            targets if targets is not None else {"sign": sign_entry, "speech": speech_entry}
        )
        self._timeout = UI.switch_timeout_s if switch_timeout_s is None else switch_timeout_s
        self._ctx = multiprocessing.get_context("spawn")
        self.mode: str | None = None
        self._proc: Any = None
        self._frames: DropOldestQueue | None = None
        self._events: DropOldestQueue | None = None
        self._stop: Any = None
        self._deadline: float | None = None  # set once a stop was requested
        # Events drained while stopping; handed out by the next drain_events().
        self._pending: list = []

    def start(self, mode: str) -> None:
        if self._proc is not None:
            raise RuntimeError("a worker is already running; stop it first")
        self._frames = DropOldestQueue(UI.frames_queue_maxsize, backend=self._ctx.Queue)
        self._events = DropOldestQueue(RUNTIME.transcript_queue_maxsize, backend=self._ctx.Queue)
        self._stop = self._ctx.Event()
        # daemon=False: the sign worker starts its own model processes (ParallelBackend),
        # and daemonic processes are not allowed to have children.
        self._proc = self._ctx.Process(
            target=self._targets[mode],
            args=(self._frames, self._events, self._stop),
            name=f"senseless-{mode}",
            daemon=False,
        )
        self._proc.start()
        self.mode = mode
        self._deadline = None

    def request_stop(self) -> None:
        if self._proc is not None and self._deadline is None:
            self._stop.set()
            self._deadline = time.monotonic() + self._timeout

    def poll_stopped(self) -> bool:
        """True once no worker is running. Non-blocking; terminates after the timeout.

        Events drained here are kept for the next ``drain_events()``.
        """
        if self._proc is None:
            return True
        self._pending.extend(self._drain_queue())  # keep pipes flowing; keep the events
        self.latest_frame()
        if self._proc.is_alive():
            if self._deadline is None or time.monotonic() < self._deadline:
                return False
            self._proc.terminate()
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=2.0)
        self._cleanup()
        return True

    def has_exited(self) -> bool:
        """The worker died without being asked to stop (crash, killed, error exit)."""
        return self._proc is not None and self._deadline is None and not self._proc.is_alive()

    def exitcode(self) -> int | None:
        return None if self._proc is None else self._proc.exitcode

    def drain_events(self) -> list:
        """Events buffered while stopping, then whatever is queued now, in order."""
        out, self._pending = self._pending, []
        out.extend(self._drain_queue())
        return out

    def _drain_queue(self) -> list:
        out: list = []
        if self._events is None:
            return out
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def latest_frame(self) -> np.ndarray | None:
        frame = None
        if self._frames is None:
            return None
        while True:
            try:
                frame = self._frames.get_nowait()
            except queue.Empty:
                return frame

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop the worker, waiting up to ``timeout`` (blocking; for app exit)."""
        if self._proc is None:
            return
        self.request_stop()
        end = time.monotonic() + (self._timeout if timeout is None else timeout)
        while self._proc.is_alive() and time.monotonic() < end:
            self._pending.extend(self._drain_queue())
            self.latest_frame()
            time.sleep(0.05)
        self._deadline = 0.0  # anything still running gets terminated now
        self.poll_stopped()

    def _cleanup(self) -> None:
        for q in (self._frames, self._events):
            if q is not None:
                q.close()
        self._proc = self._frames = self._events = self._stop = None
        self.mode = None
        self._deadline = None
