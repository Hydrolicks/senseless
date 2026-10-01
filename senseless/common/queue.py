"""Bounded, drop-oldest queue shared between pipeline stages.

CLAUDE.md mandates bounded ring buffers with a DROP-OLDEST policy (never
unbounded queues) so end-to-end latency stays bounded: under backpressure the
consumer should get the *freshest* data, not a growing backlog.

``DropOldestQueue.put`` therefore never blocks -- when the queue is full it
discards the oldest item (and returns it, e.g. for drop metrics) before
enqueuing the new one.

The backing queue is injectable. The default is ``multiprocessing.Queue`` (the
stages run as separate processes -- 4-core budget, GIL). Tests inject a thread
``queue.Queue``, which shares the same ``put_nowait``/``get_nowait``/``Full``/
``Empty`` contract, so the drop-oldest policy is verified deterministically.

Drop-oldest assumes a single producer per queue (which is our topology: one
camera process feeds the frame queue, one audio process feeds the audio queue,
etc.). With multiple concurrent producers the evict-then-put step can race.

With the multiprocessing backend, full-detection is approximate (items pass
through a feeder thread + OS pipe), so drops happen near -- not exactly at --
maxsize; the backlog stays bounded either way. The deterministic drop-oldest
policy is verified against the thread-queue backend in the tests.
"""

from __future__ import annotations

from collections.abc import Callable
from queue import Empty, Full
from typing import Any


def _default_backend(maxsize: int) -> Any:
    import multiprocessing

    return multiprocessing.Queue(maxsize)


class DropOldestQueue:
    """A bounded queue whose ``put`` drops the oldest item instead of blocking."""

    def __init__(self, maxsize: int, backend: Callable[[int], Any] | None = None) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._q = (backend or _default_backend)(maxsize)

    def put(self, item: Any) -> Any:
        """Enqueue ``item`` without blocking; return the evicted item or None."""
        try:
            self._q.put_nowait(item)
            return None
        except Full:
            try:
                dropped = self._q.get_nowait()
            except Empty:
                dropped = None
            self._q.put_nowait(item)
            return dropped

    def get(self, timeout: float | None = None) -> Any:
        """Block (up to ``timeout`` seconds) for the next item."""
        return self._q.get(timeout=timeout)

    def get_nowait(self) -> Any:
        """Return the next item or raise ``queue.Empty``."""
        return self._q.get_nowait()

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    def full(self) -> bool:
        return self._q.full()

    def cancel_join_thread(self) -> None:
        """Let this process exit without flushing items nobody will read.

        A multiprocessing.Queue joins its feeder thread at interpreter exit; if the
        reader is gone and the pipe is full, that join never returns. Use it only
        when the reader is known to be dead (queued items are then lost).
        """
        canceller = getattr(self._q, "cancel_join_thread", None)
        if callable(canceller):
            canceller()

    def close(self) -> None:
        """Release the backing queue if it supports it (multiprocessing.Queue)."""
        closer = getattr(self._q, "close", None)
        if callable(closer):
            closer()
