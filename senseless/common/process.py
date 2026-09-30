"""Helpers for worker processes spawned by the GUI or by another worker."""

from __future__ import annotations

import multiprocessing


def parent_alive() -> bool:
    """False once the process that spawned this one has died.

    Workers check this next to their stop event so a GUI (or sign worker) that is
    killed instead of closed cleanly does not leave them holding the camera and
    the CPU. True when not running as a spawned child (parent_process() is None).
    """
    parent = multiprocessing.parent_process()
    return parent is None or parent.is_alive()
