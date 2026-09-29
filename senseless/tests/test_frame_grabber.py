"""TDD specs for LatestFrameGrabber (sign/capture.py).

The grabber reads a FrameSource on a background thread and keeps only the newest
frame: the drop-oldest policy applied to the camera, so a slow consumer (the Pi
at ~10 FPS) never works through a backlog of stale frames. Uses a fake source
whose frames carry their sequence number.
"""

import threading
import time

import numpy as np
import pytest

from senseless.sign.capture import FrameSource, LatestFrameGrabber


class CountingSource(FrameSource):
    """Frames 0..n-1 (value = index), one every ``interval`` seconds, then None."""

    def __init__(self, n: int, interval: float = 0.001, fail_at: int | None = None) -> None:
        self.n, self.interval, self.fail_at = n, interval, fail_at
        self.i = 0
        self.closed = threading.Event()

    def read(self) -> np.ndarray | None:
        if self.i >= self.n:
            return None
        if self.fail_at is not None and self.i == self.fail_at:
            raise OSError("camera unplugged")
        time.sleep(self.interval)
        frame = np.full((2, 2, 3), self.i % 256, dtype=np.uint8)
        self.i += 1
        return frame

    def close(self) -> None:
        self.closed.set()


def _drain(grabber: LatestFrameGrabber, delay: float) -> list[tuple[int, float]]:
    out = []
    while (item := grabber.read_stamped()) is not None:
        frame, stamp = item
        out.append((int(frame[0, 0, 0]), stamp))
        time.sleep(delay)
    return out


def test_slow_reader_gets_newest_frames_and_skips_stale_ones() -> None:
    with LatestFrameGrabber(CountingSource(200)) as grabber:
        got = _drain(grabber, delay=0.01)
    values = [v for v, _ in got]
    assert values == sorted(set(values))  # strictly increasing: never a repeat
    assert max(b - a for a, b in zip(values, values[1:], strict=False)) > 1  # stale ones dropped


def test_timestamps_increase() -> None:
    with LatestFrameGrabber(CountingSource(50)) as grabber:
        stamps = [s for _, s in _drain(grabber, delay=0.002)]
    assert len(stamps) > 1
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))


def test_read_returns_none_once_the_source_ends() -> None:
    with LatestFrameGrabber(CountingSource(3)) as grabber:
        _drain(grabber, delay=0.0)
        assert grabber.read() is None


def test_source_error_is_raised_to_the_reader() -> None:
    with LatestFrameGrabber(CountingSource(100, fail_at=5)) as grabber:
        with pytest.raises(OSError, match="unplugged"):
            _drain(grabber, delay=0.0)


def test_close_stops_the_thread_and_closes_the_source() -> None:
    source = CountingSource(10_000, interval=0.001)
    grabber = LatestFrameGrabber(source)
    grabber.read()
    grabber.close()
    assert source.closed.is_set()
    assert not grabber._thread.is_alive()
