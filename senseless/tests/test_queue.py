"""TDD specs for the bounded drop-oldest queue (common/queue.py).

Tests inject a thread ``queue.Queue`` as the backend so the drop-oldest policy is
deterministic and in-process; the production default is ``multiprocessing.Queue``,
which shares the same put_nowait/get_nowait/Full/Empty contract.
"""

import queue

import pytest

from senseless.common.queue import DropOldestQueue


def _q(maxsize: int = 3) -> DropOldestQueue:
    return DropOldestQueue(maxsize=maxsize, backend=queue.Queue)


def test_put_below_capacity_does_not_drop() -> None:
    q = _q(3)
    assert q.put("a") is None
    assert q.put("b") is None
    assert q.qsize() == 2


def test_put_when_full_drops_oldest_and_returns_it() -> None:
    q = _q(3)
    for item in ("a", "b", "c"):
        assert q.put(item) is None
    assert q.full()
    dropped = q.put("d")
    assert dropped == "a"
    assert q.qsize() == 3  # stays bounded


def test_drop_oldest_preserves_fifo_order() -> None:
    q = _q(3)
    for item in (1, 2, 3):
        q.put(item)
    q.put(4)  # evicts 1
    assert [q.get_nowait() for _ in range(3)] == [2, 3, 4]


def test_maxsize_one_keeps_only_latest() -> None:
    q = _q(1)
    q.put("old")
    assert q.put("new") == "old"
    assert q.get_nowait() == "new"
    assert q.empty()


def test_get_nowait_on_empty_raises() -> None:
    q = _q(3)
    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_never_blocks_under_sustained_overflow() -> None:
    q = _q(2)
    for i in range(100):
        q.put(i)  # must never block or raise
    assert q.qsize() == 2
    assert [q.get_nowait() for _ in range(2)] == [98, 99]


def test_cancel_join_thread_is_forwarded_to_a_backend_that_has_it() -> None:
    calls = []

    class Backend(queue.Queue):
        def cancel_join_thread(self):
            calls.append("cancel")

    q = DropOldestQueue(maxsize=2, backend=Backend)
    q.cancel_join_thread()
    assert calls == ["cancel"]


def test_cancel_join_thread_is_a_no_op_for_a_thread_queue() -> None:
    _q(2).cancel_join_thread()  # queue.Queue has no feeder thread; must not raise
