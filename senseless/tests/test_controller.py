"""ModeController: one spawned worker at a time, stop with a timeout, crash detection."""

import time

import ui_fakes as fakes

from senseless.common.events import WorkerError, WorkerReady
from senseless.ui.controller import ModeController


def _wait(predicate, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_an_idle_controller_counts_as_stopped() -> None:
    assert ModeController({}).poll_stopped()


def test_shutdown_is_safe_to_call_twice_and_on_an_idle_controller() -> None:
    ModeController({}).shutdown()
    ctl = ModeController({"fake": fakes.ready_worker})
    ctl.start("fake")
    ctl.shutdown()
    ctl.shutdown()
    assert ctl.mode is None and ctl.poll_stopped()


def test_a_worker_delivers_events_and_frames_then_stops_on_request() -> None:
    ctl = ModeController({"fake": fakes.ready_worker})
    ctl.start("fake")
    try:
        seen = []
        assert _wait(lambda: seen.extend(ctl.drain_events()) or WorkerReady("fake") in seen)
        assert _wait(lambda: ctl.latest_frame() is not None)
        ctl.request_stop()
        assert _wait(ctl.poll_stopped)
        assert ctl.mode is None
    finally:
        ctl.shutdown()


def test_a_worker_that_ignores_stop_is_terminated_after_the_timeout() -> None:
    ctl = ModeController({"fake": fakes.stubborn_worker}, switch_timeout_s=0.5)
    ctl.start("fake")
    try:
        seen = []
        assert _wait(lambda: seen.extend(ctl.drain_events()) or WorkerReady("fake") in seen)
        ctl.request_stop()
        assert _wait(ctl.poll_stopped)
    finally:
        ctl.shutdown()


def test_an_unexpected_exit_is_detected_with_its_code() -> None:
    ctl = ModeController({"fake": fakes.crashing_worker})
    ctl.start("fake")
    try:
        assert _wait(ctl.has_exited)
        assert ctl.exitcode() == 3
    finally:
        ctl.shutdown()


def test_events_drained_while_stopping_are_kept_for_the_next_drain() -> None:
    ctl = ModeController({"fake": fakes.one_shot_worker})
    ctl.start("fake")
    try:
        assert _wait(ctl.has_exited)
        ctl.request_stop()
        assert _wait(ctl.poll_stopped)
        assert ctl.drain_events() == [WorkerReady("fake"), WorkerError("boom")]
        assert ctl.drain_events() == []
    finally:
        ctl.shutdown()
