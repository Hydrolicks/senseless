"""TakeMachine: the recorder's take logic, driven by synthetic frames (no camera)."""

from pathlib import Path

import numpy as np

from senseless.collect.recorder import TakeMachine, undo_last
from senseless.common import landmark_schema as ls

SPAN = 1.0
DT = 0.1


def _feed(machine: TakeMachine, t0: float, n: int, hands: bool) -> tuple[list, float]:
    """Feed n frames DT apart from t0 (each vector filled with t + 1); return takes, next t."""
    takes = []
    t = t0
    for _ in range(n):
        out = machine.step(t, np.full(ls.FEATURE_DIM, t + 1.0, dtype=np.float32), hands)
        if out is not None:
            takes.append(out)
        t = round(t + DT, 6)
    return takes, t


def test_manual_take_starts_on_hand_after_arm_and_returns_to_idle() -> None:
    m = TakeMachine(span_s=SPAN, auto=False, clear_s=0.5, length=45)
    takes, t = _feed(m, 0.0, 5, hands=True)
    assert takes == [] and m.state == "idle"  # never records before arm()
    m.toggle_pause()
    assert m.state == "idle"  # pause is an auto-mode control
    m.arm()
    _, t = _feed(m, t, 3, hands=False)
    assert m.state == "armed"
    takes, t = _feed(m, t, 12, hands=True)
    assert len(takes) == 1
    assert takes[0].shape == (45, ls.FEATURE_DIM) and takes[0].dtype == np.float32
    assert m.state == "idle"
    assert m.frames >= 11


def test_take_covers_onset_to_onset_plus_span() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    _, t = _feed(m, 0.0, 3, hands=False)  # armed, no hand yet
    takes, _ = _feed(m, t, 12, hands=True)  # onset at t = 0.3
    assert m.onset == 0.3
    take = takes[0]
    np.testing.assert_allclose(take[0], 1.3, atol=1e-4)  # vector at the onset (t + 1)
    np.testing.assert_allclose(take[-1], 2.3, atol=1e-4)  # vector at onset + span


def test_auto_rearms_only_after_clear_time_without_hands() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    assert m.state == "armed"  # auto mode starts armed
    takes, t = _feed(m, 0.0, 12, hands=True)
    assert len(takes) == 1 and m.state == "waiting"
    _, t = _feed(m, t, 5, hands=True)  # hands still up: keep waiting
    assert m.state == "waiting"
    _, t = _feed(m, t, 5, hands=False)  # 0.4 s clear: not yet
    assert m.state == "waiting"
    _, t = _feed(m, t, 2, hands=False)  # 0.6 s clear
    assert m.state == "armed"
    takes, _ = _feed(m, t, 12, hands=True)
    assert len(takes) == 1  # the next take


def test_auto_short_dropout_does_not_rearm() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    _, t = _feed(m, 0.0, 12, hands=True)
    _, t = _feed(m, t, 4, hands=False)  # 0.3 s without hands (a tracker dropout)
    _, t = _feed(m, t, 1, hands=True)  # hand back: the clock restarts
    _, t = _feed(m, t, 4, hands=False)
    assert m.state == "waiting"


def test_pause_blocks_arming_and_resume_waits_for_clear() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    m.toggle_pause()
    assert m.state == "paused"
    takes, t = _feed(m, 0.0, 20, hands=True)
    assert takes == [] and m.state == "paused"
    m.toggle_pause()
    assert m.state == "waiting"  # hands already up must not start a take
    takes, _ = _feed(m, t, 5, hands=True)
    assert takes == [] and m.state == "waiting"


def test_pause_mid_recording_discards_the_take() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    _, t = _feed(m, 0.0, 5, hands=True)
    assert m.state == "recording"
    m.toggle_pause()
    m.toggle_pause()
    takes, _ = _feed(m, t, 20, hands=True)
    assert takes == []


def test_undo_last_deletes_newest_session_file(tmp_path: Path) -> None:
    a, b = tmp_path / "0000.npy", tmp_path / "0001.npy"
    for p in (a, b):
        np.save(p, np.zeros(3))
    saved = [a, b]
    assert undo_last(saved) == b
    assert not b.exists() and a.exists() and saved == [a]
    assert undo_last(saved) == a
    assert not a.exists()
    assert undo_last(saved) is None
