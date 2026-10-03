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


def _armed_auto(clear_s: float = 0.5) -> tuple[TakeMachine, float]:
    """An auto machine whose hands were hidden long enough to arm it; return it and next t."""
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=clear_s, length=45)
    _, t = _feed(m, 0.0, 7, hands=False)
    assert m.state == "armed"
    return m, t


def test_take_covers_onset_to_onset_plus_span() -> None:
    m, t = _armed_auto()
    onset = t
    takes, _ = _feed(m, t, 12, hands=True)
    assert m.onset == onset  # the first hand frame, not the confirming one
    take = takes[0]
    np.testing.assert_allclose(take[0], onset + 1.0, atol=1e-4)  # vector at the onset (t + 1)
    np.testing.assert_allclose(take[-1], onset + SPAN + 1.0, atol=1e-4)  # vector at onset + span


def test_auto_starts_waiting_so_hands_in_view_do_not_record() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    assert m.state == "waiting"
    takes, _ = _feed(m, 0.0, 20, hands=True)
    assert takes == [] and m.state == "waiting"


def test_auto_single_frame_blip_does_not_start_a_take() -> None:
    m, t = _armed_auto()
    _, t = _feed(m, t, 1, hands=True)
    assert m.state == "pending"
    _, t = _feed(m, t, 1, hands=False)
    assert m.state == "armed"
    takes, _ = _feed(m, t, 20, hands=False)
    assert takes == [] and m.state == "armed"


def test_auto_two_hand_frames_confirm_onset_at_first_frame() -> None:
    m, t = _armed_auto()
    first = t
    _, t = _feed(m, t, 1, hands=True)
    assert m.state == "pending"
    takes, _ = _feed(m, t, 12, hands=True)
    assert len(takes) == 1
    assert m.onset == first


def test_manual_mode_records_on_first_hand_frame_without_pending() -> None:
    m = TakeMachine(span_s=SPAN, auto=False, clear_s=0.5, length=45)
    m.arm()
    _feed(m, 0.0, 1, hands=True)
    assert m.state == "recording"


def test_auto_rearms_only_after_clear_time_without_hands() -> None:
    m, t = _armed_auto()
    takes, t = _feed(m, t, 12, hands=True)
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
    m, t = _armed_auto()
    _, t = _feed(m, t, 12, hands=True)
    _, t = _feed(m, t, 4, hands=False)  # 0.3 s without hands (a tracker dropout)
    _, t = _feed(m, t, 1, hands=True)  # hand back: the clock restarts
    _, t = _feed(m, t, 4, hands=False)
    assert m.state == "waiting"


def test_pause_blocks_arming_and_resume_waits_for_clear() -> None:
    m, t = _armed_auto()
    m.toggle_pause()
    assert m.state == "paused"
    takes, t = _feed(m, t, 20, hands=True)
    assert takes == [] and m.state == "paused"
    m.toggle_pause()
    assert m.state == "waiting"  # hands already up must not start a take
    takes, _ = _feed(m, t, 5, hands=True)
    assert takes == [] and m.state == "waiting"


def test_pause_mid_recording_discards_the_take() -> None:
    m, t = _armed_auto()
    _, t = _feed(m, t, 5, hands=True)
    assert m.state == "recording"
    m.toggle_pause()
    m.toggle_pause()
    assert m.state == "waiting"
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
