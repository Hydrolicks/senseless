"""TDD specs for onset-triggered sign segmentation (sign/segmenter.py).

Training windows start the moment a hand enters the frame and cover one sign.
The live "onset" mode reproduces that: rest with hands out of view, a hand
appears -> capture one span from that first frame -> emit exactly one window ->
wait until the hands have been gone for ``release_s`` before arming again.
"""

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.sign.segmenter import OnsetSegmenter

F = ls.FEATURE_DIM
POSE = slice(ls.POSE_START, ls.POSE_END)
DT = 1 / 10  # a 10 FPS camera


def _vec(value: float) -> np.ndarray:
    v = np.zeros(F, dtype=np.float32)
    v[POSE] = value
    v[: ls.HAND_DIM] = 1.0
    return v


def _seg() -> OnsetSegmenter:
    return OnsetSegmenter(length=45, span_s=1.0, release_s=0.3, confirm_frames=2)


def _feed(seg: OnsetSegmenter, t0: float, t1: float, hands: bool) -> list[np.ndarray]:
    """Feed frames at 10 FPS over [t0, t1); return the windows emitted."""
    out = []
    for t in np.arange(t0, t1 - 1e-9, DT):
        w = seg.update(float(t), _vec(float(t)), hands)
        if w is not None:
            out.append(w)
    return out


def test_no_hands_never_emits() -> None:
    seg = _seg()
    assert _feed(seg, 0.0, 5.0, hands=False) == []
    assert seg.state == "idle"


def test_one_sign_emits_exactly_one_window_from_the_onset() -> None:
    seg = _seg()
    _feed(seg, 0.0, 1.0, hands=False)
    got = _feed(seg, 1.0, 4.0, hands=True)  # hands stay up long after the sign
    assert len(got) == 1
    w = got[0]
    assert w.shape == (45, F)
    assert np.isclose(w[0, POSE.start], 1.0, atol=1e-4)  # starts at the onset frame (t=1.0)
    assert np.isclose(w[-1, POSE.start], 2.0, atol=1e-4)  # ends one span later
    assert seg.state == "holding"


def test_it_rearms_only_after_the_hands_have_been_gone_for_release_s() -> None:
    seg = _seg()
    assert len(_feed(seg, 0.0, 2.5, hands=True)) == 1
    _feed(seg, 2.5, 2.7, hands=False)  # gone only 0.2 s < release_s
    assert _feed(seg, 2.7, 4.0, hands=True) == []  # still holding: no second word
    _feed(seg, 4.0, 4.5, hands=False)  # gone 0.5 s -> re-armed
    assert seg.state == "idle"
    assert len(_feed(seg, 4.5, 6.0, hands=True)) == 1  # the next sign


def test_a_single_frame_false_detection_does_not_start_a_capture() -> None:
    seg = _seg()
    seg.update(0.0, _vec(0.0), True)  # one-frame blip
    _feed(seg, 0.1, 3.0, hands=False)
    assert seg.state == "idle"


def test_a_short_dropout_during_the_sign_does_not_abort_it() -> None:
    seg = _seg()
    _feed(seg, 0.0, 0.5, hands=True)
    _feed(seg, 0.5, 0.7, hands=False)  # hand lost for two frames mid-sign
    got = _feed(seg, 0.7, 1.5, hands=True)
    assert len(got) == 1


def test_progress_reports_the_capture_fraction() -> None:
    seg = _seg()
    assert seg.progress(0.0) == 0.0
    _feed(seg, 0.0, 0.6, hands=True)  # onset at 0.0, confirmed at 0.1
    assert seg.state == "capturing"
    assert np.isclose(seg.progress(0.5), 0.5)
