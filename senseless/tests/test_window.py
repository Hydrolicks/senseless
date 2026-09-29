"""TDD specs for time-based windowing (sign/window.py).

The classifier was trained on 45 consecutive frames at ~30 FPS. On the Pi,
perception runs at ~10 FPS, so "the last 45 frames" would be 4.5 s of motion.
``resample_window`` turns timestamped frames into the model's 45 evenly spaced
steps over the same span, and ``TimeWindow`` keeps a rolling buffer of them.
Blocks (left hand, right hand, pose) are interpolated only where present in both
neighbouring frames, so a hand is never blended with a zero "absent" block.
"""

import numpy as np
import pytest

from senseless.common import landmark_schema as ls
from senseless.sign.window import TimeWindow, resample_window

F = ls.FEATURE_DIM
LEFT = slice(ls.LEFT_HAND_START, ls.LEFT_HAND_END)
RIGHT = slice(ls.RIGHT_HAND_START, ls.RIGHT_HAND_END)
POSE = slice(ls.POSE_START, ls.POSE_END)


def _vec(left: float | None = 1.0, right: float | None = 2.0, pose: float = 3.0) -> np.ndarray:
    v = np.zeros(F, dtype=np.float32)
    if left is not None:
        v[LEFT] = left
    if right is not None:
        v[RIGHT] = right
    v[POSE] = pose
    return v


def test_resampling_at_the_original_frame_times_is_identity() -> None:
    times = np.arange(45) / 30.0
    vecs = np.random.default_rng(0).random((45, F)).astype(np.float32)
    out = resample_window(times, vecs, end_time=times[-1], length=45, span_s=44 / 30.0)
    assert out.shape == (45, F)
    assert out.dtype == np.float32
    assert np.allclose(out, vecs, atol=1e-5)


def test_blocks_present_in_both_neighbours_are_interpolated_linearly() -> None:
    times = np.array([0.0, 1.0])
    vecs = np.stack([_vec(1.0, 10.0, 4.0), _vec(2.0, 20.0, 8.0)])  # (all-zero = absent)
    out = resample_window(times, vecs, end_time=1.0, length=5, span_s=1.0)  # 0, .25, .5, .75, 1
    assert np.allclose(out[2, LEFT], 1.5)
    assert np.allclose(out[2, RIGHT], 15.0)
    assert np.allclose(out[1, POSE], 5.0)


def test_block_missing_in_a_neighbour_takes_the_nearest_frame() -> None:
    times = np.array([0.0, 1.0])
    vecs = np.stack([_vec(left=5.0), _vec(left=None)])  # left hand disappears
    out = resample_window(times, vecs, end_time=1.0, length=11, span_s=1.0)  # steps of 0.1
    assert np.allclose(out[4, LEFT], 5.0)  # t=0.4 -> nearest is t=0, hand present
    assert np.all(out[6, LEFT] == 0.0)  # t=0.6 -> nearest is t=1, hand absent
    assert np.allclose(out[6, RIGHT], 2.0)  # right hand present in both: still interpolated


def test_samples_outside_the_frames_hold_the_edge_values() -> None:
    times = np.array([1.0, 2.0])
    vecs = np.stack([_vec(pose=1.0), _vec(pose=9.0)])
    out = resample_window(times, vecs, end_time=3.0, length=5, span_s=3.0)  # 0 .. 3
    assert np.allclose(out[0, POSE], 1.0)  # before the first frame
    assert np.allclose(out[-1, POSE], 9.0)  # after the last frame


def test_low_fps_capture_resamples_close_to_the_30fps_version() -> None:
    # A hand moving smoothly, seen at 30 FPS and at 10 FPS over the same 1.5 s.
    def motion(t: np.ndarray) -> np.ndarray:
        v = np.zeros((len(t), F), dtype=np.float32)
        v[:, LEFT] = 2.0 + np.sin(2.0 * t)[:, None]  # offset: an all-zero block means "absent"
        v[:, POSE] = 1.0
        return v

    t30 = np.arange(0.0, 1.5 + 1e-9, 1 / 30)
    t10 = np.arange(0.0, 1.5 + 1e-9, 1 / 10)
    kw = dict(end_time=1.5, length=45, span_s=44 / 30.0)
    fast = resample_window(t30, motion(t30), **kw)
    slow = resample_window(t10, motion(t10), **kw)
    assert np.max(np.abs(fast - slow)) < 0.01


def test_time_window_is_ready_only_once_it_covers_the_span() -> None:
    win = TimeWindow(length=45, span_s=1.0)
    for i in range(10):  # 0.0 .. 0.9 s at 10 FPS: not yet a full second
        win.append(i / 10, _vec())
        assert not win.ready()
    win.append(1.0, _vec())
    assert win.ready()
    assert win.sample().shape == (45, F)


def test_time_window_sample_uses_the_newest_span() -> None:
    win = TimeWindow(length=3, span_s=1.0)
    for i in range(31):  # 0 .. 3 s at 10 FPS, pose value = time
        win.append(i / 10, _vec(pose=i / 10))
    out = win.sample()  # samples at t = 2.0, 2.5, 3.0
    assert np.allclose(out[:, POSE.start], [2.0, 2.5, 3.0], atol=1e-5)


def test_time_window_buffer_stays_bounded() -> None:
    win = TimeWindow(length=45, span_s=1.5)
    for i in range(3000):
        win.append(i / 30, _vec())
    assert len(win) <= int(1.5 * 30) + 2


def test_time_window_rejects_time_going_backwards() -> None:
    win = TimeWindow(length=45, span_s=1.5)
    win.append(1.0, _vec())
    with pytest.raises(ValueError, match="increase"):
        win.append(0.5, _vec())
