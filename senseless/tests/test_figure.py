"""Stick-figure geometry for the Speech-mode signing figure (senseless/ui/figure.py)."""

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.ui.figure import figure_geometry, frame_index, rest_frame

BOX = (10.0, 20.0, 200.0, 300.0)  # x, y, width, height
EXTENT = (-2.0, 2.0, -2.0, 2.0)


def _frame(with_hands: bool) -> np.ndarray:
    v = np.zeros(ls.FEATURE_DIM, dtype=np.float32)
    pose = np.array(
        [
            [0, -0.8, 0],
            [-0.5, 0, 0],
            [0.5, 0, 0],
            [-0.7, 0.8, 0],
            [0.7, 0.8, 0],
            [-0.8, 1.5, 0],
            [0.8, 1.5, 0],
            [-0.3, 2.0, 0],
            [0.3, 2.0, 0],
        ],
        dtype=np.float32,
    )
    v[ls.POSE_START : ls.POSE_END] = pose.reshape(-1)
    if with_hands:
        v[ls.LEFT_HAND_START : ls.LEFT_HAND_END] = np.tile([-0.8, 1.5, 0], 21) + 0.05
    return v


def _inside(x: float, y: float) -> bool:
    bx, by, bw, bh = BOX
    return bx - 1e-6 <= x <= bx + bw + 1e-6 and by - 1e-6 <= y <= by + bh + 1e-6


def test_an_empty_frame_draws_nothing() -> None:
    assert figure_geometry(np.zeros(ls.FEATURE_DIM, np.float32), BOX, EXTENT) == ([], [])


def test_body_only_frame_has_body_lines_a_head_and_no_hand_lines() -> None:
    segments, dots = figure_geometry(_frame(with_hands=False), BOX, EXTENT)
    assert segments and all(s.kind == "body" for s in segments)
    assert [d.kind for d in dots].count("head") == 1


def test_a_present_hand_adds_its_lines() -> None:
    segments, _ = figure_geometry(_frame(with_hands=True), BOX, EXTENT)
    assert any(s.kind == "left" for s in segments)
    assert not any(s.kind == "right" for s in segments)


def test_the_figure_fits_inside_the_box_and_the_origin_is_centred() -> None:
    segments, dots = figure_geometry(_frame(with_hands=True), BOX, EXTENT)
    assert all(_inside(s.x0, s.y0) and _inside(s.x1, s.y1) for s in segments)
    head = next(d for d in dots if d.kind == "head")
    bx, by, bw, bh = BOX
    assert np.isclose(head.x, bx + bw / 2)  # nose at x' = 0 -> horizontal centre


def test_frame_index_walks_the_take_then_finishes() -> None:
    assert frame_index(0.0, 45, 30.0) == 0
    assert frame_index(44 / 30.0, 45, 30.0) == 44
    assert frame_index(45 / 30.0, 45, 30.0) is None


def test_rest_frame_is_a_body_without_hands() -> None:
    lib = {"HELLO": np.stack([_frame(with_hands=True)] * 45)}
    rest = rest_frame(lib)
    assert np.all(rest[ls.LEFT_HAND_START : ls.RIGHT_HAND_END] == 0.0)
    assert np.any(rest[ls.POSE_START : ls.POSE_END] != 0.0)
    assert rest_frame({}) is None


def test_rest_frame_prefers_our_own_takes_over_longer_ms_asl_ones() -> None:
    def take(n: int, value: float) -> np.ndarray:
        frames = np.zeros((n, ls.FEATURE_DIM), np.float32)
        frames[:, ls.POSE_START : ls.POSE_END] = value
        return frames

    lib = {"AFRAID": take(61, 0.9), "HELLO": take(45, 0.3)}
    assert np.allclose(rest_frame(lib)[ls.POSE_START : ls.POSE_END], 0.3)
    # Without a 45-frame take it falls back to the first word; poseless takes never qualify.
    assert np.allclose(rest_frame({"B": take(61, 0.5), "A": take(61, 0.9)})[ls.POSE_START], 0.9)
    assert np.allclose(rest_frame({"A": take(45, 0.0), "B": take(61, 0.5)})[ls.POSE_START], 0.5)
    assert rest_frame({"A": take(45, 0.0)}) is None
