"""TDD specs for hand-landmark perturbation (sign/augment.py).

The Pi runs MediaPipe's lite hand model, whose finger positions differ slightly
from the full model the training data was recorded with. ``perturb_hands`` makes
training copies with small per-window hand rotations/scales around the wrist, a
fixed per-landmark offset (like another model's systematic bias) and per-frame
jitter, so the classifier stops relying on the full model's exact finger geometry.
"""

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.sign.augment import perturb_hands

F = ls.FEATURE_DIM
LEFT = slice(ls.LEFT_HAND_START, ls.LEFT_HAND_END)
RIGHT = slice(ls.RIGHT_HAND_START, ls.RIGHT_HAND_END)
POSE = slice(ls.POSE_START, ls.POSE_END)
NONE = dict(bias_sigma=0.0, jitter_sigma=0.0, max_rot_deg=0.0, max_scale=0.0)


def _window(right_present: bool = False) -> np.ndarray:
    rng = np.random.default_rng(7)
    w = np.zeros((45, F), dtype=np.float32)
    w[:, LEFT] = rng.normal(0.0, 0.3, (45, 63)) + 1.0
    if right_present:
        w[:, RIGHT] = rng.normal(0.0, 0.3, (45, 63)) - 1.0
    w[:, POSE] = rng.normal(0.0, 0.5, (45, 27))
    return w


def _hand(w: np.ndarray, block: slice) -> np.ndarray:
    return w[:, block].reshape(len(w), 21, 3)


def test_zero_strength_is_identity() -> None:
    w = _window(right_present=True)
    assert np.allclose(perturb_hands(w, np.random.default_rng(0), **NONE), w, atol=1e-6)


def test_absent_hands_stay_absent_and_pose_is_untouched() -> None:
    w = _window(right_present=False)
    out = perturb_hands(w, np.random.default_rng(0))
    assert np.all(out[:, RIGHT] == 0.0)
    assert np.array_equal(out[:, POSE], w[:, POSE])
    assert not np.allclose(out[:, LEFT], w[:, LEFT])  # the present hand did change
    assert out.dtype == np.float32


def test_frames_where_a_hand_is_missing_stay_zero() -> None:
    w = _window()
    w[10:20, LEFT] = 0.0  # detection dropout mid-sign
    out = perturb_hands(w, np.random.default_rng(0))
    assert np.all(out[10:20, LEFT] == 0.0)


def test_rotation_and_scale_pivot_on_the_wrist() -> None:
    w = _window()
    out = perturb_hands(
        w,
        np.random.default_rng(3),
        bias_sigma=0.0,
        jitter_sigma=0.0,
        max_rot_deg=15.0,
        max_scale=0.2,
    )
    before, after = _hand(w, LEFT), _hand(out, LEFT)
    assert np.allclose(after[:, 0], before[:, 0], atol=1e-6)  # wrist fixed
    # one scale per window: every landmark's xy distance to the wrist scales alike
    d0 = np.linalg.norm(before[:, 1:, :2] - before[:, :1, :2], axis=-1)
    d1 = np.linalg.norm(after[:, 1:, :2] - after[:, :1, :2], axis=-1)
    ratio = d1 / d0
    assert np.allclose(ratio, ratio.flat[0], atol=1e-4)
    assert 0.8 - 1e-6 <= ratio.flat[0] <= 1.2 + 1e-6


def test_perturbation_is_small_and_reproducible() -> None:
    w = _window()
    a = perturb_hands(w, np.random.default_rng(5))
    b = perturb_hands(w, np.random.default_rng(5))
    assert np.array_equal(a, b)
    # defaults stay within a few percent of a ~0.55-shoulder-width hand
    assert np.median(np.abs(a[:, LEFT] - w[:, LEFT])) < 0.05


def test_drop_hand_run_blanks_a_short_run_of_a_moving_hand_only() -> None:
    from senseless.sign.augment import drop_hand_run

    w = _window()  # left hand present and moving, right hand absent
    hit = 0
    for seed in range(20):
        out = drop_hand_run(w, np.random.default_rng(seed), prob=1.0)
        blank = ~np.any(out[:, LEFT] != 0, axis=1)
        if blank.any():
            hit += 1
            idx = np.flatnonzero(blank)
            assert 4 <= len(idx) <= 13 and np.all(np.diff(idx) == 1)  # one short run
        assert not np.any(out[:, RIGHT])  # absent hand stays absent
        np.testing.assert_array_equal(out[:, POSE], w[:, POSE])
        np.testing.assert_array_equal(out[~blank], w[~blank])
    assert hit == 20


def test_drop_hand_run_leaves_still_or_absent_hands_alone() -> None:
    from senseless.sign.augment import drop_hand_run

    w = np.zeros((45, F), dtype=np.float32)
    w[:, LEFT] = 0.5  # present but never moves
    out = drop_hand_run(w, np.random.default_rng(0), prob=1.0)
    np.testing.assert_array_equal(out, w)
    out = drop_hand_run(w, np.random.default_rng(0), prob=0.0)
    np.testing.assert_array_equal(out, w)
