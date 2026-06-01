"""TDD specs for the pure landmark-processing path in sign/landmarks.py.

Exercises ONLY the framework-agnostic numpy function (RawLandmarks -> 153-vector):
output shape/dtype, translation- and scale-normalization invariants, and the
missing-hand zero-fill policy. No MediaPipe here -- the backend adapters are
integration code (benchmarked on the Pi), not unit-tested.
"""

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.sign.landmarks import RawLandmarks, frame_landmarks_to_vector


def _make_pose(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pose = rng.random((33, 3)).astype(np.float32)
    # Pin the shoulders so the body frame is well-defined: midpoint x=0.5, width=0.2.
    pose[ls.LEFT_SHOULDER_INDEX] = (0.40, 0.50, 0.0)
    pose[ls.RIGHT_SHOULDER_INDEX] = (0.60, 0.50, 0.0)
    return pose


def _make_hand(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((ls.NUM_HAND_LANDMARKS, ls.COORDS_PER_LANDMARK)).astype(np.float32)


def _full_raw() -> RawLandmarks:
    return RawLandmarks(left_hand=_make_hand(1), right_hand=_make_hand(2), pose=_make_pose())


def test_output_is_flat_feature_dim_float32() -> None:
    vec = frame_landmarks_to_vector(_full_raw())
    assert vec.shape == (ls.FEATURE_DIM,)
    assert vec.dtype == np.float32


def test_translation_invariance() -> None:
    raw = _full_raw()
    shift = np.array([0.13, -0.07, 0.05], dtype=np.float32)
    shifted = RawLandmarks(
        left_hand=raw.left_hand + shift,
        right_hand=raw.right_hand + shift,
        pose=raw.pose + shift,
    )
    assert np.allclose(
        frame_landmarks_to_vector(raw), frame_landmarks_to_vector(shifted), atol=1e-5
    )


def test_scale_invariance() -> None:
    raw = _full_raw()
    factor = 2.5
    scaled = RawLandmarks(
        left_hand=raw.left_hand * factor,
        right_hand=raw.right_hand * factor,
        pose=raw.pose * factor,
    )
    assert np.allclose(frame_landmarks_to_vector(raw), frame_landmarks_to_vector(scaled), atol=1e-5)


def test_missing_right_hand_zeroes_only_its_block() -> None:
    raw = _full_raw()
    full = frame_landmarks_to_vector(raw)
    one_handed = frame_landmarks_to_vector(
        RawLandmarks(left_hand=raw.left_hand, right_hand=None, pose=raw.pose)
    )
    assert np.all(one_handed[ls.RIGHT_HAND_START : ls.RIGHT_HAND_END] == 0.0)
    # Left-hand and pose blocks are unaffected by the missing right hand.
    left = slice(ls.LEFT_HAND_START, ls.LEFT_HAND_END)
    pose = slice(ls.POSE_START, ls.POSE_END)
    assert np.array_equal(one_handed[left], full[left])
    assert np.array_equal(one_handed[pose], full[pose])


def test_missing_left_hand_zeroes_only_its_block() -> None:
    raw = _full_raw()
    one_handed = frame_landmarks_to_vector(
        RawLandmarks(left_hand=None, right_hand=raw.right_hand, pose=raw.pose)
    )
    assert np.all(one_handed[ls.LEFT_HAND_START : ls.LEFT_HAND_END] == 0.0)
    assert not np.all(one_handed[ls.RIGHT_HAND_START : ls.RIGHT_HAND_END] == 0.0)


def test_no_pose_returns_all_zeros() -> None:
    raw = RawLandmarks(left_hand=_make_hand(1), right_hand=_make_hand(2), pose=None)
    vec = frame_landmarks_to_vector(raw)
    assert vec.shape == (ls.FEATURE_DIM,)
    assert np.all(vec == 0.0)


def test_degenerate_shoulder_width_returns_zeros() -> None:
    pose = _make_pose()
    pose[ls.RIGHT_SHOULDER_INDEX] = pose[ls.LEFT_SHOULDER_INDEX]  # zero width -> no scale
    raw = RawLandmarks(left_hand=_make_hand(1), right_hand=_make_hand(2), pose=pose)
    assert np.all(frame_landmarks_to_vector(raw) == 0.0)


def test_pose_block_follows_schema_index_order_and_normalization() -> None:
    raw = _full_raw()
    pose_block = frame_landmarks_to_vector(raw)[ls.POSE_START : ls.POSE_END]
    pose_block = pose_block.reshape(ls.NUM_POSE_LANDMARKS, ls.COORDS_PER_LANDMARK)
    left_sh = raw.pose[ls.LEFT_SHOULDER_INDEX]
    right_sh = raw.pose[ls.RIGHT_SHOULDER_INDEX]
    origin = (left_sh + right_sh) / 2.0
    scale = np.linalg.norm((left_sh - right_sh)[:2])
    for row, mp_index in enumerate(ls.POSE_LANDMARK_INDICES):
        expected = (raw.pose[mp_index] - origin) / scale
        assert np.allclose(pose_block[row], expected, atol=1e-5)
