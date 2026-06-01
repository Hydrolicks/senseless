"""Smoke tests: the package imports and the landmark schema is self-consistent.

No feature behavior is exercised yet -- this just proves the skeleton, config,
and schema wire together and that the pytest harness runs.
"""

from senseless.common import config
from senseless.common import landmark_schema as ls


def test_config_singletons_present() -> None:
    assert config.SIGN.window_length == 30
    assert config.AUDIO.sample_rate_hz == 16_000


def test_feature_dim_is_internally_consistent() -> None:
    # The declared FEATURE_DIM equals the sum of the block sizes...
    assert sum(block.size for block in ls.FEATURE_BLOCKS) == ls.FEATURE_DIM == 153
    # ...and the individual pieces add up the way the docstring claims.
    assert ls.HAND_DIM == 63
    assert ls.POSE_DIM == 27
    assert 2 * ls.HAND_DIM + ls.POSE_DIM == ls.FEATURE_DIM


def test_pose_subset_excludes_face_and_lower_body() -> None:
    assert ls.NUM_POSE_LANDMARKS == len(ls.POSE_LANDMARK_INDICES)
    # Face mesh excluded: none of the eye/ear/mouth indices (1-10) appear.
    assert not (set(ls.POSE_LANDMARK_INDICES) & set(range(1, 11)))
    # Lower body excluded: nothing at or beyond the knees (25+).
    assert all(idx < 25 for idx in ls.POSE_LANDMARK_INDICES)
