"""TDD specs for the pure, MediaPipe-free parts of backend selection.

``slot_hands`` maps MediaPipe hand detections (landmarks + "Left"/"Right" label)
onto the signer's left/right slots; every backend shares it, so the feature
vector's hand blocks mean the same thing whichever backend produced them.
``create_backend`` must reject unknown names before touching MediaPipe.
"""

import numpy as np
import pytest

from senseless.sign import landmarks
from senseless.sign.landmarks import slot_hands


def _hand(value: float) -> np.ndarray:
    return np.full((21, 3), value, dtype=np.float32)


def test_labels_map_to_matching_slots() -> None:
    left, right = slot_hands([(_hand(1), "Right"), (_hand(2), "Left")], mirror=False)
    assert np.array_equal(left, _hand(2))
    assert np.array_equal(right, _hand(1))


def test_mirror_swaps_the_slots() -> None:
    left, right = slot_hands([(_hand(1), "Left")], mirror=True)
    assert left is None
    assert np.array_equal(right, _hand(1))


def test_no_detections_gives_two_empty_slots() -> None:
    assert slot_hands([], mirror=False) == (None, None)


def test_duplicate_label_falls_back_to_the_free_slot() -> None:
    # MediaPipe occasionally labels both hands the same; keep both, don't drop one.
    left, right = slot_hands([(_hand(1), "Left"), (_hand(2), "Left")], mirror=False)
    assert np.array_equal(left, _hand(1))
    assert np.array_equal(right, _hand(2))


def test_extra_detections_beyond_two_are_ignored() -> None:
    left, right = slot_hands(
        [(_hand(1), "Left"), (_hand(2), "Right"), (_hand(3), "Right")], mirror=False
    )
    assert np.array_equal(left, _hand(1))
    assert np.array_equal(right, _hand(2))


def test_backend_names_include_lite() -> None:
    assert set(landmarks.BACKEND_NAMES) == {"tasks", "holistic", "lite"}


def test_unknown_backend_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="lite"):
        landmarks.create_backend("turbo")
