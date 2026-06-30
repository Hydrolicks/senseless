"""TDD specs for the collected-sample dataset layout (collect/dataset.py).

This is the on-disk contract shared by the collector (writes) and the Colab
trainer (reads): one (window_length, FEATURE_DIM) float32 window per .npy file,
under data/<label>/. Tested against a tmp dir -- no camera involved.
"""

import numpy as np
import pytest

from senseless.collect import dataset
from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN


def _window(fill: float = 1.0) -> np.ndarray:
    return np.full((SIGN.window_length, ls.FEATURE_DIM), fill, dtype=np.float32)


def test_window_shape_matches_schema() -> None:
    assert dataset.WINDOW_SHAPE == (SIGN.window_length, ls.FEATURE_DIM) == (30, 153)


def test_save_then_count_and_load(tmp_path) -> None:
    p1 = dataset.save_window(_window(0.5), "HELLO", data_dir=tmp_path)
    p2 = dataset.save_window(_window(0.25), "HELLO", data_dir=tmp_path)
    assert p1 != p2
    assert dataset.count_samples("HELLO", data_dir=tmp_path) == 2
    loaded = dataset.load_label("HELLO", data_dir=tmp_path)
    assert loaded.shape == (2, 30, 153)
    assert loaded.dtype == np.float32


def test_next_index_zero_for_new_label_then_increments(tmp_path) -> None:
    assert dataset.next_sample_index("NEW", data_dir=tmp_path) == 0
    dataset.save_window(_window(), "NEW", data_dir=tmp_path)
    assert dataset.next_sample_index("NEW", data_dir=tmp_path) == 1


def test_save_rejects_wrong_shape(tmp_path) -> None:
    with pytest.raises(ValueError):
        dataset.save_window(np.zeros((10, 153), np.float32), "BAD", data_dir=tmp_path)


def test_count_is_zero_for_missing_label(tmp_path) -> None:
    assert dataset.count_samples("MISSING", data_dir=tmp_path) == 0


def test_list_labels(tmp_path) -> None:
    dataset.save_window(_window(), "A", data_dir=tmp_path)
    dataset.save_window(_window(), "B", data_dir=tmp_path)
    assert sorted(dataset.list_labels(data_dir=tmp_path)) == ["A", "B"]
