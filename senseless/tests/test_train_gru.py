"""Pure helpers of the trainer: Pi-recorded data and Pi-like augmentation (no TensorFlow)."""

import numpy as np

from senseless.collect import dataset
from senseless.common import landmark_schema as ls
from senseless.notebooks import train_gru

SHAPE = dataset.WINDOW_SHAPE


def _save(root, label: str, n: int, value: float = 1.0) -> None:
    for _ in range(n):
        dataset.save_window(np.full(SHAPE, value, np.float32), label, root)


def test_load_pi_data_maps_to_the_training_labels_and_skips_unknown_words(tmp_path) -> None:
    _save(tmp_path, "CAT", 3)
    _save(tmp_path, "HELLO", 2)
    _save(tmp_path, "GO", 4)  # not in the vocabulary
    x, y, skipped = train_gru.load_pi_data(str(tmp_path), ["CAT", "DOG", "HELLO", "IDLE"])
    assert x.shape == (5, *SHAPE) and x.dtype == np.float32
    assert sorted(y.tolist()) == [0, 0, 0, 2, 2]
    assert skipped == ["GO"]


def test_split_pi_holds_out_a_share_of_every_word_and_keeps_singletons_for_training() -> None:
    x = np.zeros((21, *SHAPE), np.float32)
    y = np.array([0] * 10 + [1] * 10 + [2])
    (x_tr, y_tr), (x_te, y_te) = train_gru.split_pi(x, y, test_frac=0.3, seed=1)
    assert sorted(np.bincount(y_te, minlength=3).tolist()) == [0, 3, 3]
    assert np.bincount(y_tr, minlength=3).tolist() == [7, 7, 1]
    (x_tr, y_tr), (x_te, y_te) = train_gru.split_pi(x, y, test_frac=0.0, seed=1)
    assert len(y_tr) == 21 and len(y_te) == 0


def test_pi_like_augmentation_adds_copies_with_dropped_hands() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.3, (4, *SHAPE)).astype(np.float32)
    x[:, :, ls.POSE_START : ls.POSE_END] = 0.2
    y = np.arange(4)
    xa, ya = train_gru.augment_low_fps(x, y, 3, 6.0, 1, hand_perturb=True, pi_like=True)
    assert xa.shape == (16, *SHAPE) and ya.tolist() == list(range(4)) * 4
    np.testing.assert_array_equal(xa[:4], x)  # originals first, unchanged
