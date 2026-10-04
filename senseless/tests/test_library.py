"""Sign library: the most typical take per word, for the Speech-mode figure."""

import numpy as np

from senseless.collect import dataset
from senseless.sign.library import build_library, load_library, medoid_index, save_library


def _take(value: float) -> np.ndarray:
    return np.full((45, 153), value, dtype=np.float32)


def test_medoid_is_the_most_central_take_never_an_outlier() -> None:
    takes = np.stack([_take(v) for v in (1.0, 1.1, 0.9, 1.05, 9.0)])  # 9.0 is an outlier
    idx = medoid_index(takes)
    assert takes[idx, 0, 0] != 9.0
    assert takes[idx, 0, 0] in (1.0, 1.05)  # the takes closest to all the others


def test_build_library_picks_one_take_per_word_and_skips_idle(tmp_path) -> None:
    for v in (1.0, 1.1, 5.0):
        dataset.save_window(_take(v), "HELLO", tmp_path)
    for v in (0.0, 0.1):
        dataset.save_window(_take(v), "IDLE", tmp_path)
    lib = build_library(tmp_path)
    assert set(lib) == {"HELLO"}
    assert lib["HELLO"].shape == (45, 153) and lib["HELLO"].dtype == np.float32
    assert lib["HELLO"][0, 0] in (1.0, 1.1)


def test_save_and_load_round_trip(tmp_path) -> None:
    lib = {"HELLO": _take(1.0), "YES": _take(2.0)}
    path = save_library(lib, tmp_path / "sign_library.npz")
    loaded = load_library(path)
    assert set(loaded) == {"HELLO", "YES"}
    assert np.array_equal(loaded["YES"], lib["YES"])


def test_add_extra_keeps_our_own_takes(tmp_path) -> None:
    from senseless.sign import library as lib

    ours = {"HELLO": np.zeros((45, 153), np.float32)}
    extra = {"HELLO": np.ones((61, 153), np.float32), "MILK": np.ones((61, 153), np.float32)}
    path = lib.save_library(extra, tmp_path / "extra.npz")
    merged = lib.add_extra(ours, path)
    assert sorted(merged) == ["HELLO", "MILK"] and merged["HELLO"].shape == (45, 153)
    assert lib.add_extra(ours, tmp_path / "missing.npz") is ours
