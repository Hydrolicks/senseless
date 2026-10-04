"""MS-ASL takes for the signing figure: cleaning and ranking (synthetic data, no tracker)."""

import json

import numpy as np
import pytest

from senseless.collect import msasl
from senseless.common import landmark_schema as ls
from senseless.sign import extra_library as xl

L = slice(ls.LEFT_HAND_START, ls.LEFT_HAND_END)
R = slice(ls.RIGHT_HAND_START, ls.RIGHT_HAND_END)
P = slice(ls.POSE_START, ls.POSE_END)


def _frames(n: int) -> np.ndarray:
    v = np.zeros((n, ls.FEATURE_DIM), np.float32)
    v[:, P] = 0.25
    return v


def test_hand_coverage_counts_frames_with_any_hand() -> None:
    v = _frames(10)
    v[:4, L] = 1.0
    v[3:6, R] = 1.0
    assert xl.hand_coverage(v) == 0.6
    assert xl.hand_coverage(_frames(0)) == 0.0


def test_fill_gaps_interpolates_short_interior_gaps_only() -> None:
    t = np.arange(11) / 10.0
    v = _frames(11)
    v[0:3, R] = [[1.0], [1.0], [1.0]]
    v[5:7, R] = [[4.0], [4.0]]  # gap 0.2 -> 0.5 s (0.3 s): filled
    v[0, L] = 2.0
    v[6, L] = 2.0  # gap 0.0 -> 0.6 s: too long, kept
    out = xl.fill_gaps(t, v)
    np.testing.assert_allclose(out[3, R], 2.0)
    np.testing.assert_allclose(out[4, R], 3.0)
    assert not np.any(out[7:, R])  # trailing gap stays empty
    assert not np.any(out[1:6, L])


def test_mirror_flips_x_and_swaps_sides_and_is_its_own_inverse() -> None:
    v = _frames(2)
    v[:, L] = np.tile([0.5, 0.7, 0.1], 21)
    shoulders = ls.POSE_START + 1 * 3, ls.POSE_START + 2 * 3  # left, right shoulder
    v[:, shoulders[0] : shoulders[0] + 3] = [-0.5, 0.0, 0.0]
    v[:, shoulders[1] : shoulders[1] + 3] = [0.5, 0.0, 0.0]
    m = xl.mirror(v)
    assert not np.any(m[:, L])
    np.testing.assert_allclose(m[0, R][:3], [-0.5, 0.7, 0.1])
    np.testing.assert_allclose(m[0, shoulders[0] : shoulders[0] + 3], [-0.5, 0.0, 0.0])
    np.testing.assert_allclose(m[0, shoulders[1] : shoulders[1] + 3], [0.5, 0.0, 0.0])
    np.testing.assert_array_equal(xl.mirror(m), v)


def test_natural_resample_keeps_real_speed_and_caps_at_3_s() -> None:
    t = np.arange(0, 4.01, 1 / 25)
    v = _frames(len(t))
    assert len(xl.natural_resample(t, v, 2.0)) == 61
    assert len(xl.natural_resample(t, v, 4.0)) == 91


def test_clean_clip_trims_mirrors_left_dominant_and_rejects_handless_clips() -> None:
    t = np.arange(-0.5, 2.5, 1 / 30)
    v = _frames(len(t))
    v[:, L] = np.tile([1.0, 0.5, 0.0], 21) * (1.0 + t[:, None])  # the moving hand
    v[:, R] = np.tile([-1.0, 0.5, 0.0], 21)  # a still hand
    take = xl.clean_clip(t, v, 2.0)
    assert take.shape == (61, ls.FEATURE_DIM)
    assert xl.motion(take, R) > xl.motion(take, L)  # now right-hand dominant
    bare = _frames(len(t))
    assert xl.clean_clip(t, bare, 2.0) is None


def test_rank_puts_the_typical_take_first_and_the_outlier_last() -> None:
    a = _frames(61)
    b = a + 0.01
    odd = a + 3.0
    order = xl.rank([odd, a, b])
    assert order[0] in (1, 2) and order[-1] == 0
    assert xl.rank([a]) == [0] and xl.rank([]) == []


def _seq(work, clip, hands: bool) -> None:
    t = np.arange(-0.5, 2.5, 1 / 30)
    v = _frames(len(t))
    if hands:
        v[:, R] = np.tile([-1.0, 0.5, 0.0], 21) * (1.0 + t[:, None])
    path = msasl.sequence_path(work, clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, times=t, vecs=v, duration=2.0)


def _clip(clip_id: str, label: str, status: str = "extracted") -> msasl.Clip:
    return msasl.Clip(clip_id, label, "train", 1, "u", 1.0, 3.0, 30.0, "0 0 1 1", status)


def test_collect_candidates_groups_ranks_and_skips_bad_clips(tmp_path) -> None:
    clips = [_clip("a", "MILK"), _clip("b", "MILK"), _clip("c", "TEA"), _clip("d", "CAT", "failed")]
    _seq(tmp_path, clips[0], True)
    _seq(tmp_path, clips[1], True)
    _seq(tmp_path, clips[2], False)  # no hands: rejected
    cands = xl.collect_candidates(clips, tmp_path)
    assert list(cands) == ["MILK"] and len(cands["MILK"]) == 2
    assert cands["MILK"][0].shape == (61, ls.FEATURE_DIM)


def test_candidates_round_trip(tmp_path) -> None:
    cands = {"MILK": [_frames(61), _frames(31)], "TEA": [_frames(91)]}
    xl.save_candidates(cands, tmp_path / "c.npz")
    back = xl.load_candidates(tmp_path / "c.npz")
    assert sorted(back) == ["MILK", "TEA"]
    assert [len(t) for t in back["MILK"]] == [61, 31]


def test_choose_defaults_to_the_first_take_and_validates() -> None:
    a, b = _frames(61), _frames(31)
    cands = {"MILK": [a, b], "TEA": [a], "BUT": [b]}
    picked = xl.choose(cands, {"MILK": 1, "BUT": "reject"})
    assert sorted(picked) == ["MILK", "TEA"] and len(picked["MILK"]) == 31
    with pytest.raises(ValueError):
        xl.choose(cands, {"SODA": 0})
    with pytest.raises(ValueError):
        xl.choose(cands, {"TEA": 1})


def test_review_page_embeds_every_word_as_valid_json() -> None:
    cands = {"MILK": [_frames(61), _frames(31)], "TEA": [_frames(91)]}
    page = xl.build_review_html(cands)
    data = json.loads(page.split("const D = ", 1)[1].split(";\n", 1)[0])
    assert [w["word"] for w in data["words"]] == ["MILK", "TEA"]
    assert len(data["words"][0]["takes"]) == 2
    assert len(data["words"][0]["takes"][0]) == 31  # 61 frames previewed at 15 FPS
    assert "Export choices" in page
