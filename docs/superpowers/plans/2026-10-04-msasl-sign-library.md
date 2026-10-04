# MS-ASL Sign Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add stick-figure takes for 151 common ASL words, built from MS-ASL clips, to the
Speech-mode sign library. A review page lets a person reject takes or pick alternates.

**Architecture:**
- `collect/msasl.py` (carried over from the experiment) gains a `select-anim` step. It
  picks up to 5 clips for each of MS-ASL's 200 most frequent glosses that are new to us.
- A new pure-numpy module, `sign/extra_library.py`, does the rest:
  - cleans each extracted clip (trim, coverage, gap filling, mirroring, natural-speed
    resampling);
  - ranks a word's clips;
  - writes a self-contained `review.html`;
  - builds `models/sign_library_extra.npz`, which `sign/library.py` merges under our
    own takes.
- The app plays takes of any length.

**Tech Stack:**
- Python 3.11, numpy, pytest;
- vanilla JS/HTML for the review page;
- the extraction environment `.venv-msasl` with mediapipe 0.10.18 for `extract`.

Spec: `docs/superpowers/specs/2026-10-04-msasl-sign-library-design.md`.

## Global Constraints

- Our own recorded takes always win. A word in `data/` keeps its own take, even if
  MS-ASL has it.
- The library format is `{WORD: (N, 153) float32}`. Our takes have N = 45. MS-ASL takes
  are 30 FPS at natural speed: `ceil(min(duration, 3.0) * 30) + 1` frames, at most 91.
- Constants, used verbatim:
  - `FPS = 30.0`, `MAX_TAKE_S = 3.0`, `MIN_HAND_COVERAGE = 0.6`, `MAX_GAP_S = 0.3`,
    `COMPARE_STEPS = 45`;
  - `ANIM_TOP = 200`, `ANIM_PER_WORD = 5`, `ANIM_WORK_DIR = Path("C:/Senseless_anim")`.
- Word keys are the gloss in upper case.
- Skip glosses that:
  - map to our labels (`gloss_to_label`);
  - are not `[a-z]+`;
  - are `hoddog`.
- Recognition does not change: classifier, training, `data/` and the deployed model are
  untouched.
- `sign/extra_library.py` must not import mediapipe or cv2. It runs in the main `.venv`.
- Run tests from the worktree root
  `C:\Users\Asaf Amrani\Desktop\EE Engineering\Fourth Year\Senseless-signlib` with
  `"../Senseless/.venv/Scripts/python" -m pytest -q -p no:cacheprovider senseless/tests`.
  ruff and black use a line length of 100.

---

### Task 1: Select the animation words and clips (`msasl select-anim`)

**Files:**
- Modify: `senseless/collect/msasl.py`
- Test: `senseless/tests/test_msasl.py`

**Interfaces:**
- Produces:
  - `ANIM_WORK_DIR`, `ANIM_TOP`, `ANIM_PER_WORD`;
  - `animation_glosses(classes: list[str], taken: Iterable[str], top: int = ANIM_TOP) -> list[str]`;
  - `select_animation_clips(entries_by_split: dict[str, list[dict]], glosses: list[str], per_word: int = ANIM_PER_WORD) -> list[Clip]`;
  - the CLI step `select-anim`.

- [ ] **Step 1: Write the failing tests** (append to `senseless/tests/test_msasl.py`)

```python
def test_animation_glosses_skip_ours_phrases_and_typos() -> None:
    classes = ["hello", "teacher", "not know", "how_many", "hoddog", "Mother ", "thanks", "milk"]
    assert msasl.animation_glosses(classes, {"hello", "thanks"}, top=7) == ["teacher", "mother"]


def _entry(gloss: str, signer: int) -> dict:
    return {
        "clean_text": gloss,
        "signer_id": signer,
        "url": "https://youtu.be/x",
        "start_time": 1.0,
        "end_time": 2.0,
        "fps": 30.0,
        "box": [0.0, 0.0, 1.0, 1.0],
    }


def test_select_animation_clips_caps_per_word_and_prefers_new_signers() -> None:
    entries = {
        "train": [_entry("milk", 1), _entry("milk", 1), _entry("milk", 2), _entry("cat", 5)],
        "val": [_entry("milk", 3)],
        "test": [_entry("milk", 1)],
    }
    clips = msasl.select_animation_clips(entries, ["milk"], per_word=3)
    assert [c.label for c in clips] == ["MILK"] * 3
    assert [c.signer for c in clips] == [1, 2, 3]
    assert [c.clip_id for c in clips] == ["train_00000", "train_00002", "val_00000"]
    more = msasl.select_animation_clips(entries, ["milk"], per_word=5)
    assert [c.clip_id for c in more][3:] == ["train_00001", "test_00000"]
```

- [ ] **Step 2: Run them and confirm they fail** (AttributeError: no `animation_glosses`)

Run: `"../Senseless/.venv/Scripts/python" -m pytest -q -p no:cacheprovider senseless/tests/test_msasl.py -k animation`

- [ ] **Step 3: Implement**

In `senseless/collect/msasl.py`:

1. Add `import re` to the standard-library imports if it is not there.
2. After `MIN_TRAIN_CLIPS = ...`, add:

```python
# Takes for the Speech-mode figure (sign/extra_library.py): MS-ASL's most frequent words
# that we did not record ourselves.
ANIM_WORK_DIR = Path("C:/Senseless_anim")
ANIM_TOP = 200  # MSASL_classes.json is ordered most frequent first
ANIM_PER_WORD = 5
_SKIP_GLOSSES = ("hoddog",)  # a typo in MS-ASL's class list
```

3. Replace the body of `select_clips` so that it builds each clip with a new helper.
   Keep the signature and behaviour:

```python
def _clip_from_entry(clip_id: str, label: str, split: str, entry: dict) -> Clip:
    return Clip(
        clip_id=clip_id,
        label=label,
        split=split,
        signer=int(entry["signer_id"]),
        url=entry["url"],
        start=float(entry["start_time"]),
        end=float(entry["end_time"]),
        fps=float(entry.get("fps", 30.0)),
        box=" ".join(f"{float(v):.4f}" for v in entry["box"]),
    )


def select_clips(entries_by_split: dict[str, list[dict]], mapping: dict[str, str]) -> list[Clip]:
    """One Clip per MS-ASL entry whose gloss maps to our vocabulary."""
    clips = []
    for split in SPLITS:
        for i, entry in enumerate(entries_by_split.get(split, [])):
            label = mapping.get(str(entry["clean_text"]).strip().lower())
            if label is not None:
                clips.append(_clip_from_entry(f"{split}_{i:05d}", label, split, entry))
    return clips


def animation_glosses(classes: list[str], taken: Iterable[str], top: int = ANIM_TOP) -> list[str]:
    """MS-ASL's most frequent glosses that are new to us: single plain words only."""
    taken = set(taken)
    glosses = (c.strip().lower() for c in classes[:top])
    return [
        g
        for g in glosses
        if g not in taken and re.fullmatch(r"[a-z]+", g) and g not in _SKIP_GLOSSES
    ]


def select_animation_clips(
    entries_by_split: dict[str, list[dict]], glosses: list[str], per_word: int = ANIM_PER_WORD
) -> list[Clip]:
    """Up to ``per_word`` clips per gloss: one per signer first, then the rest, in split order."""
    rows: dict[str, list[tuple[str, int, dict]]] = {g: [] for g in glosses}
    for split in SPLITS:
        for i, entry in enumerate(entries_by_split.get(split, [])):
            gloss = str(entry["clean_text"]).strip().lower()
            if gloss in rows:
                rows[gloss].append((split, i, entry))
    clips = []
    for gloss in glosses:
        found = rows[gloss]
        firsts, signers = set(), set()
        for k, (_, _, entry) in enumerate(found):
            if int(entry["signer_id"]) not in signers:
                signers.add(int(entry["signer_id"]))
                firsts.add(k)
        order = sorted(range(len(found)), key=lambda k: (k not in firsts, k))
        for k in order[:per_word]:
            split, i, entry = found[k]
            clips.append(_clip_from_entry(f"{split}_{i:05d}", gloss.upper(), split, entry))
    return clips
```

4. In `main()`:
   - add `"select-anim"` to the `step` choices;
   - change `--work` to `default=None`, with help `"Videos, sequences, manifest (default C:/Senseless_msasl; C:/Senseless_anim for select-anim)."`;
   - replace the `work, manifest = ...` line with:

```python
    default_work = ANIM_WORK_DIR if args.step == "select-anim" else WORK_DIR
    work = Path(args.work) if args.work else default_work
    manifest = work / "manifest.csv"
```

   - before the existing `if args.step == "select":` block, add:

```python
    if args.step == "select-anim":
        if manifest.exists() and not args.force:
            raise SystemExit(f"{manifest} already exists. Use --force to overwrite it.")
        entries, synonyms = load_msasl(Path(args.msasl))
        classes = json.loads((Path(args.msasl) / "MSASL_classes.json").read_text())
        glosses = animation_glosses(classes, gloss_to_label(vocabulary(), synonyms))
        clips = select_animation_clips(entries, glosses)
        write_manifest(clips, manifest)
        print(f"{len(clips)} clips of {len(glosses)} words -> {manifest}")
        return
```

   - update the parser description to `"MS-ASL clips: our vocabulary (select) or new words for the signing figure (select-anim)."`.

- [ ] **Step 4: Run the tests, the full suite, ruff and black.** Everything must pass.

- [ ] **Step 5: Commit**

```bash
git add senseless/collect/msasl.py senseless/tests/test_msasl.py
git commit -m "Select MS-ASL clips of new words for the signing figure (select-anim)"
```

---

### Task 2: Clean and rank MS-ASL takes (`sign/extra_library.py`, part 1)

**Files:**
- Create: `senseless/sign/extra_library.py`
- Test: `senseless/tests/test_extra_library.py`

**Interfaces:**
- Produces:
  - `hand_coverage(vecs) -> float`;
  - `fill_gaps(times, vecs, max_gap_s=MAX_GAP_S) -> np.ndarray`;
  - `motion(vecs, block: slice) -> float`;
  - `mirror(vecs) -> np.ndarray`;
  - `natural_resample(times, vecs, duration) -> np.ndarray`;
  - `clean_clip(times, vecs, duration) -> np.ndarray | None`;
  - `rank(takes: list[np.ndarray]) -> list[int]`;
  - the constants of the Global Constraints.

- [ ] **Step 1: Write the failing tests** (create `senseless/tests/test_extra_library.py`)

```python
"""MS-ASL takes for the signing figure: cleaning and ranking (synthetic data, no tracker)."""

import numpy as np

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
```

- [ ] **Step 2: Run them and confirm they fail** (ImportError).

- [ ] **Step 3: Implement** (create `senseless/sign/extra_library.py`)

```python
"""MS-ASL takes for the Speech-mode figure: clean, rank, review, add to the library.

Words we did not record get a stick-figure take from MS-ASL clips (lite-tracker
landmark sequences from ``collect.msasl``). Each clip is trimmed to the annotated
sign, dropped if the hands are missing in most frames, has short tracker gaps
interpolated, is mirrored to right-hand dominance like our own takes, and is
resampled to 30 FPS at its natural speed (capped at 3 s). Per word, the most
typical clip is the automatic pick; a review page lets a person reject words or
pick alternates. Our own recorded takes always win (``sign/library.py``).

    python -m senseless.sign.extra_library review   # C:/Senseless_anim -> review.html
    python -m senseless.sign.extra_library build [--choices choices.json]
"""

from __future__ import annotations

import math

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.sign.window import resample_window

FPS = 30.0  # playback rate of the figure (UI.figure_tick_ms ~ 33 ms)
MAX_TAKE_S = 3.0
MIN_HAND_COVERAGE = 0.6
MAX_GAP_S = 0.3
COMPARE_STEPS = 45  # takes are resampled to this many steps only to compare them

_LEFT = slice(ls.LEFT_HAND_START, ls.LEFT_HAND_END)
_RIGHT = slice(ls.RIGHT_HAND_START, ls.RIGHT_HAND_END)
_POSE = slice(ls.POSE_START, ls.POSE_END)
_BLOCKS = (_LEFT, _RIGHT, _POSE)
_C = ls.COORDS_PER_LANDMARK
_POSE_PAIRS = tuple(
    (i, ls.POSE_LANDMARK_NAMES.index("right_" + name[len("left_") :]))
    for i, name in enumerate(ls.POSE_LANDMARK_NAMES)
    if name.startswith("left_")
)


def _present(vecs: np.ndarray, block: slice) -> np.ndarray:
    return np.any(vecs[:, block] != 0.0, axis=1)


def hand_coverage(vecs: np.ndarray) -> float:
    """Share of frames with at least one hand."""
    if len(vecs) == 0:
        return 0.0
    return float(np.mean(_present(vecs, _LEFT) | _present(vecs, _RIGHT)))


def fill_gaps(times: np.ndarray, vecs: np.ndarray, max_gap_s: float = MAX_GAP_S) -> np.ndarray:
    """Linearly fill a block missing between two frames that have it, if they are <= max_gap_s apart."""
    out = np.array(vecs, dtype=np.float32, copy=True)
    for block in _BLOCKS:
        idx = np.flatnonzero(_present(out, block))
        for a, b in zip(idx[:-1], idx[1:], strict=False):
            if b - a > 1 and times[b] - times[a] <= max_gap_s + 1e-9:
                w = ((times[a + 1 : b] - times[a]) / (times[b] - times[a]))[:, None]
                out[a + 1 : b, block] = out[a, block] * (1.0 - w) + out[b, block] * w
    return out


def motion(vecs: np.ndarray, block: slice) -> float:
    """Summed frame-to-frame change of a block, over consecutive frames that both have it."""
    present = _present(vecs, block)
    both = present[1:] & present[:-1]
    return float(np.abs(np.diff(vecs[:, block], axis=0))[both].sum())


def mirror(vecs: np.ndarray) -> np.ndarray:
    """Left-right mirror: negate x, swap the hands and the left/right pose points."""
    out = np.array(vecs, dtype=np.float32, copy=True)
    out[:, 0::_C] *= -1.0  # every point starts on a multiple of 3: x of every point
    out[:, _LEFT], out[:, _RIGHT] = out[:, _RIGHT].copy(), out[:, _LEFT].copy()
    for a, b in _POSE_PAIRS:
        pa = slice(ls.POSE_START + a * _C, ls.POSE_START + (a + 1) * _C)
        pb = slice(ls.POSE_START + b * _C, ls.POSE_START + (b + 1) * _C)
        out[:, pa], out[:, pb] = out[:, pb].copy(), out[:, pa].copy()
    return out


def natural_resample(times: np.ndarray, vecs: np.ndarray, duration: float) -> np.ndarray:
    """Resample [0, min(duration, MAX_TAKE_S)] to FPS frames per second."""
    span = min(float(duration), MAX_TAKE_S)
    if span <= 0:
        raise ValueError("duration must be positive")
    n = int(math.ceil(span * FPS - 1e-9)) + 1
    return resample_window(np.asarray(times), np.asarray(vecs), end_time=span, length=n, span_s=span)


def clean_clip(times: np.ndarray, vecs: np.ndarray, duration: float) -> np.ndarray | None:
    """One playable take from a stored MS-ASL sequence, or None if the hands are mostly missing."""
    times, vecs = np.asarray(times, dtype=np.float64), np.asarray(vecs, dtype=np.float32)
    keep = (times >= 0.0) & (times <= duration + 1e-9)
    t, v = times[keep], vecs[keep]
    if len(t) == 0 or hand_coverage(v) < MIN_HAND_COVERAGE:
        return None
    v = fill_gaps(t, v)
    if motion(v, _LEFT) > motion(v, _RIGHT):
        v = mirror(v)
    return natural_resample(t, v, duration)


def _compare_form(take: np.ndarray) -> np.ndarray:
    n = len(take)
    if n < 2:
        return np.repeat(take, COMPARE_STEPS, axis=0)
    span = (n - 1) / FPS
    return resample_window(np.arange(n) / FPS, take, end_time=span, length=COMPARE_STEPS, span_s=span)


def rank(takes: list[np.ndarray]) -> list[int]:
    """Indices from most to least typical (smallest summed distance to the others first)."""
    if len(takes) <= 1:
        return list(range(len(takes)))
    flat = np.stack([_compare_form(t) for t in takes]).reshape(len(takes), -1).astype(np.float64)
    sq = (flat**2).sum(axis=1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T, 0.0))
    return [int(i) for i in np.argsort(dist.sum(axis=1), kind="stable")]
```

- [ ] **Step 4: Run the tests, the full suite, ruff and black.** Everything must pass.
  If black reformats a line, accept its formatting.

- [ ] **Step 5: Commit**

```bash
git add senseless/sign/extra_library.py senseless/tests/test_extra_library.py
git commit -m "Clean and rank MS-ASL takes for the signing figure"
```

---

### Task 3: Review page, choices and library build (`sign/extra_library.py`, part 2)

**Files:**
- Modify: `senseless/sign/extra_library.py`
- Modify: `senseless/sign/library.py`
- Modify: `senseless/common/config.py`: add the `PATHS.sign_library_extra` field
- Test: `senseless/tests/test_extra_library.py`, `senseless/tests/test_library.py`

**Interfaces:**
- Consumes:
  - `clean_clip` and `rank` from Task 2;
  - `msasl.read_manifest`, `msasl.sequence_path`, `msasl.ANIM_WORK_DIR` and
    `msasl.Clip(status="extracted")` from Task 1.
- Produces:
  - `collect_candidates(clips, work) -> dict[str, list[np.ndarray]]`;
  - `save_candidates(candidates, path)` and `load_candidates(path)`;
  - `choose(candidates, choices) -> dict[str, np.ndarray]`;
  - `build_review_html(candidates) -> str`;
  - `library.add_extra(library, extra_path) -> dict`;
  - `PATHS.sign_library_extra`;
  - the CLI steps `review` and `build`.

- [ ] **Step 1: Write the failing tests**

Add `import json`, `import pytest` and `from senseless.collect import msasl` to the imports
at the top of `senseless/tests/test_extra_library.py`, then append:

```python
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
```

Append to `senseless/tests/test_library.py`. Use the imports that file already has
(`np`, `library` and so on); add any that are missing.

```python
def test_add_extra_keeps_our_own_takes(tmp_path) -> None:
    from senseless.sign import library as lib

    ours = {"HELLO": np.zeros((45, 153), np.float32)}
    extra = {"HELLO": np.ones((61, 153), np.float32), "MILK": np.ones((61, 153), np.float32)}
    path = lib.save_library(extra, tmp_path / "extra.npz")
    merged = lib.add_extra(ours, path)
    assert sorted(merged) == ["HELLO", "MILK"] and merged["HELLO"].shape == (45, 153)
    assert lib.add_extra(ours, tmp_path / "missing.npz") is ours
```

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement**

**3a.** In `senseless/common/config.py`, directly below the
`sign_library: Path = MODELS_DIR / "sign_library.npz"` field of the paths dataclass, add:

```python
    # MS-ASL takes for words we did not record (sign/extra_library.py); merged under ours.
    sign_library_extra: Path = MODELS_DIR / "sign_library_extra.npz"
```

**3b.** In `senseless/sign/library.py`:

- add this function after `load_library`:

```python
def add_extra(
    library: dict[str, np.ndarray], extra_path: Path | str = PATHS.sign_library_extra
) -> dict[str, np.ndarray]:
    """Add the MS-ASL takes (sign/extra_library.py) for words we did not record; ours win."""
    extra_path = Path(extra_path)
    if not extra_path.exists():
        return library
    return {**load_library(extra_path), **library}
```

- in `main()`, change `library = build_library(args.data_dir)` to
  `library = add_extra(build_library(args.data_dir))`;
- in the module docstring, add one sentence after "Saved as ...":
  "Takes for words we did not record (from MS-ASL, ``sign/extra_library.py``) are
  merged in from ``models/sign_library_extra.npz`` if it exists; our own takes win."

**3c.** Append to `senseless/sign/extra_library.py`. Add `import argparse`, `import json`
and `from pathlib import Path` to its imports.

```python
def collect_candidates(clips: list, work: Path | str) -> dict[str, list[np.ndarray]]:
    """Cleaned takes per word from the extracted clips, most typical first."""
    from senseless.collect import msasl

    groups: dict[str, list[np.ndarray]] = {}
    for clip in clips:
        if clip.status != "extracted":
            continue
        path = msasl.sequence_path(Path(work), clip)
        if not path.exists():
            continue
        with np.load(path) as seq:
            take = clean_clip(seq["times"], seq["vecs"], float(seq["duration"]))
        if take is not None:
            groups.setdefault(clip.label, []).append(take)
    return {w: [takes[i] for i in rank(takes)] for w, takes in sorted(groups.items())}


def save_candidates(candidates: dict[str, list[np.ndarray]], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{w}__{k}": t for w, takes in candidates.items() for k, t in enumerate(takes)}
    np.savez_compressed(path, **arrays)
    return path


def load_candidates(path: Path | str) -> dict[str, list[np.ndarray]]:
    found: dict[str, dict[int, np.ndarray]] = {}
    with np.load(path) as archive:
        for key in archive.files:
            word, k = key.rsplit("__", 1)
            found.setdefault(word, {})[int(k)] = archive[key]
    return {w: [ks[k] for k in sorted(ks)] for w, ks in sorted(found.items())}


def choose(
    candidates: dict[str, list[np.ndarray]], choices: dict[str, int | str]
) -> dict[str, np.ndarray]:
    """One take per word: the choice's index (default 0, the automatic pick), or none if rejected."""
    for word, pick in choices.items():
        if word not in candidates:
            raise ValueError(f"choices name an unknown word: {word}")
        if pick != "reject" and not (
            isinstance(pick, int) and 0 <= pick < len(candidates[word])
        ):
            raise ValueError(f"bad choice for {word}: {pick!r}")
    return {
        w: takes[choices.get(w, 0)]
        for w, takes in candidates.items()
        if choices.get(w) != "reject"
    }


def _preview(take: np.ndarray) -> list:
    """15 FPS, x/y only, in hundredths: [pose 18, left hand 42 or [], right hand 42 or []]."""
    frames = []
    for vec in take[::2]:
        parts = []
        for block in (_POSE, _LEFT, _RIGHT):
            xy = vec[block].reshape(-1, _C)[:, :2]
            keep = block == _POSE or np.any(vec[block])
            parts.append(np.round(xy * 100).astype(int).ravel().tolist() if keep else [])
        frames.append(parts)
    return frames


def build_review_html(candidates: dict[str, list[np.ndarray]]) -> str:
    """A self-contained page that plays every word's takes as the app's stick figure."""
    from senseless.common.config import UI
    from senseless.ui.figure import HAND_EDGES, HEAD_RADIUS, POSE_EDGES

    data = {
        "words": [
            {"word": w, "takes": [_preview(t) for t in takes]} for w, takes in candidates.items()
        ],
        "pose_edges": POSE_EDGES,
        "hand_edges": HAND_EDGES,
        "head": HEAD_RADIUS,
        "extent": list(UI.figure_extent),
    }
    return _PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign takes review</title>
<style>
body{font-family:system-ui,sans-serif;background:#13262b;color:#e8eef0;margin:16px}
header{position:sticky;top:0;background:#13262b;padding:8px 0;display:flex;gap:12px;
  align-items:center;z-index:1;flex-wrap:wrap}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}
.card{background:#1c363d;border-radius:8px;padding:8px}
.card.rejected{opacity:.35}
.card h3{margin:0 0 4px;font-size:15px}
canvas{background:#0e1d21;border-radius:4px;width:100%;height:auto}
.btns button{margin:4px 4px 0 0;min-width:28px}
button.on{outline:2px solid #f0a030}
</style></head><body>
<header><strong>Sign takes review</strong><span id="count"></span>
<button id="export">Export choices</button></header>
<p>Each card loops the automatic pick (1). Click another number to use an alternate, or
Reject to leave the word out. Then export <code>choices.json</code> and run
<code>python -m senseless.sign.extra_library build --choices choices.json</code>.</p>
<div id="grid"></div>
<script>
const D = __DATA__;
const choice = {};
const [xmin, xmax, ymin, ymax] = D.extent;
const cards = [];
function draw(ctx, w, h, f) {
  ctx.clearRect(0, 0, w, h);
  const [pose, left, right] = f;
  if (!pose.some(v => v !== 0)) return;
  const s = Math.min(w / (xmax - xmin), h / (ymax - ymin));
  const ox = w / 2 - s * (xmin + xmax) / 2, oy = h / 2 - s * (ymin + ymax) / 2;
  const P = (a, i) => [ox + s * Math.min(xmax, Math.max(xmin, a[2 * i] / 100)),
                       oy + s * Math.min(ymax, Math.max(ymin, a[2 * i + 1] / 100))];
  const seg = (a, i, j, col) => { const p = P(a, i), q = P(a, j); ctx.strokeStyle = col;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); };
  ctx.lineWidth = 2;
  for (const [i, j] of D.pose_edges) seg(pose, i, j, "#dfe7ea");
  const n = P(pose, 0);
  ctx.beginPath(); ctx.arc(n[0], n[1], D.head * s, 0, 2 * Math.PI); ctx.stroke();
  ctx.lineWidth = 1.5;
  if (left.length) for (const [i, j] of D.hand_edges) seg(left, i, j, "#4cc38a");
  if (right.length) for (const [i, j] of D.hand_edges) seg(right, i, j, "#f0a030");
}
function updateCount() {
  const rejected = Object.values(choice).filter(c => c === "reject").length;
  document.getElementById("count").textContent =
    `${D.words.length} words, ${rejected} rejected`;
}
for (const item of D.words) {
  const card = document.createElement("div"); card.className = "card";
  const title = document.createElement("h3"); title.textContent = item.word;
  const canvas = document.createElement("canvas"); canvas.width = 170; canvas.height = 160;
  const btns = document.createElement("div"); btns.className = "btns";
  const state = { item, canvas, ctx: canvas.getContext("2d"), pick: 0, t0: performance.now(), last: -1 };
  const buttons = [];
  const select = (pick) => {
    choice[item.word] = pick; state.pick = pick === "reject" ? 0 : pick;
    state.t0 = performance.now(); state.last = -1;
    card.classList.toggle("rejected", pick === "reject");
    buttons.forEach((b, k) => b.classList.toggle("on", (k === item.takes.length ? "reject" : k) === pick));
    updateCount();
  };
  item.takes.forEach((_, k) => {
    const b = document.createElement("button"); b.textContent = String(k + 1);
    b.onclick = () => select(k); buttons.push(b); btns.appendChild(b);
  });
  const rej = document.createElement("button"); rej.textContent = "Reject";
  rej.onclick = () => select("reject"); buttons.push(rej); btns.appendChild(rej);
  buttons[0].classList.add("on");
  card.append(title, canvas, btns); document.getElementById("grid").appendChild(card);
  cards.push(state);
}
function tick(now) {
  for (const c of cards) {
    const frames = c.item.takes[c.pick];
    const k = Math.floor((now - c.t0) / 1000 * 15) % (frames.length + 8);
    if (k !== c.last) { c.last = k; draw(c.ctx, 170, 160, frames[Math.min(k, frames.length - 1)]); }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
updateCount();
document.getElementById("export").onclick = () => {
  const out = {};
  for (const [w, c] of Object.entries(choice)) if (c !== 0) out[w] = c;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
  a.download = "choices.json"; a.click();
};
</script></body></html>
"""


def main() -> None:
    from senseless.collect import msasl
    from senseless.common.config import DATA_DIR, PATHS
    from senseless.sign.library import add_extra, build_library, save_library

    parser = argparse.ArgumentParser(description="MS-ASL takes for the signing figure.")
    parser.add_argument("step", choices=["review", "build"])
    parser.add_argument("--work", default=str(msasl.ANIM_WORK_DIR))
    parser.add_argument("--choices", default=None, help="choices.json from the review page.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out", default=str(PATHS.sign_library))
    args = parser.parse_args()
    work = Path(args.work)
    cand_path = work / "candidates.npz"

    if args.step == "review":
        clips = msasl.read_manifest(work / "manifest.csv")
        candidates = collect_candidates(clips, work)
        save_candidates(candidates, cand_path)
        page = work / "review.html"
        page.write_text(build_review_html(candidates), encoding="utf-8")
        n = sum(len(t) for t in candidates.values())
        print(f"{len(candidates)} words, {n} candidate takes -> {page}")
        missing = sorted({c.label for c in clips} - set(candidates))
        if missing:
            print(f"no usable clip for {len(missing)} words: {', '.join(missing)}")
        return

    candidates = load_candidates(cand_path)
    choices = json.loads(Path(args.choices).read_text()) if args.choices else {}
    extra = choose(candidates, choices)
    save_library(extra, PATHS.sign_library_extra)
    ours = build_library(args.data_dir)
    library = add_extra(ours, PATHS.sign_library_extra)
    path = save_library(library, args.out)
    print(f"saved {path}: {len(library)} words ({len(library) - len(ours)} from MS-ASL)")


if __name__ == "__main__":
    main()
```

Accept black's formatting of the long lines.

- [ ] **Step 4: Run the tests, the full suite, ruff and black.** Everything must pass.
  Then generate a sample page from the test data and open it in a browser to check it
  draws. Skip this check if no browser is available, and say so in the report.

- [ ] **Step 5: Commit**

```bash
git add senseless/sign/extra_library.py senseless/sign/library.py senseless/common/config.py senseless/tests/test_extra_library.py senseless/tests/test_library.py
git commit -m "Review page and library build for MS-ASL figure takes"
```

---

### Task 4: Play takes of any length, and document the new library step

**Files:**
- Modify: `senseless/ui/app.py` (`_play_tick_body`)
- Test: `senseless/tests/test_ui_app.py`
- Modify: `ONBOARDING.md`, `senseless_retraining.md`, `ARCHITECTURE.md`

- [ ] **Step 1: Write the failing test** (append to `senseless/tests/test_ui_app.py`)

```python
def test_a_longer_take_plays_for_its_own_length(speech_app) -> None:
    speech_app.root.after = lambda ms, fn: None
    speech_app.library["HELLO"] = np.repeat(speech_app.library["HELLO"][:1], 90, axis=0)
    speech_app.handle_event(SpeechText("hello", True))
    speech_app._play_tick()
    assert speech_app._playing == "HELLO"
    speech_app._play_t0 -= 2.0  # past 45 frames, inside 90
    speech_app._play_tick()
    assert speech_app._playing == "HELLO"
    speech_app._play_t0 -= 1.1  # past 90 frames
    speech_app._play_tick()
    assert speech_app._playing is None
```

- [ ] **Step 2: Run it and confirm it fails.** Today the take ends after 45 frames.

- [ ] **Step 3: Implement.** In `_play_tick_body`, change
  `index = frame_index(now - self._play_t0)` to:

```python
            index = frame_index(now - self._play_t0, len(self.library[self._playing]))
```

- [ ] **Step 4: Docs**
  - `ONBOARDING.md`:
    - in the table row for `models/sign_library.npz`, change the description to
      "One take per word for the signing figure: ours, plus MS-ASL takes for other
      words";
    - after the "Rebuild the figure library" step, add one sentence: "Words you did not
      record keep their MS-ASL takes from `models/sign_library_extra.npz`
      (`python -m senseless.sign.extra_library`, see its docstring)."
  - `senseless_retraining.md`, step 8: after the command block, add: "This keeps the
    MS-ASL takes for words you did not record (`models/sign_library_extra.npz`). A word
    you record yourself replaces its MS-ASL take."
  - `ARCHITECTURE.md`: add a module-table row next to the `sign/library.py` row:
    `| [sign/extra_library.py](senseless/sign/extra_library.py) | MS-ASL takes for words we did not record: clean, rank, review page, build | ✅ |`.
    Match the neighbouring rows' format. Update the test count in that file to the
    number `pytest --collect-only -q` reports.

- [ ] **Step 5: Run the full suite, ruff and black.** Everything must pass.

- [ ] **Step 6: Commit**

```bash
git add senseless/ui/app.py senseless/tests/test_ui_app.py ONBOARDING.md senseless_retraining.md ARCHITECTURE.md
git commit -m "Play figure takes of any length; document the MS-ASL takes"
```

---

### Task 5: Build the takes (operational, run by the controller)

There is no new code in this task. It downloads from YouTube, which the user approved
in the design.

1. `"../Senseless/.venv-msasl/Scripts/python" -m senseless.collect.msasl select-anim`.
   Expect about 750 clips for 151 words in `C:\Senseless_anim\manifest.csv`.
2. Run `download --work C:\Senseless_anim --pause 5`, then `extract --work C:\Senseless_anim`.
   Use `.venv-msasl`, detached, and log to `C:\Senseless_anim\run.log`.
3. `"../Senseless/.venv/Scripts/python" -m senseless.sign.extra_library review`. Report
   the counts and the words with no take. Give the user `C:\Senseless_anim\review.html`.
4. After the user exports `choices.json`, run
   `python -m senseless.sign.extra_library build --choices <path>`. Check the
   `models/sign_library.npz` word count, then run the app windowed and say a few new
   words.
5. Tell the user which files to copy to the Pi: `sign_library.npz` and
   `sign_library_extra.npz`.
