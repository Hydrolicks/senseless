# Hands-free Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add a `--auto` mode to the sign recorder. Each take starts when a hand appears.
The recorder re-arms once the hands have been out of frame for 0.5 s. BACKSPACE undoes
the last saved take.

**Architecture:** the take logic moves out of the camera loop in
`senseless/collect/recorder.py` into a pure `TakeMachine` class, which has no camera and
no cv2, so it can be unit-tested with synthetic timestamps. `main()` feeds it frames,
saves the takes it returns, and keeps a session list of saved paths for undo.

**Tech Stack:** Python 3.11, numpy, OpenCV (GUI only), pytest. ruff and black use a line
length of 100.

## Global Constraints

- Without `--auto`, the recorder behaves exactly as before. SPACE arms one take, and
  after the take it returns to idle.
- The re-arm clear time is `SIGN.collect_clear_s = 0.5`. Hands must be absent for that
  long without a break, and any hand frame restarts the clock.
- Takes are `(SIGN.window_length, ls.FEATURE_DIM)` = `(45, 153)` float32, built with
  `resample_window(times, vecs, end_time=onset + span, length=45, span_s=span)`. This is
  the existing take format.
- Undo only deletes files saved in the current run.
- cv2 stays lazily imported inside `main()`/`_draw()`. `senseless.collect.recorder`
  must import without cv2.
- Run the tests from the worktree root, `C:\Users\Asaf Amrani\Desktop\EE Engineering\Fourth Year\Senseless-autorec`,
  with the main checkout's venv:
  `"../Senseless/.venv/Scripts/python" -m pytest -q senseless/tests`.
  `python -m` puts the current folder first on `sys.path`, so the worktree's `senseless`
  package is the one imported.

---

### Task 1: `TakeMachine`, `undo_last` and the clear-time config

**Files:**
- Modify: `senseless/common/config.py` (`SignConfig`, after `release_s`)
- Modify: `senseless/collect/recorder.py` (add the class and the helper, leaving `main()` unchanged)
- Create: `senseless/tests/test_recorder_machine.py`

**Interfaces:**
- Produces:
  - `TakeMachine(span_s: float, auto: bool = False, clear_s: float = SIGN.collect_clear_s, length: int = SIGN.window_length)`
  - attributes: `state: str`, one of `"idle"`, `"armed"`, `"recording"`, `"waiting"` or
    `"paused"`; `onset: float`; `span_s: float`; `frames: int` (the frame count of the
    last completed take)
  - methods: `step(stamp: float, vec: np.ndarray, hands_present: bool) -> np.ndarray | None`,
    `arm() -> None` and `toggle_pause() -> None`
  - `undo_last(saved: list[Path]) -> Path | None`

- [ ] **Step 1: Add the config value**

In `senseless/common/config.py`, in `SignConfig`, directly after the line
`release_s: float = 0.3  # hands must be out of view this long before the next sign`,
add:

```python
    # Recorder --auto (collect/recorder.py): hands must be out of view this long, without
    # a break, before the next take is armed. Longer than release_s, because a missed
    # hand mid-sign must not re-arm and start a take on the second half of a sign.
    collect_clear_s: float = 0.5
```

- [ ] **Step 2: Write the failing tests**

Create `senseless/tests/test_recorder_machine.py`:

```python
"""TakeMachine: the recorder's take logic, driven by synthetic frames (no camera)."""

from pathlib import Path

import numpy as np

from senseless.collect.recorder import TakeMachine, undo_last
from senseless.common import landmark_schema as ls

SPAN = 1.0
DT = 0.1


def _feed(machine: TakeMachine, t0: float, n: int, hands: bool) -> tuple[list, float]:
    """Feed n frames DT apart from t0 (each vector filled with t + 1); return takes, next t."""
    takes = []
    t = t0
    for _ in range(n):
        out = machine.step(t, np.full(ls.FEATURE_DIM, t + 1.0, dtype=np.float32), hands)
        if out is not None:
            takes.append(out)
        t = round(t + DT, 6)
    return takes, t


def test_manual_take_starts_on_hand_after_arm_and_returns_to_idle() -> None:
    m = TakeMachine(span_s=SPAN, auto=False, clear_s=0.5, length=45)
    takes, t = _feed(m, 0.0, 5, hands=True)
    assert takes == [] and m.state == "idle"  # never records before arm()
    m.toggle_pause()
    assert m.state == "idle"  # pause is an auto-mode control
    m.arm()
    _, t = _feed(m, t, 3, hands=False)
    assert m.state == "armed"
    takes, t = _feed(m, t, 12, hands=True)
    assert len(takes) == 1
    assert takes[0].shape == (45, ls.FEATURE_DIM) and takes[0].dtype == np.float32
    assert m.state == "idle"
    assert m.frames >= 11


def test_take_covers_onset_to_onset_plus_span() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    _, t = _feed(m, 0.0, 3, hands=False)  # armed, no hand yet
    takes, _ = _feed(m, t, 12, hands=True)  # onset at t = 0.3
    assert m.onset == 0.3
    take = takes[0]
    np.testing.assert_allclose(take[0], 1.3, atol=1e-4)  # vector at the onset (t + 1)
    np.testing.assert_allclose(take[-1], 2.3, atol=1e-4)  # vector at onset + span


def test_auto_rearms_only_after_clear_time_without_hands() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    assert m.state == "armed"  # auto mode starts armed
    takes, t = _feed(m, 0.0, 12, hands=True)
    assert len(takes) == 1 and m.state == "waiting"
    _, t = _feed(m, t, 5, hands=True)  # hands still up: keep waiting
    assert m.state == "waiting"
    _, t = _feed(m, t, 5, hands=False)  # 0.4 s clear: not yet
    assert m.state == "waiting"
    _, t = _feed(m, t, 2, hands=False)  # 0.6 s clear
    assert m.state == "armed"
    takes, _ = _feed(m, t, 12, hands=True)
    assert len(takes) == 1  # the next take


def test_auto_short_dropout_does_not_rearm() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    _, t = _feed(m, 0.0, 12, hands=True)
    _, t = _feed(m, t, 4, hands=False)  # 0.3 s without hands (a tracker dropout)
    _, t = _feed(m, t, 1, hands=True)  # hand back: the clock restarts
    _, t = _feed(m, t, 4, hands=False)
    assert m.state == "waiting"


def test_pause_blocks_arming_and_resume_waits_for_clear() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    m.toggle_pause()
    assert m.state == "paused"
    takes, t = _feed(m, 0.0, 20, hands=True)
    assert takes == [] and m.state == "paused"
    m.toggle_pause()
    assert m.state == "waiting"  # hands already up must not start a take
    takes, _ = _feed(m, t, 5, hands=True)
    assert takes == [] and m.state == "waiting"


def test_pause_mid_recording_discards_the_take() -> None:
    m = TakeMachine(span_s=SPAN, auto=True, clear_s=0.5, length=45)
    _, t = _feed(m, 0.0, 5, hands=True)
    assert m.state == "recording"
    m.toggle_pause()
    m.toggle_pause()
    takes, _ = _feed(m, t, 20, hands=True)
    assert takes == []


def test_undo_last_deletes_newest_session_file(tmp_path: Path) -> None:
    a, b = tmp_path / "0000.npy", tmp_path / "0001.npy"
    for p in (a, b):
        np.save(p, np.zeros(3))
    saved = [a, b]
    assert undo_last(saved) == b
    assert not b.exists() and a.exists() and saved == [a]
    assert undo_last(saved) == a
    assert not a.exists()
    assert undo_last(saved) is None
```

- [ ] **Step 3: Run the tests to check they fail**

Run: `"../Senseless/.venv/Scripts/python" -m pytest -q senseless/tests/test_recorder_machine.py`
Expected: an ImportError for `TakeMachine`, so collection fails.

- [ ] **Step 4: Implement**

In `senseless/collect/recorder.py`:

1. Add `from pathlib import Path` to the imports. The other imports are already there.
2. Add the following after the `_SPACE_KEY = 32` line and before `def _draw(`:

```python
class TakeMachine:
    """Take logic of the recorder, with no camera and no GUI, so it can be tested.

    Manual mode: ``idle`` -> ``arm()`` -> ``armed`` -> first hand frame -> ``recording``
    -> one span later the take is returned -> ``idle``.
    Auto mode starts ``armed``. After a take it goes to ``waiting`` and re-arms once no
    hand has been seen for ``clear_s`` without a break. ``toggle_pause()`` switches to
    ``paused`` (dropping a take in progress) and back to ``waiting``.
    """

    def __init__(
        self,
        span_s: float,
        auto: bool = False,
        clear_s: float = SIGN.collect_clear_s,
        length: int = SIGN.window_length,
    ) -> None:
        self.span_s = span_s
        self.auto = auto
        self.clear_s = clear_s
        self.length = length
        self.state = "armed" if auto else "idle"
        self.onset = 0.0
        self.frames = 0
        self._times: list[float] = []
        self._vecs: list[np.ndarray] = []
        self._clear_since: float | None = None

    def arm(self) -> None:
        """Manual mode: arm one take (only from idle)."""
        if not self.auto and self.state == "idle":
            self.state = "armed"

    def toggle_pause(self) -> None:
        """Auto mode: pause (dropping any take in progress) or resume into waiting."""
        if not self.auto:
            return
        if self.state == "paused":
            self.state, self._clear_since = "waiting", None
        else:
            self.state, self._times, self._vecs = "paused", [], []

    def step(self, stamp: float, vec: np.ndarray, hands_present: bool) -> np.ndarray | None:
        """Advance by one frame; return a finished ``(length, FEATURE_DIM)`` take or None."""
        if self.state == "armed" and hands_present:
            self.state, self.onset = "recording", stamp
            self._times, self._vecs = [stamp], [vec]
        elif self.state == "recording":
            self._times.append(stamp)
            self._vecs.append(vec)
            if stamp - self.onset >= self.span_s:
                take = resample_window(
                    np.array(self._times),
                    np.stack(self._vecs),
                    end_time=self.onset + self.span_s,
                    length=self.length,
                    span_s=self.span_s,
                )
                self.frames = len(self._times)
                self._times, self._vecs = [], []
                self.state = "waiting" if self.auto else "idle"
                self._clear_since = None
                return take
        elif self.state == "waiting":
            if hands_present:
                self._clear_since = None
            elif self._clear_since is None:
                self._clear_since = stamp
            elif stamp - self._clear_since >= self.clear_s:
                self.state = "armed"
        return None


def undo_last(saved: list[Path]) -> Path | None:
    """Delete the newest take saved this session; return its path, or None if none are left."""
    if not saved:
        return None
    path = saved.pop()
    path.unlink(missing_ok=True)
    return path
```

- [ ] **Step 5: Run the tests to check they pass, then run the full suite and the linters**

Run:

```
"../Senseless/.venv/Scripts/python" -m pytest -q senseless/tests/test_recorder_machine.py
"../Senseless/.venv/Scripts/python" -m pytest -q senseless/tests
"../Senseless/.venv/Scripts/python" -m ruff check senseless
"../Senseless/.venv/Scripts/python" -m black --check senseless
```

Expected: all pass. A Tk test may skip intermittently.

- [ ] **Step 6: Commit**

```bash
git add senseless/common/config.py senseless/collect/recorder.py senseless/tests/test_recorder_machine.py
git commit -m "Add the recorder's TakeMachine with an auto re-arm mode and undo helper"
```

---

### Task 2: Wire `--auto`, pause and undo into the recorder, and update the docs

**Files:**
- Modify: `senseless/collect/recorder.py` (module docstring and `main()`)
- Modify: `instructions.md` (the "Record samples" step, around lines 170-178)
- Modify: `ONBOARDING.md` (the recording steps, around lines 206-211)
- Modify: `ARCHITECTURE.md` (the `collect/recorder.py` row, line 80)
- Test: `senseless/tests/test_collect_cli.py` (the existing import test must still pass)

**Interfaces:**
- Consumes: `TakeMachine` and `undo_last` from Task 1.

- [ ] **Step 1: Replace the module docstring of `senseless/collect/recorder.py`**

```python
"""Interactive sign data-collection recorder (DEV TOOL, PC).

Live monitoring window plus a take recorder. A take starts automatically on the
first frame a hand appears, runs for one window span (~1.5 s, see
``sign/window.py``) and is resampled to the model's ``window_length`` steps, so a
take means the same stretch of time at any frame rate. Starting at the onset means
the sign fills the window from the moment your hands enter frame: no wasted empty
lead-in, and no clipping of the start.

Two ways to arm a take (the logic lives in ``TakeMachine``):

- default: press SPACE before each take.
- ``--auto``: hands-free. After a take, hide your hands (e.g. behind your back); once
  no hand has been seen for ``SIGN.collect_clear_s`` (0.5 s) the next take is armed.
  SPACE pauses and resumes.

BACKSPACE deletes the last take saved in this session (repeatable). The overlay
reports which hands are detected, so you can tell an empty window from a real one.

Windows are saved under ``data/<label>/`` via ``collect.dataset``, using the same
normalized landmark pipeline as inference.

Run via ``python -m senseless.collect --label WORD [--auto]`` (see instructions.md).
"""
```

- [ ] **Step 2: Rewrite `main()`**

Add `_BACKSPACE_KEY = 8` directly below `_SPACE_KEY = 32`, then this table below it:

```python
_STATUS = {
    "idle": "SPACE = arm a take",
    "armed": "armed - waiting for a hand...",
    "waiting": "hands down to re-arm",
    "paused": "PAUSED - SPACE to resume",
}
```

Replace the whole `main()` function with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Record sign samples (capture at hand onset).")
    parser.add_argument("--label", required=True, help="Vocabulary word to record.")
    parser.add_argument("--samples", type=int, default=30, help="Target number of takes.")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Hands-free: re-arm after the hands leave the frame; SPACE pauses.",
    )
    parser.add_argument("--backend", choices=landmarks.BACKEND_NAMES, default=None)
    parser.add_argument("--source", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--parallel", action="store_true", help="Run pose and hands in separate processes."
    )
    args = parser.parse_args()

    import cv2

    backend = landmarks.create_backend(args.backend, parallel=True if args.parallel else None)
    src = capture.LatestFrameGrabber(capture.open_frame_source(source=args.source))
    count = dataset.count_samples(args.label)
    machine = TakeMachine(default_span_s(), auto=args.auto)
    saved: list[Path] = []
    start = time.perf_counter()
    window_name = f"Senseless collect - {args.label}"
    space = "SPACE = pause" if args.auto else "SPACE = arm"
    controls = f"{space}  |  BACKSPACE = undo  |  q = quit"
    print(f"Recording '{args.label}'. {controls}. Already have {count}.")

    try:
        with backend, src:
            while (item := src.read_stamped()) is not None:
                frame_rgb, stamp = item
                raw = backend.extract(frame_rgb, int((stamp - start) * 1000))
                vec = landmarks.frame_landmarks_to_vector(raw)
                hands_present = raw.left_hand is not None or raw.right_hand is not None

                take = machine.step(stamp, vec, hands_present)
                if take is not None:
                    path = dataset.save_window(take, args.label)
                    saved.append(path)
                    count += 1
                    print(f"saved {path.name}  ({count}/{args.samples}, {machine.frames} frames)")

                left = "L" if raw.left_hand is not None else "-"
                right = "R" if raw.right_hand is not None else "-"
                if machine.state == "recording":
                    status = f"REC {stamp - machine.onset:.1f}/{machine.span_s:.1f} s"
                else:
                    status = _STATUS[machine.state]
                lines = [
                    f"{args.label}    saved {count}/{args.samples}",
                    f"hands: {left}{right}    {status}",
                    controls,
                ]
                cv2.imshow(window_name, _draw(frame_rgb, raw, lines, machine.state == "recording"))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == _SPACE_KEY:
                    if args.auto:
                        machine.toggle_pause()
                    else:
                        machine.arm()
                elif key == _BACKSPACE_KEY:
                    path = undo_last(saved)
                    if path is not None:
                        count -= 1
                        print(f"deleted {path.name}  ({count}/{args.samples})")
    finally:
        cv2.destroyAllWindows()
    print(f"Done. {count} samples for '{args.label}' in {dataset.label_dir(args.label)}.")
```

- [ ] **Step 3: Update `instructions.md`**

In step "2. **Record samples**", replace the paragraph that starts "Press **SPACE** to
record one take" and ends "so you can collect across multiple sessions." with:

```markdown
   Press **SPACE** to arm a take. Capture starts by itself when a hand appears and
   lasts ~1.5 s (resampled to a 45-step window). The overlay shows which hands are
   detected (`hands: LR`), a red dot + `REC` while capturing, and `saved N/40`.
   **BACKSPACE** deletes the last take of this session; **q** quits. Each take is one
   `(45, 153)` array under `data/<label>/`, already normalized. Re-running the same
   `--label` resumes the count, so you can collect across multiple sessions.

   **Hands-free (`--auto`):** add `--auto` and no key is needed per take. Sign, then
   hide both hands (e.g. behind your back); after 0.5 s with no hand in view the next
   take is armed (`hands down to re-arm` → `armed`). SPACE pauses and resumes.
```

- [ ] **Step 4: Update `ONBOARDING.md`**

Replace these three bullets:

```markdown
   - Press SPACE to arm a take.
   - Raise your hands and sign. Capture starts by itself when a hand appears.
   - Lower your hands between takes.
```

with:

```markdown
   - Press SPACE to arm a take, or add `--auto` to re-arm automatically.
   - Raise your hands and sign. Capture starts by itself when a hand appears.
   - Between takes, move your hands out of view (with `--auto`, for at least 0.5 s).
   - BACKSPACE deletes the last take if you fumbled it.
```

- [ ] **Step 5: Update `ARCHITECTURE.md`**

In the `collect/recorder.py` row (line 80), replace "Keypress-per-take data-collection CLI"
with "Data-collection CLI (SPACE per take, or hands-free `--auto`)".

- [ ] **Step 6: Verify**

```
"../Senseless/.venv/Scripts/python" -m pytest -q senseless/tests
"../Senseless/.venv/Scripts/python" -m ruff check senseless
"../Senseless/.venv/Scripts/python" -m black --check senseless
"../Senseless/.venv/Scripts/python" -m senseless.collect --help
```

Expected: all tests pass, the linters are clean, and `--help` lists `--auto`. Do not open
the camera. The controller will ask the user to try it live.

- [ ] **Step 7: Commit**

```bash
git add senseless/collect/recorder.py instructions.md ONBOARDING.md ARCHITECTURE.md
git commit -m "Add hands-free --auto recording, pause and undo to the collector"
```
