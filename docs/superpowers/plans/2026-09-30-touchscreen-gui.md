# Senseless Touchscreen App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A full-screen Tkinter app on the Pi's 7" 800×480 touch screen with a Sign mode (onset-mode recognition, camera preview, word + sentence) and a Speech mode (Vosk transcript + a stick figure that signs spoken vocabulary words), one mode at a time.

**Architecture:** The Tk GUI is the main process. `ui/controller.py` runs one spawned, non-daemon worker process for the active mode (`sign/worker.py` or `asr/worker.py`); workers send preview frames and small event dataclasses back over two `DropOldestQueue`s that the GUI drains every 40 ms. Pure logic (`ui/state.py`, `ui/figure.py`, `sign/library.py`) is test-first; workers are tested with injected fake parts; the GUI has a hidden-window smoke test.

**Tech Stack:** Python 3.11, Tkinter (+ Pillow `ImageTk`, already installed as a matplotlib/mediapipe dependency), multiprocessing (spawn), NumPy, OpenCV (preview drawing), existing Senseless modules. Spec: `docs/superpowers/specs/2026-09-30-touchscreen-gui-design.md`.

## Global Constraints

- Python 3.11; no new pip dependencies. On the Pi, Tkinter comes from APT: `sudo apt install -y python3-tk` (the venv uses `--system-site-packages`).
- Screen 800×480, full screen on the Pi; `python -m senseless.ui --windowed` on a PC.
- One worker at a time. Workers are started with `multiprocessing.get_context("spawn")` and **`daemon=False`** (the sign worker's `ParallelBackend` starts its own processes; daemonic processes may not have children).
- Two worker→GUI queues, both `DropOldestQueue`: `frames` max `UI.frames_queue_maxsize` = 2; `events` max `RUNTIME.transcript_queue_maxsize` = 32. GUI polls every `UI.poll_ms` = 40 ms.
- Switch/stop timeout `UI.switch_timeout_s` = 5.0 s, then terminate.
- Sentence cap `UI.sentence_max_words` = 8; transcript cap `UI.transcript_max_lines` = 50; text scale steps `UI.text_scales` = (0.8, 1.0, 1.25, 1.5), default index 1.
- Sign worker hardware: `UI.camera` = "opencv", `UI.camera_source` = 0 (Logitech), `UI.perception_backend` = "lite", `UI.parallel_perception` = True. Speech: existing `find_input_device()` (Logitech mic matches the "USB Audio" hint).
- Palette: bg `#0F2A31`, panel `#0B2227`, text `#EAF3F4`, muted `#8FB3B8`, teal `#0E7C86`, coral `#E4572E`, word `#7FD6C8`.
- Aliases (spoken → label): "thank you"/"thanks" → THANKYOU, "hi" → HELLO, "bye"/"good bye" → GOODBYE.
- Sign figure: triggered by **final** speech results only; each take plays 45 frames at `SIGN.reference_fps` (30).
- Code style: black + ruff, line length 100; run `.venv\Scripts\python -m ruff check .` and `.venv\Scripts\python -m black --check .` before finishing a task.
- **Commits: this project commits only when the user explicitly asks.** Each task ends in a commit-ready state (tests, ruff, black green); the "Commit" step means: report the task done and commit only on the user's request, using the suggested message.
- All test commands below run from the repo root on the Windows dev PC: `.venv\Scripts\python -m pytest -o addopts="" -q <path>`.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `senseless/common/config.py` | modify | `UIConfig` + `UI` singleton; `PATHS.sign_library` |
| `senseless/common/events.py` | create | Worker→GUI event dataclasses |
| `senseless/ui/state.py` | create | `SignSentence`, `SpeechTranscript`, `TextScale`, `words_to_sign`, `highlight_spans` |
| `senseless/sign/worker.py` | create | `SignParts`, `make_preview`, `run_sign_worker` |
| `senseless/asr/worker.py` | create | `SpeechParts`, `run_speech_worker` |
| `senseless/ui/controller.py` | create | `ModeController`, `sign_entry`, `speech_entry` |
| `senseless/ui/app.py` | create | `SenselessApp` (Tk views, polling, dialogs, figure playback) |
| `senseless/ui/__main__.py` | create | `python -m senseless.ui` |
| `senseless/sign/library.py` | create | `medoid_index`, `build_library`, `save_library`, `load_library`, CLI |
| `senseless/ui/figure.py` | create | `Segment`, `Dot`, `figure_geometry`, `frame_index`, `rest_frame` |
| `deploy/senseless-ui.sh`, `deploy/senseless.desktop` | create | Autostart launcher |
| `PI_Instructions.md`, `ARCHITECTURE.md`, `README.md` | modify | Docs + on-Pi acceptance checklist |
| `senseless/tests/test_events.py`, `test_ui_state.py`, `test_sign_worker.py`, `test_speech_worker.py`, `test_controller.py`, `ui_fakes.py`, `test_ui_app.py`, `test_library.py`, `test_figure.py` | create | Tests |

---

# Stage 1: the app with both modes

### Task 1: UI config and worker events

**Files:**
- Modify: `senseless/common/config.py` (add `UIConfig`, `UI`; add `ModelPaths.sign_library`)
- Create: `senseless/common/events.py`
- Test: `senseless/tests/test_events.py`

**Interfaces:**
- Produces: `from senseless.common.config import UI` (fields listed below); `PATHS.sign_library: Path`.
- Produces: `from senseless.common.events import SignStatus, SignResult, SpeechText, WorkerReady, WorkerError` with fields:
  `SignStatus(state: str, progress: float, hands: bool, fps: float)`, `SignResult(word: str | None, best: str, confidence: float)`, `SpeechText(text: str, is_final: bool)`, `WorkerReady(mode: str)`, `WorkerError(message: str)`. All frozen dataclasses, picklable.

- [ ] **Step 1: Write the failing test**

```python
# senseless/tests/test_events.py
"""Worker -> GUI events must survive a trip between processes (pickle)."""

import pickle

from senseless.common.config import PATHS, UI
from senseless.common.events import SignResult, SignStatus, SpeechText, WorkerError, WorkerReady


def test_events_round_trip_through_pickle() -> None:
    events = [
        SignStatus("capturing", 0.5, True, 7.2),
        SignResult("HELLO", "HELLO", 0.93),
        SignResult(None, "WANT", 0.41),
        SpeechText("hello there", True),
        WorkerReady("sign"),
        WorkerError("Camera not found."),
    ]
    for event in events:
        assert pickle.loads(pickle.dumps(event)) == event


def test_ui_config_values() -> None:
    assert (UI.width, UI.height) == (800, 480)
    assert UI.frames_queue_maxsize == 2
    assert UI.text_scales[UI.default_text_scale] == 1.0
    assert ("thank you", "THANKYOU") in UI.sign_aliases
    assert PATHS.sign_library.name == "sign_library.npz"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_events.py`
Expected: FAIL with `ImportError` (no `UI` / no `senseless.common.events`).

- [ ] **Step 3: Implement**

In `senseless/common/config.py`, add to `ModelPaths` (after `hand_landmarker_task`):

```python
    # One representative take per word for the Speech-mode signing figure
    # (built by `python -m senseless.sign.library`).
    sign_library: Path = MODELS_DIR / "sign_library.npz"
```

Add this dataclass after `RuntimeConfig`:

```python
@dataclass(frozen=True)
class UIConfig:
    """Touchscreen app (senseless.ui): Tkinter on the Pi's 7" 800x480 display."""

    width: int = 800
    height: int = 480
    poll_ms: int = 40  # the GUI drains the worker queues this often
    preview_size: tuple[int, int] = (320, 240)  # (width, height) of the camera preview
    frames_queue_maxsize: int = 2  # preview images: newest wins
    switch_timeout_s: float = 5.0  # force-stop a worker that won't exit
    sentence_max_words: int = 8
    transcript_max_lines: int = 50
    text_scales: tuple[float, ...] = (0.8, 1.0, 1.25, 1.5)
    default_text_scale: int = 1  # index into text_scales
    # Sign worker: the Logitech webcam, lite models in parallel (fastest on a Pi 4).
    camera: str = "opencv"
    camera_source: int = 0
    perception_backend: str = "lite"
    parallel_perception: bool = True
    # Spoken phrase -> vocabulary label, for the signing figure.
    sign_aliases: tuple[tuple[str, str], ...] = (
        ("thank you", "THANKYOU"),
        ("thanks", "THANKYOU"),
        ("hi", "HELLO"),
        ("bye", "GOODBYE"),
        ("good bye", "GOODBYE"),
    )
    # Figure view window in normalized units (shoulder widths): x_min, x_max, y_min, y_max.
    figure_extent: tuple[float, float, float, float] = (-1.8, 1.8, -2.0, 2.8)
    figure_tick_ms: int = 33  # playback frame interval (~30 FPS)
    # Palette (deck colours).
    bg: str = "#0F2A31"
    panel: str = "#0B2227"
    text: str = "#EAF3F4"
    muted: str = "#8FB3B8"
    teal: str = "#0E7C86"
    coral: str = "#E4572E"
    word: str = "#7FD6C8"
```

And add the singleton after `PATHS = ModelPaths()`:

```python
UI = UIConfig()
```

Create `senseless/common/events.py`:

```python
"""Events sent from a mode worker process to the touchscreen GUI.

Small frozen dataclasses: they travel through a multiprocessing queue, so they
must pickle. Camera preview images are not events; they go on a separate
drop-oldest queue so a burst of frames can never evict a recognized word.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerReady:
    """The worker's devices and models are up; ``mode`` is "sign" or "speech"."""

    mode: str


@dataclass(frozen=True)
class WorkerError:
    """A failure the user should see (camera missing, model file missing, ...)."""

    message: str


@dataclass(frozen=True)
class SignStatus:
    """Per-frame sign pipeline status: segmenter state, capture progress 0..1, hands, FPS."""

    state: str
    progress: float
    hands: bool
    fps: float


@dataclass(frozen=True)
class SignResult:
    """One classified sign. ``word`` is None when rejected (low confidence or IDLE)."""

    word: str | None
    best: str
    confidence: float


@dataclass(frozen=True)
class SpeechText:
    """A Vosk result: a changing partial or a final line."""

    text: str
    is_final: bool
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_events.py`
Expected: `2 passed`. Then run the whole suite: `.venv\Scripts\python -m pytest -o addopts="" -q` (all pass), ruff, black.

- [ ] **Step 5: Commit** (when the user asks)

```bash
git add senseless/common/config.py senseless/common/events.py senseless/tests/test_events.py
git commit -m "Add UI config and worker-to-GUI events for the touchscreen app"
```

---

### Task 2: Pure UI state (sentence, transcript, text scale, speech→sign matching)

**Files:**
- Create: `senseless/ui/state.py`
- Test: `senseless/tests/test_ui_state.py`

**Interfaces:**
- Consumes: `UI.sentence_max_words`, `UI.transcript_max_lines`, `UI.text_scales`, `UI.default_text_scale`, `UI.sign_aliases`.
- Produces:
  - `SignSentence(max_words: int | None = None)`: `.add(word: str) -> None`, `.undo() -> str | None`, `.clear() -> None`, `.words: list[str]`, `.text: str` (words joined by one space).
  - `SpeechTranscript(max_lines: int | None = None)`: `.add(text: str, is_final: bool) -> None`, `.clear() -> None`, `.lines: list[str]`, `.partial: str`.
  - `TextScale(steps: tuple[float, ...] | None = None, index: int | None = None)`: `.factor: float`, `.bigger() -> float`, `.smaller() -> float`.
  - `words_to_sign(text: str, available: Collection[str], aliases=UI.sign_aliases) -> list[str]`.
  - `highlight_spans(text: str, available: Collection[str], aliases=UI.sign_aliases) -> list[tuple[int, int]]` (character spans in `text`).

- [ ] **Step 1: Write the failing test**

```python
# senseless/tests/test_ui_state.py
"""TDD specs for the touchscreen app's pure state (senseless/ui/state.py)."""

from senseless.ui.state import (
    SignSentence,
    SpeechTranscript,
    TextScale,
    highlight_spans,
    words_to_sign,
)

ALIASES = (("thank you", "THANKYOU"), ("thanks", "THANKYOU"), ("hi", "HELLO"))


def test_sentence_adds_undoes_and_clears() -> None:
    s = SignSentence(max_words=8)
    s.add("HELLO")
    s.add("THANKYOU")
    assert s.text == "HELLO THANKYOU"
    assert s.undo() == "THANKYOU"
    assert s.words == ["HELLO"]
    s.clear()
    assert s.text == ""
    assert s.undo() is None


def test_sentence_keeps_only_the_newest_words() -> None:
    s = SignSentence(max_words=3)
    for w in ("A", "B", "C", "D"):
        s.add(w)
    assert s.words == ["B", "C", "D"]


def test_transcript_partial_is_replaced_and_cleared_by_a_final() -> None:
    t = SpeechTranscript(max_lines=50)
    t.add("hel", False)
    t.add("hello th", False)
    assert t.partial == "hello th" and t.lines == []
    t.add("hello there", True)
    assert t.lines == ["hello there"] and t.partial == ""


def test_transcript_empty_final_only_clears_the_partial() -> None:
    t = SpeechTranscript(max_lines=50)
    t.add("uh", False)
    t.add("", True)
    assert t.lines == [] and t.partial == ""


def test_transcript_keeps_the_newest_lines_and_clears() -> None:
    t = SpeechTranscript(max_lines=2)
    for line in ("one", "two", "three"):
        t.add(line, True)
    assert t.lines == ["two", "three"]
    t.add("par", False)
    t.clear()
    assert t.lines == [] and t.partial == ""


def test_text_scale_steps_within_bounds() -> None:
    s = TextScale(steps=(0.8, 1.0, 1.25, 1.5), index=1)
    assert s.factor == 1.0
    assert s.bigger() == 1.25
    assert s.bigger() == 1.5
    assert s.bigger() == 1.5  # capped
    for _ in range(5):
        s.smaller()
    assert s.factor == 0.8  # floored


def test_words_to_sign_matches_words_and_multiword_aliases() -> None:
    avail = {"HELLO", "HOW", "THANKYOU", "YES"}
    assert words_to_sign("Hello there", avail, ALIASES) == ["HELLO"]
    assert words_to_sign("thank you so much", avail, ALIASES) == ["THANKYOU"]
    assert words_to_sign("hi, how are you?", avail, ALIASES) == ["HELLO", "HOW"]
    assert words_to_sign("yes no", avail, ALIASES) == ["YES"]  # only words the library has


def test_words_to_sign_does_not_guess_inflections() -> None:
    assert words_to_sign("cats and dogs", {"CAT", "DOG"}, ALIASES) == []


def test_highlight_spans_point_at_the_matched_text() -> None:
    text = "well thank you and hello"
    spans = highlight_spans(text, {"THANKYOU", "HELLO"}, ALIASES)
    assert [text[s:e] for s, e in spans] == ["thank you", "hello"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_ui_state.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'senseless.ui.state'`.

- [ ] **Step 3: Implement**

```python
# senseless/ui/state.py
"""Pure state for the touchscreen app: no Tk, no processes, fully unit-tested.

* ``SignSentence``   - recognized sign words (add / undo / clear, capped).
* ``SpeechTranscript`` - Vosk finals as lines plus the live partial line.
* ``TextScale``      - bounded text-size steps for the A- / A+ buttons.
* ``words_to_sign`` / ``highlight_spans`` - which spoken words the signing figure
  can show (vocabulary words and aliases such as "thank you" -> THANKYOU).
"""

from __future__ import annotations

import re
from collections.abc import Collection

from senseless.common.config import UI

_TOKEN = re.compile(r"[a-z']+")


class SignSentence:
    """The running sentence of recognized sign words."""

    def __init__(self, max_words: int | None = None) -> None:
        self.max_words = UI.sentence_max_words if max_words is None else max_words
        self.words: list[str] = []

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def add(self, word: str) -> None:
        self.words = (self.words + [word])[-self.max_words :]

    def undo(self) -> str | None:
        return self.words.pop() if self.words else None

    def clear(self) -> None:
        self.words = []


class SpeechTranscript:
    """Final lines plus the current partial (which the next result replaces)."""

    def __init__(self, max_lines: int | None = None) -> None:
        self.max_lines = UI.transcript_max_lines if max_lines is None else max_lines
        self.lines: list[str] = []
        self.partial = ""

    def add(self, text: str, is_final: bool) -> None:
        if not is_final:
            self.partial = text
            return
        self.partial = ""
        if text:
            self.lines = (self.lines + [text])[-self.max_lines :]

    def clear(self) -> None:
        self.lines = []
        self.partial = ""


class TextScale:
    """Text-size factor, stepped by A- / A+ within ``steps``."""

    def __init__(self, steps: tuple[float, ...] | None = None, index: int | None = None) -> None:
        self.steps = UI.text_scales if steps is None else steps
        self.index = UI.default_text_scale if index is None else index

    @property
    def factor(self) -> float:
        return self.steps[self.index]

    def bigger(self) -> float:
        self.index = min(self.index + 1, len(self.steps) - 1)
        return self.factor

    def smaller(self) -> float:
        self.index = max(self.index - 1, 0)
        return self.factor


def _match(
    text: str, available: Collection[str], aliases: tuple[tuple[str, str], ...]
) -> list[tuple[str, int, int]]:
    """(label, start, end) for each signable word or alias phrase, left to right."""
    tokens = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text.lower())]
    phrases = sorted(
        ((tuple(phrase.split()), label) for phrase, label in aliases), key=lambda p: -len(p[0])
    )
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(tokens):
        for words, label in phrases:
            n = len(words)
            if tuple(tok for tok, _, _ in tokens[i : i + n]) == words:
                if label in available:
                    out.append((label, tokens[i][1], tokens[i + n - 1][2]))
                i += n
                break
        else:
            word, start, end = tokens[i]
            label = word.replace("'", "").upper()
            if label in available:
                out.append((label, start, end))
            i += 1
    return out


def words_to_sign(
    text: str, available: Collection[str], aliases: tuple[tuple[str, str], ...] = UI.sign_aliases
) -> list[str]:
    """Vocabulary labels to sign for a spoken line (case/punctuation-insensitive)."""
    return [label for label, _, _ in _match(text, available, aliases)]


def highlight_spans(
    text: str, available: Collection[str], aliases: tuple[tuple[str, str], ...] = UI.sign_aliases
) -> list[tuple[int, int]]:
    """Character spans in ``text`` of the words that ``words_to_sign`` would sign."""
    return [(start, end) for _, start, end in _match(text, available, aliases)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_ui_state.py`
Expected: `9 passed`. Then the full suite, ruff, black.

- [ ] **Step 5: Commit** (when the user asks)

```bash
git add senseless/ui/state.py senseless/tests/test_ui_state.py
git commit -m "Add pure touchscreen-app state: sentence, transcript, text scale, speech-to-sign matching"
```

---

### Task 3: Sign worker

**Files:**
- Create: `senseless/sign/worker.py`
- Test: `senseless/tests/test_sign_worker.py`

**Interfaces:**
- Consumes: `capture.LatestFrameGrabber`, `capture.open_frame_source(prefer, source)`, `landmarks.create_backend(name, parallel)`, `landmarks.frame_landmarks_to_vector(raw)`, `landmarks.RawLandmarks`, `OnsetSegmenter().update(t, vec, hands) -> ndarray | None`, `.state`, `.progress(t)`, `SignClassifier().labels`, `.probabilities(window)`, `interpret(probs, labels, min_confidence, idle_label)`, events from Task 1, `UI` from Task 1.
- Produces:
  - `SignParts(source: Callable[[], FrameSource], backend: Callable[[], PerceptionBackend], classifier: Callable[[], Any])` dataclass; `default_sign_parts() -> SignParts`.
  - `make_preview(frame_rgb: np.ndarray, raw: RawLandmarks, size: tuple[int, int] = UI.preview_size) -> np.ndarray` (RGB uint8, shape `(size[1], size[0], 3)`).
  - `run_sign_worker(frames, events, stop, parts: SignParts | None = None) -> None`. `frames`/`events` expose `.put(item)`; `stop` exposes `.is_set()`. The source object must provide `read_stamped()` and be a context manager (as `LatestFrameGrabber` is).
  - Event order: `WorkerReady("sign")`, then per frame a `SignStatus` (and a preview on `frames`), a `SignResult` when a sign completes; on any failure a `WorkerError` and the function returns. If the camera stream ends: `WorkerError("Camera stream ended.")`.

- [ ] **Step 1: Write the failing test**

```python
# senseless/tests/test_sign_worker.py
"""Sign worker loop with fake parts: no camera, no MediaPipe, no TFLite.

The worker runs in-process here (queue.Queue backends, threading.Event); in the
app it runs in its own spawned process with multiprocessing queues.
"""

import queue
import threading

import numpy as np

from senseless.common.events import SignResult, SignStatus, WorkerError, WorkerReady
from senseless.common.queue import DropOldestQueue
from senseless.sign.capture import FrameSource
from senseless.sign.landmarks import PerceptionBackend, RawLandmarks
from senseless.sign.worker import SignParts, run_sign_worker


def _pose() -> np.ndarray:
    pose = np.full((33, 3), 0.5, dtype=np.float32)
    pose[11] = (0.40, 0.50, 0.0)  # shoulders set the normalization frame
    pose[12] = (0.60, 0.50, 0.0)
    return pose


class FakeCamera(FrameSource):
    """``n`` frames at 10 FPS (synthetic stamps), then the stream ends; or endless."""

    def __init__(self, n: int | None) -> None:
        self.n, self.i = n, 0

    def read_stamped(self):
        if self.n is not None and self.i >= self.n:
            return None
        stamp = self.i / 10.0
        self.i += 1
        return np.zeros((48, 64, 3), dtype=np.uint8), stamp

    def read(self):
        item = self.read_stamped()
        return None if item is None else item[0]

    def close(self) -> None:
        pass


class FakeBackend(PerceptionBackend):
    """Hands in view from 0.5 s on (frame index 5)."""

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, frame_rgb, timestamp_ms):
        hand = np.full((21, 3), 0.3, dtype=np.float32) if self.calls >= 5 else None
        self.calls += 1
        return RawLandmarks(left_hand=hand, right_hand=None, pose=_pose())

    def close(self) -> None:
        pass


class FakeClassifier:
    labels = ["HELLO", "IDLE"]

    def probabilities(self, window):
        assert window.shape == (45, 153)
        return np.array([0.9, 0.1], dtype=np.float32)


def _queues():
    return DropOldestQueue(2, backend=queue.Queue), DropOldestQueue(10_000, backend=queue.Queue)


def _drain(q) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_one_sign_gives_one_result_and_the_stream_end_is_reported() -> None:
    frames, events = _queues()
    parts = SignParts(lambda: FakeCamera(30), FakeBackend, FakeClassifier)
    run_sign_worker(frames, events, threading.Event(), parts)
    got = _drain(events)
    assert got[0] == WorkerReady("sign")
    results = [e for e in got if isinstance(e, SignResult)]
    assert results == [SignResult("HELLO", "HELLO", results[0].confidence)]
    assert any(isinstance(e, SignStatus) and e.hands for e in got)
    assert got[-1] == WorkerError("Camera stream ended.")
    preview = _drain(frames)[-1]
    assert preview.shape == (240, 320, 3) and preview.dtype == np.uint8


def test_stop_event_ends_the_loop() -> None:
    frames, events = _queues()
    stop = threading.Event()
    parts = SignParts(lambda: FakeCamera(None), FakeBackend, FakeClassifier)
    worker = threading.Thread(target=run_sign_worker, args=(frames, events, stop, parts))
    worker.start()
    stop.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()


def test_a_camera_that_fails_to_open_is_reported() -> None:
    def broken_camera():
        raise RuntimeError("Could not open OpenCV video source: 0")

    frames, events = _queues()
    run_sign_worker(frames, events, threading.Event(), SignParts(broken_camera, FakeBackend, FakeClassifier))
    (error,) = _drain(events)
    assert isinstance(error, WorkerError)
    assert error.message.startswith("Camera not found.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_sign_worker.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'senseless.sign.worker'`.

- [ ] **Step 3: Implement**

```python
# senseless/sign/worker.py
"""Sign-mode worker for the touchscreen app (runs in its own process).

Camera (newest frame only) -> perception (lite models in parallel on the Pi) ->
normalized vector -> onset segmenter -> classifier, one result per sign. Sends
a small RGB preview with the hand dots drawn on it to ``frames`` and status /
results / errors to ``events`` (see common/events.py). The parts are injectable
factories so tests can run the real loop with fakes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN, UI
from senseless.common.events import SignResult, SignStatus, WorkerError, WorkerReady
from senseless.sign import capture, landmarks
from senseless.sign.classifier import interpret
from senseless.sign.segmenter import OnsetSegmenter

_LEFT = (80, 210, 140)  # RGB, as in the GUI legend
_RIGHT = (242, 163, 58)
_POSE = (228, 87, 46)


@dataclass
class SignParts:
    """Factories for the pipeline parts; swapped for fakes in tests."""

    source: Callable[[], Any]  # -> object with read_stamped(), usable as a context manager
    backend: Callable[[], landmarks.PerceptionBackend]
    classifier: Callable[[], Any]  # -> object with .labels and .probabilities(window)


def default_sign_parts() -> SignParts:
    from senseless.sign.classifier import SignClassifier

    return SignParts(
        source=lambda: capture.LatestFrameGrabber(
            capture.open_frame_source(prefer=UI.camera, source=UI.camera_source)
        ),
        backend=lambda: landmarks.create_backend(
            UI.perception_backend, parallel=UI.parallel_perception
        ),
        classifier=SignClassifier,
    )


def make_preview(
    frame_rgb: np.ndarray, raw: landmarks.RawLandmarks, size: tuple[int, int] = UI.preview_size
) -> np.ndarray:
    """Downscale the frame to ``size`` (w, h) and draw the hand and pose dots (RGB)."""
    import cv2

    width, height = size
    img = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)

    def dots(points, color) -> None:
        if points is None:
            return
        for x, y, _z in np.asarray(points):
            cv2.circle(img, (int(x * width), int(y * height)), 3, color, -1)

    dots(raw.left_hand, _LEFT)
    dots(raw.right_hand, _RIGHT)
    if raw.pose is not None:
        dots(np.asarray(raw.pose)[list(ls.POSE_LANDMARK_INDICES)], _POSE)
    return np.ascontiguousarray(img, dtype=np.uint8)


def run_sign_worker(frames, events, stop, parts: SignParts | None = None) -> None:
    """Run the sign pipeline until ``stop`` is set or something fails."""
    try:
        parts = parts if parts is not None else default_sign_parts()
        classifier = parts.classifier()
        backend = parts.backend()
    except BaseException as exc:  # noqa: BLE001 -- shown to the user
        events.put(WorkerError(f"Sign model or MediaPipe files could not load: {exc}"))
        return
    try:
        source = parts.source()
    except BaseException as exc:  # noqa: BLE001
        backend.close()
        events.put(WorkerError(f"Camera not found. Check the USB cable, then tap Retry. ({exc})"))
        return

    segmenter = OnsetSegmenter()
    t0: float | None = None
    prev = time.perf_counter()
    fps = 0.0
    try:
        with backend, source:
            events.put(WorkerReady("sign"))
            while not stop.is_set():
                item = source.read_stamped()
                if item is None:
                    events.put(WorkerError("Camera stream ended."))
                    return
                frame, stamp = item
                if t0 is None:
                    t0 = stamp
                raw = backend.extract(frame, int((stamp - t0) * 1000))
                vec = landmarks.frame_landmarks_to_vector(raw)
                hands = raw.left_hand is not None or raw.right_hand is not None

                now = time.perf_counter()
                if now > prev:
                    inst = 1.0 / (now - prev)
                    fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
                prev = now

                window = segmenter.update(stamp, vec, hands)
                if window is not None:
                    word, best, conf = interpret(
                        classifier.probabilities(window),
                        classifier.labels,
                        SIGN.min_confidence,
                        SIGN.idle_label,
                    )
                    events.put(SignResult(word, best, conf))
                events.put(SignStatus(segmenter.state, segmenter.progress(stamp), hands, fps))
                frames.put(make_preview(frame, raw))
    except BaseException as exc:  # noqa: BLE001
        events.put(WorkerError(f"Sign engine error: {exc}"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_sign_worker.py`
Expected: `3 passed`. Then the full suite, ruff, black.

- [ ] **Step 5: Commit** (when the user asks)

```bash
git add senseless/sign/worker.py senseless/tests/test_sign_worker.py
git commit -m "Add sign-mode worker for the touchscreen app"
```

---

### Task 4: Speech worker

**Files:**
- Create: `senseless/asr/worker.py`
- Test: `senseless/tests/test_speech_worker.py`

**Interfaces:**
- Consumes: `audio.MicrophoneSource(device)`, `audio.find_input_device()`, `VoskTranscriber()` with `.accept(pcm) -> Transcript(text, is_final)`, `Transcript` from `senseless.asr.transcriber`, events from Task 1.
- Produces: `SpeechParts(source: Callable[[], AudioSource], transcriber: Callable[[], Any])`, `default_speech_parts()`, `run_speech_worker(events, stop, parts: SpeechParts | None = None) -> None`. Emits `WorkerReady("speech")`, then `SpeechText(partial, False)` only when the partial changes, `SpeechText(final, True)` for a final with text (or an empty final that clears a shown partial); on failure `WorkerError`; when the mic stream ends `WorkerError("Microphone stream ended.")`.

- [ ] **Step 1: Write the failing test**

```python
# senseless/tests/test_speech_worker.py
"""Speech worker loop with a fake microphone and a scripted fake Vosk."""

import queue
import threading

from senseless.asr.audio import AudioSource
from senseless.asr.transcriber import Transcript
from senseless.asr.worker import SpeechParts, run_speech_worker
from senseless.common.events import SpeechText, WorkerError, WorkerReady
from senseless.common.queue import DropOldestQueue


class FakeMic(AudioSource):
    def __init__(self, n_blocks: int | None) -> None:
        self.n, self.i = n_blocks, 0

    def read(self):
        if self.n is not None and self.i >= self.n:
            return None
        self.i += 1
        return b"\x00\x00" * 8000

    def close(self) -> None:
        pass


class ScriptedVosk:
    SCRIPT = [
        Transcript("hel", False),
        Transcript("hello", False),
        Transcript("hello", False),  # unchanged partial: not re-sent
        Transcript("hello there", True),
        Transcript("", True),  # nothing new
    ]

    def __init__(self) -> None:
        self.i = 0

    def accept(self, pcm: bytes) -> Transcript:
        result = self.SCRIPT[min(self.i, len(self.SCRIPT) - 1)]
        self.i += 1
        return result


def _drain(q) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_partials_are_deduplicated_and_finals_passed_on() -> None:
    events = DropOldestQueue(10_000, backend=queue.Queue)
    run_speech_worker(events, threading.Event(), SpeechParts(lambda: FakeMic(5), ScriptedVosk))
    assert _drain(events) == [
        WorkerReady("speech"),
        SpeechText("hel", False),
        SpeechText("hello", False),
        SpeechText("hello there", True),
        WorkerError("Microphone stream ended."),
    ]


def test_stop_event_ends_the_loop() -> None:
    events = DropOldestQueue(10_000, backend=queue.Queue)
    stop = threading.Event()
    parts = SpeechParts(lambda: FakeMic(None), ScriptedVosk)
    worker = threading.Thread(target=run_speech_worker, args=(events, stop, parts))
    worker.start()
    stop.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()


def test_a_missing_speech_model_is_reported() -> None:
    def no_model():
        raise FileNotFoundError("Vosk model not found at models/vosk-model-small-en-us-0.15")

    events = DropOldestQueue(10, backend=queue.Queue)
    run_speech_worker(events, threading.Event(), SpeechParts(lambda: FakeMic(1), no_model))
    (error,) = _drain(events)
    assert isinstance(error, WorkerError) and "Speech model" in error.message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_speech_worker.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'senseless.asr.worker'`.

- [ ] **Step 3: Implement**

```python
# senseless/asr/worker.py
"""Speech-mode worker for the touchscreen app (runs in its own process).

Microphone -> Vosk -> ``SpeechText`` events: a partial only when it changed,
and each final line. Errors become ``WorkerError`` events the GUI shows with a
Retry button. Parts are injectable factories so tests run the real loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from senseless.asr import audio
from senseless.common.events import SpeechText, WorkerError, WorkerReady


@dataclass
class SpeechParts:
    """Factories for the microphone and the recognizer; swapped for fakes in tests."""

    source: Callable[[], audio.AudioSource]
    transcriber: Callable[[], Any]  # -> object with .accept(pcm) -> Transcript


def default_speech_parts() -> SpeechParts:
    from senseless.asr.transcriber import VoskTranscriber

    return SpeechParts(
        source=lambda: audio.MicrophoneSource(device=audio.find_input_device()),
        transcriber=VoskTranscriber,
    )


def run_speech_worker(events, stop, parts: SpeechParts | None = None) -> None:
    """Run microphone -> Vosk until ``stop`` is set or something fails."""
    try:
        parts = parts if parts is not None else default_speech_parts()
        transcriber = parts.transcriber()
    except BaseException as exc:  # noqa: BLE001 -- shown to the user
        events.put(WorkerError(f"Speech model could not load: {exc}"))
        return
    try:
        mic = parts.source()
    except BaseException as exc:  # noqa: BLE001
        events.put(
            WorkerError(f"Microphone could not open. Check the USB cable, then tap Retry. ({exc})")
        )
        return

    shown_partial = ""
    try:
        with mic:
            events.put(WorkerReady("speech"))
            while not stop.is_set():
                block = mic.read()
                if block is None:
                    events.put(WorkerError("Microphone stream ended."))
                    return
                result = transcriber.accept(block)
                if result.is_final:
                    if result.text or shown_partial:
                        events.put(SpeechText(result.text, True))
                    shown_partial = ""
                elif result.text != shown_partial:
                    events.put(SpeechText(result.text, False))
                    shown_partial = result.text
    except BaseException as exc:  # noqa: BLE001
        events.put(WorkerError(f"Speech engine error: {exc}"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_speech_worker.py`
Expected: `3 passed`. Then the full suite, ruff, black.

- [ ] **Step 5: Commit** (when the user asks)

```bash
git add senseless/asr/worker.py senseless/tests/test_speech_worker.py
git commit -m "Add speech-mode worker for the touchscreen app"
```

---

### Task 5: Mode controller (one worker process at a time)

**Files:**
- Create: `senseless/ui/controller.py`
- Create: `senseless/tests/ui_fakes.py` (spawn-importable fake workers; not a `test_` file)
- Test: `senseless/tests/test_controller.py`

**Interfaces:**
- Consumes: `run_sign_worker` (Task 3), `run_speech_worker` (Task 4), `DropOldestQueue`, `UI`, `RUNTIME`.
- Produces:
  - `sign_entry(frames, events, stop)`, `speech_entry(frames, events, stop)`: module-level process targets.
  - `ModeController(targets: dict[str, Callable] | None = None, switch_timeout_s: float | None = None)` with:
    `.mode: str | None`, `.start(mode: str) -> None` (raises `RuntimeError` if a worker is running),
    `.request_stop() -> None`, `.poll_stopped() -> bool` (non-blocking; terminates after the timeout; True once no worker remains),
    `.has_exited() -> bool` (worker died without a stop request), `.exitcode() -> int | None`,
    `.drain_events() -> list`, `.latest_frame() -> np.ndarray | None`, `.shutdown(timeout: float | None = None) -> None` (blocking, for app exit).

- [ ] **Step 1: Write the fakes and the failing test**

```python
# senseless/tests/ui_fakes.py
"""Fake mode workers for the ModeController tests.

They live in their own module (not a ``test_`` file) because the controller
spawns them in child processes, which must import them by name.
"""

import time

import numpy as np

from senseless.common.events import SignStatus, WorkerReady


def ready_worker(frames, events, stop) -> None:
    events.put(WorkerReady("fake"))
    while not stop.is_set():
        frames.put(np.zeros((240, 320, 3), dtype=np.uint8))
        events.put(SignStatus("idle", 0.0, False, 10.0))
        time.sleep(0.05)


def stubborn_worker(frames, events, stop) -> None:
    events.put(WorkerReady("fake"))
    while True:  # ignores the stop request
        time.sleep(0.1)


def crashing_worker(frames, events, stop) -> None:
    raise SystemExit(3)
```

```python
# senseless/tests/test_controller.py
"""ModeController: one spawned worker at a time, stop with a timeout, crash detection."""

import time

import ui_fakes as fakes

from senseless.common.events import WorkerReady
from senseless.ui.controller import ModeController


def _wait(predicate, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_an_idle_controller_counts_as_stopped() -> None:
    assert ModeController({}).poll_stopped()


def test_a_worker_delivers_events_and_frames_then_stops_on_request() -> None:
    ctl = ModeController({"fake": fakes.ready_worker})
    ctl.start("fake")
    try:
        seen = []
        assert _wait(lambda: seen.extend(ctl.drain_events()) or WorkerReady("fake") in seen)
        assert _wait(lambda: ctl.latest_frame() is not None)
        ctl.request_stop()
        assert _wait(ctl.poll_stopped)
        assert ctl.mode is None
    finally:
        ctl.shutdown()


def test_a_worker_that_ignores_stop_is_terminated_after_the_timeout() -> None:
    ctl = ModeController({"fake": fakes.stubborn_worker}, switch_timeout_s=0.5)
    ctl.start("fake")
    try:
        seen = []
        assert _wait(lambda: seen.extend(ctl.drain_events()) or WorkerReady("fake") in seen)
        ctl.request_stop()
        assert _wait(ctl.poll_stopped)
    finally:
        ctl.shutdown()


def test_an_unexpected_exit_is_detected_with_its_code() -> None:
    ctl = ModeController({"fake": fakes.crashing_worker})
    ctl.start("fake")
    try:
        assert _wait(ctl.has_exited)
        assert ctl.exitcode() == 3
    finally:
        ctl.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_controller.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'senseless.ui.controller'`.

- [ ] **Step 3: Implement**

```python
# senseless/ui/controller.py
"""Runs one mode worker at a time for the touchscreen app.

Each worker is a spawned process with two queues back to the GUI: ``frames``
(preview images, max 2, newest wins) and ``events`` (status, words, speech
text, errors; max RUNTIME.transcript_queue_maxsize). Stopping is non-blocking
so the Tk loop never freezes: ``request_stop()`` then ``poll_stopped()`` on each
GUI tick; a worker still alive after the timeout is terminated.
"""

from __future__ import annotations

import multiprocessing
import queue
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from senseless.common.config import RUNTIME, UI
from senseless.common.queue import DropOldestQueue


def sign_entry(frames, events, stop) -> None:
    """Process target for Sign mode."""
    from senseless.sign.worker import run_sign_worker

    run_sign_worker(frames, events, stop)


def speech_entry(frames, events, stop) -> None:
    """Process target for Speech mode (no preview frames)."""
    from senseless.asr.worker import run_speech_worker

    run_speech_worker(events, stop)


class ModeController:
    """Starts, stops and watches the single worker process of the active mode."""

    def __init__(
        self,
        targets: dict[str, Callable] | None = None,
        switch_timeout_s: float | None = None,
    ) -> None:
        self._targets = targets if targets is not None else {"sign": sign_entry, "speech": speech_entry}
        self._timeout = UI.switch_timeout_s if switch_timeout_s is None else switch_timeout_s
        self._ctx = multiprocessing.get_context("spawn")
        self.mode: str | None = None
        self._proc: Any = None
        self._frames: DropOldestQueue | None = None
        self._events: DropOldestQueue | None = None
        self._stop: Any = None
        self._deadline: float | None = None  # set once a stop was requested

    def start(self, mode: str) -> None:
        if self._proc is not None:
            raise RuntimeError("a worker is already running; stop it first")
        self._frames = DropOldestQueue(UI.frames_queue_maxsize, backend=self._ctx.Queue)
        self._events = DropOldestQueue(RUNTIME.transcript_queue_maxsize, backend=self._ctx.Queue)
        self._stop = self._ctx.Event()
        # daemon=False: the sign worker starts its own model processes (ParallelBackend),
        # and daemonic processes are not allowed to have children.
        self._proc = self._ctx.Process(
            target=self._targets[mode],
            args=(self._frames, self._events, self._stop),
            name=f"senseless-{mode}",
            daemon=False,
        )
        self._proc.start()
        self.mode = mode
        self._deadline = None

    def request_stop(self) -> None:
        if self._proc is not None and self._deadline is None:
            self._stop.set()
            self._deadline = time.monotonic() + self._timeout

    def poll_stopped(self) -> bool:
        """True once no worker is running. Non-blocking; terminates after the timeout."""
        if self._proc is None:
            return True
        self.drain_events()  # keep the pipes flowing so the worker can exit
        self.latest_frame()
        if self._proc.is_alive():
            if self._deadline is None or time.monotonic() < self._deadline:
                return False
            self._proc.terminate()
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=2.0)
        self._cleanup()
        return True

    def has_exited(self) -> bool:
        """The worker died without being asked to stop (crash, killed, error exit)."""
        return self._proc is not None and self._deadline is None and not self._proc.is_alive()

    def exitcode(self) -> int | None:
        return None if self._proc is None else self._proc.exitcode

    def drain_events(self) -> list:
        out: list = []
        if self._events is None:
            return out
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def latest_frame(self) -> np.ndarray | None:
        frame = None
        if self._frames is None:
            return None
        while True:
            try:
                frame = self._frames.get_nowait()
            except queue.Empty:
                return frame

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop the worker, waiting up to ``timeout`` (blocking; for app exit)."""
        if self._proc is None:
            return
        self.request_stop()
        end = time.monotonic() + (self._timeout if timeout is None else timeout)
        while self._proc.is_alive() and time.monotonic() < end:
            self.drain_events()
            self.latest_frame()
            time.sleep(0.05)
        self._deadline = 0.0  # anything still running gets terminated now
        self.poll_stopped()

    def _cleanup(self) -> None:
        for q in (self._frames, self._events):
            if q is not None:
                q.close()
        self._proc = self._frames = self._events = self._stop = None
        self.mode = None
        self._deadline = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_controller.py`
Expected: `4 passed` (takes several seconds: each test spawns a process). Then the full suite, ruff, black.

- [ ] **Step 5: Commit** (when the user asks)

```bash
git add senseless/ui/controller.py senseless/tests/ui_fakes.py senseless/tests/test_controller.py
git commit -m "Add ModeController: one worker process at a time with non-blocking stop"
```

---

### Task 6: The Tk app (both modes, controls, banners, power dialog)

**Files:**
- Create: `senseless/ui/app.py`
- Create: `senseless/ui/__main__.py`
- Test: `senseless/tests/test_ui_app.py`

**Interfaces:**
- Consumes: `ModeController` API (Task 5), state classes (Task 2), events (Task 1), `UI`.
- Produces: `SenselessApp(root: tk.Tk, controller=None, start_workers: bool = True, fullscreen: bool = True, initial_mode: str = "sign")` with public `handle_event(event)`, `set_mode(mode)`, `retry()`, `undo()`, `clear()`, `bigger()`, `smaller()`, `show_error(message)`, `hide_error()`, `exit_app()`, and Tk variables `word_var`, `conf_var`, `hint_var`, `sentence_var`, `status_var`, `banner_var`, plus `transcript_text` (a `tk.Text`). Task 10 extends this class.

- [ ] **Step 1: Write the failing test**

```python
# senseless/tests/test_ui_app.py
"""Smoke test of the Tk app: hidden window, no worker processes, synthetic events."""

import pytest

tk = pytest.importorskip("tkinter")

from senseless.common.events import (  # noqa: E402
    SignResult,
    SignStatus,
    SpeechText,
    WorkerError,
    WorkerReady,
)
from senseless.ui.app import SenselessApp  # noqa: E402


class FakeController:
    """Stands in for ModeController: records calls, runs nothing."""

    def __init__(self) -> None:
        self.mode = None
        self.started: list[str] = []

    def start(self, mode):
        self.mode = mode
        self.started.append(mode)

    def request_stop(self):
        pass

    def poll_stopped(self):
        self.mode = None
        return True

    def has_exited(self):
        return False

    def exitcode(self):
        return None

    def drain_events(self):
        return []

    def latest_frame(self):
        return None

    def shutdown(self, timeout=None):
        pass


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    application = SenselessApp(root, controller=FakeController(), start_workers=False, fullscreen=False)
    yield application
    root.destroy()


def test_sign_results_build_the_sentence_and_undo_removes_a_word(app) -> None:
    app.handle_event(WorkerReady("sign"))
    app.handle_event(SignStatus("capturing", 0.5, True, 7.0))
    app.handle_event(SignResult("HELLO", "HELLO", 0.93))
    app.handle_event(SignResult("THANKYOU", "THANKYOU", 0.88))
    assert app.word_var.get() == "THANKYOU"
    assert app.sentence_var.get() == "HELLO THANKYOU"
    app.undo()
    assert app.sentence_var.get() == "HELLO"


def test_a_rejected_sign_shows_the_best_guess_and_is_not_added(app) -> None:
    app.handle_event(SignResult(None, "WANT", 0.41))
    assert app.word_var.get() == "?"
    assert "WANT" in app.conf_var.get()
    assert app.sentence_var.get() == ""


def test_speech_text_appears_in_the_transcript(app) -> None:
    app.handle_event(SpeechText("hello th", False))
    app.handle_event(SpeechText("hello there", True))
    app.handle_event(SpeechText("how are", False))
    text = app.transcript_text.get("1.0", "end")
    assert "hello there" in text and "how are" in text


def test_a_worker_error_shows_a_banner_and_switching_modes_restarts_a_worker(app) -> None:
    app.handle_event(WorkerError("Camera not found."))
    assert app.banner_var.get() == "Camera not found."
    app.set_mode("speech")
    app._tick()  # one GUI tick: the (fake) worker has stopped -> the new mode starts
    assert app.ctl.started[-1] == "speech"
    assert app.banner_var.get() == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_ui_app.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'senseless.ui.app'`.

- [ ] **Step 3: Implement `senseless/ui/app.py`**

```python
# senseless/ui/app.py
"""Senseless touchscreen app: Tkinter on the Pi's 7" 800x480 display.

Two modes, one at a time (each gets the whole CPU): Sign (camera preview, the
recognized word, a running sentence) and Speech (a scrolling live transcript).
The heavy work runs in a worker process managed by ``ModeController``; this
class only drains its queues every ``UI.poll_ms`` and draws.
"""

from __future__ import annotations

import subprocess
import tkinter as tk

import numpy as np

from senseless.common.config import UI
from senseless.common.events import SignResult, SignStatus, SpeechText, WorkerError, WorkerReady
from senseless.ui.state import SignSentence, SpeechTranscript, TextScale

FONT = "DejaVu Sans"
HINTS = {
    "idle": "Rest hands out of view, then sign",
    "pending": "Hand detected...",
    "capturing": "Capturing sign...",
    "holding": "Lower your hands for the next sign",
}
MODE_COLOR = {"sign": UI.teal, "speech": UI.coral}


def _button(parent, text, command, fg=UI.text, width=None) -> tk.Button:
    return tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        bg=UI.panel,
        fg=fg,
        activebackground=UI.teal,
        activeforeground=UI.text,
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground="#2E5A62",
        font=(FONT, 13),
        padx=10,
        pady=6,
    )


class SenselessApp:
    """The touchscreen GUI. See the module docstring."""

    def __init__(
        self,
        root: tk.Tk,
        controller=None,
        start_workers: bool = True,
        fullscreen: bool = True,
        initial_mode: str = "sign",
    ) -> None:
        if controller is None:
            from senseless.ui.controller import ModeController

            controller = ModeController()
        self.root = root
        self.ctl = controller
        self.mode = initial_mode
        self.sentence = SignSentence()
        self.transcript = SpeechTranscript()
        self.scale = TextScale()
        self._pending_mode: str | None = None
        self._error: str | None = None
        self._alive = True
        self._photo = None  # keep a reference, or Tk drops the image

        self.word_var = tk.StringVar(value="...")
        self.conf_var = tk.StringVar(value="")
        self.hint_var = tk.StringVar(value="Starting camera...")
        self.sentence_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.banner_var = tk.StringVar(value="")
        self.dialog_error_var = tk.StringVar(value="")

        root.title("Senseless")
        root.configure(bg=UI.bg)
        if fullscreen:
            root.attributes("-fullscreen", True)
        else:
            root.geometry(f"{UI.width}x{UI.height}")
        self._build()
        self._apply_fonts()
        self._show_view(self.mode)
        if start_workers:
            self.ctl.start(self.mode)
        root.after(UI.poll_ms, self._tick)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        bar = tk.Frame(self.root, bg=UI.panel, height=52)
        bar.pack(side="top", fill="x")
        self.mode_buttons = {
            "sign": _button(bar, "Sign", lambda: self.set_mode("sign"), width=7),
            "speech": _button(bar, "Speech", lambda: self.set_mode("speech"), width=7),
        }
        self.mode_buttons["sign"].pack(side="left", padx=(8, 0), pady=6)
        self.mode_buttons["speech"].pack(side="left", pady=6)
        _button(bar, "⏻", self.open_power_dialog, fg=UI.coral).pack(side="right", padx=8)
        _button(bar, "A+", self.bigger).pack(side="right", padx=2)
        _button(bar, "A−", self.smaller).pack(side="right", padx=2)
        self.status_label = tk.Label(bar, textvariable=self.status_var, bg=UI.panel, fg=UI.muted)
        self.status_label.pack(side="left", expand=True)

        self.content = tk.Frame(self.root, bg=UI.bg)
        self.content.pack(side="top", fill="both", expand=True)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)
        self.sign_view = self._build_sign_view(self.content)
        self.speech_view = self._build_speech_view(self.content)
        for view in (self.sign_view, self.speech_view):
            view.grid(row=0, column=0, sticky="nsew")

        self.banner = tk.Frame(self.content, bg=UI.panel, padx=16, pady=12)
        self.banner_label = tk.Label(
            self.banner, textvariable=self.banner_var, bg=UI.panel, fg=UI.text, wraplength=560
        )
        self.banner_label.pack(side="top", pady=(0, 8))
        _button(self.banner, "Retry", self.retry).pack(side="top")

        self.dialog = tk.Frame(self.root, bg="#16373F", padx=20, pady=16)
        tk.Label(
            self.dialog, text="Close Senseless?", bg="#16373F", fg=UI.text, font=(FONT, 16, "bold")
        ).pack(side="top", pady=(0, 10))
        tk.Label(
            self.dialog, textvariable=self.dialog_error_var, bg="#16373F", fg=UI.coral
        ).pack(side="top")
        row = tk.Frame(self.dialog, bg="#16373F")
        row.pack(side="top")
        _button(row, "Cancel", self.close_power_dialog).pack(side="left", padx=4)
        _button(row, "Exit app", self.exit_app).pack(side="left", padx=4)
        _button(row, "Power off", self.power_off, fg=UI.coral).pack(side="left", padx=4)

    def _build_sign_view(self, parent) -> tk.Frame:
        view = tk.Frame(parent, bg=UI.bg)
        left = tk.Frame(view, bg=UI.bg)
        left.grid(row=0, column=0, sticky="nw", padx=(10, 6), pady=10)
        blank = tk.PhotoImage(width=UI.preview_size[0], height=UI.preview_size[1])
        self._blank = blank
        self.preview = tk.Label(left, image=blank, bg="#33474C", bd=0)
        self.preview.pack(side="top")
        self.progress = tk.Canvas(
            left, width=UI.preview_size[0], height=8, bg="#23434A", highlightthickness=0
        )
        self.progress.pack(side="top", pady=(6, 2))
        self._progress_bar = self.progress.create_rectangle(0, 0, 0, 8, fill=UI.coral, width=0)
        self.hint_label = tk.Label(left, textvariable=self.hint_var, bg=UI.bg, fg=UI.muted)
        self.hint_label.pack(side="top", anchor="w")

        right = tk.Frame(view, bg=UI.bg)
        right.grid(row=0, column=1, sticky="nsew", pady=10)
        view.grid_columnconfigure(1, weight=1)
        view.grid_rowconfigure(0, weight=1)
        self.word_label = tk.Label(right, textvariable=self.word_var, bg=UI.bg, fg=UI.word)
        self.word_label.pack(side="top", expand=True)
        self.conf_label = tk.Label(right, textvariable=self.conf_var, bg=UI.bg, fg=UI.muted)
        self.conf_label.pack(side="top")

        bottom = tk.Frame(view, bg=UI.bg)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        bottom.grid_columnconfigure(0, weight=1)
        self.sentence_label = tk.Label(
            bottom,
            textvariable=self.sentence_var,
            bg=UI.panel,
            fg=UI.text,
            anchor="w",
            padx=10,
            pady=8,
        )
        self.sentence_label.grid(row=0, column=0, sticky="ew")
        _button(bottom, "Undo", self.undo).grid(row=0, column=1, padx=(6, 0))
        _button(bottom, "Clear", self.clear).grid(row=0, column=2, padx=(6, 0))
        return view

    def _build_speech_view(self, parent) -> tk.Frame:
        view = tk.Frame(parent, bg=UI.bg)
        self.transcript_text = tk.Text(
            view, bg=UI.bg, fg=UI.text, wrap="word", bd=0, highlightthickness=0, padx=12, pady=8
        )
        self.transcript_text.pack(side="top", fill="both", expand=True)
        self.transcript_text.configure(state="disabled")
        row = tk.Frame(view, bg=UI.bg)
        row.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        _button(row, "Clear", self.clear).pack(side="right")
        return view

    def _apply_fonts(self) -> None:
        f = self.scale.factor
        self.word_label.configure(font=(FONT, int(54 * f), "bold"))
        self.conf_label.configure(font=(FONT, int(13 * f)))
        self.hint_label.configure(font=(FONT, int(12 * f)))
        self.sentence_label.configure(font=(FONT, int(22 * f), "bold"))
        self.status_label.configure(font=(FONT, 12))
        self.banner_label.configure(font=(FONT, int(15 * f)))
        size = int(20 * f)
        self.transcript_text.tag_configure("old", foreground=UI.muted, font=(FONT, size))
        self.transcript_text.tag_configure("new", foreground=UI.text, font=(FONT, size))
        self.transcript_text.tag_configure(
            "partial", foreground=UI.muted, font=(FONT, size, "italic")
        )

    def _show_view(self, mode: str) -> None:
        (self.sign_view if mode == "sign" else self.speech_view).tkraise()
        for name, btn in self.mode_buttons.items():
            btn.configure(bg=MODE_COLOR[name] if name == mode else UI.panel)

    # ------------------------------------------------------------- the loop
    def _tick(self) -> None:
        if not self._alive:
            return
        if self._pending_mode is not None:
            if self.ctl.poll_stopped():
                self.mode, self._pending_mode = self._pending_mode, None
                self._show_view(self.mode)
                self.ctl.start(self.mode)
        else:
            for event in self.ctl.drain_events():
                self.handle_event(event)
            frame = self.ctl.latest_frame()
            if frame is not None and self.mode == "sign":
                self.show_frame(frame)
            if self._error is None and self.ctl.has_exited():
                code = self.ctl.exitcode()
                self.show_error(f"The {self.mode} engine stopped unexpectedly (exit code {code}).")
        self.root.after(UI.poll_ms, self._tick)

    def handle_event(self, event) -> None:
        if isinstance(event, WorkerReady):
            if event.mode == "sign":
                self.hint_var.set(HINTS["idle"])
            else:
                self.status_var.set("● listening")
        elif isinstance(event, SignStatus):
            self.hint_var.set(HINTS.get(event.state, ""))
            self._set_progress(event.progress)
            hands = "✓" if event.hands else "–"
            self.status_var.set(f"hands {hands}  ·  {event.fps:.0f} FPS")
        elif isinstance(event, SignResult):
            if event.word:
                self.sentence.add(event.word)
                self.word_var.set(event.word)
                self.conf_var.set(f"confidence {event.confidence:.2f}")
            else:
                self.word_var.set("?")
                self.conf_var.set(f"not recognized (best guess {event.best} {event.confidence:.2f})")
            self.sentence_var.set(self.sentence.text)
        elif isinstance(event, SpeechText):
            self.on_speech(event)
        elif isinstance(event, WorkerError):
            self.show_error(event.message)

    def on_speech(self, event: SpeechText) -> None:
        self.transcript.add(event.text, event.is_final)
        self._render_transcript()

    def _render_transcript(self) -> None:
        t = self.transcript_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        lines = self.transcript.lines
        for i, line in enumerate(lines):
            t.insert("end", line + "\n", "new" if i == len(lines) - 1 else "old")
        if self.transcript.partial:
            t.insert("end", self.transcript.partial, "partial")
        t.configure(state="disabled")
        t.see("end")

    def show_frame(self, frame: np.ndarray) -> None:
        from PIL import Image, ImageTk

        self._photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.preview.configure(image=self._photo)

    def _set_progress(self, fraction: float) -> None:
        self.progress.coords(self._progress_bar, 0, 0, int(UI.preview_size[0] * fraction), 8)

    # ---------------------------------------------------------- errors / modes
    def show_error(self, message: str) -> None:
        self._error = message
        self.banner_var.set(message)
        self.banner.place(relx=0.5, rely=0.5, anchor="center")
        self.banner.tkraise()

    def hide_error(self) -> None:
        self._error = None
        self.banner_var.set("")
        self.banner.place_forget()

    def set_mode(self, mode: str) -> None:
        if mode == self.mode and self._pending_mode is None and self._error is None:
            return
        self.hide_error()
        self._pending_mode = mode
        self.hint_var.set("Switching...")
        self.status_var.set("")
        for name, btn in self.mode_buttons.items():
            btn.configure(bg=MODE_COLOR[name] if name == mode else UI.panel)
        self.ctl.request_stop()

    def retry(self) -> None:
        self.set_mode(self.mode if self._pending_mode is None else self._pending_mode)

    # ---------------------------------------------------------------- controls
    def undo(self) -> None:
        self.sentence.undo()
        self.sentence_var.set(self.sentence.text)

    def clear(self) -> None:
        if self.mode == "sign":
            self.sentence.clear()
            self.sentence_var.set("")
            self.word_var.set("...")
            self.conf_var.set("")
        else:
            self.transcript.clear()
            self._render_transcript()

    def bigger(self) -> None:
        self.scale.bigger()
        self._apply_fonts()

    def smaller(self) -> None:
        self.scale.smaller()
        self._apply_fonts()

    # ------------------------------------------------------------------- power
    def open_power_dialog(self) -> None:
        self.dialog_error_var.set("")
        self.dialog.place(relx=0.5, rely=0.5, anchor="center")
        self.dialog.tkraise()

    def close_power_dialog(self) -> None:
        self.dialog.place_forget()

    def exit_app(self) -> None:
        self._alive = False
        self.ctl.shutdown()
        self.root.destroy()

    def power_off(self) -> None:
        self.ctl.shutdown()
        try:
            subprocess.run(
                ["sudo", "systemctl", "poweroff"], check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            self.dialog_error_var.set(f"Power off failed: {detail.strip()}")
            self._pending_mode = self.mode  # bring the worker back
```

- [ ] **Step 4: Create `senseless/ui/__main__.py`**

```python
"""Run the touchscreen app: ``python -m senseless.ui`` (full screen on the Pi).

``--windowed`` opens an 800x480 window instead (development on a PC).
"""

from __future__ import annotations

import argparse
import tkinter as tk

from senseless.ui.app import SenselessApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Senseless touchscreen app.")
    parser.add_argument("--windowed", action="store_true", help="800x480 window, not full screen.")
    parser.add_argument("--mode", choices=["sign", "speech"], default="sign")
    args = parser.parse_args()
    root = tk.Tk()
    SenselessApp(root, fullscreen=not args.windowed, initial_mode=args.mode)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_ui_app.py`
Expected: `4 passed` (or `4 skipped` on a machine without a display). Then the full suite, ruff, black.

- [ ] **Step 6: Manual check on the dev PC**

Run: `.venv\Scripts\python -m senseless.ui --windowed`
Expected: an 800×480 window. Sign mode: preview appears after a few seconds. On the PC's MediaPipe 0.10.35 the `lite` backend is unavailable, so a banner shows the MediaPipe error; that proves the error path. To see Sign mode work on the PC, temporarily run with `UI.perception_backend = "tasks"` in `config.py` (do not commit that change). Speech mode: transcript updates as you speak. Tap ⏻ → Cancel / Exit app.

- [ ] **Step 7: Commit** (when the user asks)

```bash
git add senseless/ui/app.py senseless/ui/__main__.py senseless/tests/test_ui_app.py
git commit -m "Add the Senseless touchscreen app (Tkinter): Sign and Speech modes"
```

---

### Task 7: Autostart on the Pi and documentation

**Files:**
- Create: `deploy/senseless-ui.sh`
- Create: `deploy/senseless.desktop`
- Modify: `PI_Instructions.md` (new section "17. Touchscreen app"), `ARCHITECTURE.md` (module map + test count), `README.md` (one line)

**Interfaces:**
- Consumes: `python -m senseless.ui` (Task 6).
- Produces: a launcher that logs to `~/senseless.log`; an XDG autostart entry.

- [ ] **Step 1: Create the launcher and the autostart entry**

```bash
#!/usr/bin/env bash
# deploy/senseless-ui.sh -- start the touchscreen app full screen, logging to ~/senseless.log
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python -m senseless.ui "$@" >> "$HOME/senseless.log" 2>&1
```

```ini
# deploy/senseless.desktop -- copy to ~/.config/autostart/
[Desktop Entry]
Type=Application
Name=Senseless
Comment=Sign and speech to text (touchscreen)
Exec=/home/admin/senseless/deploy/senseless-ui.sh
X-GNOME-Autostart-enabled=true
```

Make the launcher executable in git: `git update-index --chmod=+x deploy/senseless-ui.sh` (after `git add`).

- [ ] **Step 2: Add the Pi guide section** (append to `PI_Instructions.md` before "## Quick reference")

````markdown
## 17. Touchscreen app (7" display)

**One-time setup on the Pi:**
```bash
sudo apt install -y python3-tk
chmod +x ~/senseless/deploy/senseless-ui.sh
mkdir -p ~/.config/autostart && cp ~/senseless/deploy/senseless.desktop ~/.config/autostart/
sudo raspi-config nonint do_blanking 1   # keep the screen on
```
Build the sign library for Speech mode's signing figure (on the PC, then copy `models/`):
`python -m senseless.sign.library` → `models/sign_library.npz`.

**Run it by hand** (over SSH, on the Pi's screen): `DISPLAY=:0 ~/senseless/deploy/senseless-ui.sh`.
After a reboot it starts by itself, full screen. Log: `~/senseless.log`.

If it does not start at boot on your desktop session, add this line to
`~/.config/labwc/autostart` (Wayland/labwc) instead: `/home/admin/senseless/deploy/senseless-ui.sh &`.

**Acceptance checklist:**
1. The Pi boots straight into the app, full screen.
2. Sign mode: rest hands out of view, sign, lower hands → the word appears and the sentence grows.
3. Undo removes the last word; Clear empties the sentence; A−/A+ change the text size.
4. Switch to Speech: the transcript shows grey partials and white finished lines; switch back works.
5. Speech mode: saying "hello" or "thank you" makes the figure sign it (after the Stage 2 tasks).
6. Unplug the camera in Sign mode → a banner appears → plug it back → Retry recovers.
7. ⏻ → Cancel closes the dialog; Exit app closes the app; Power off shuts the Pi down.
````

- [ ] **Step 3: Update `ARCHITECTURE.md` and `README.md`**

In `ARCHITECTURE.md`, replace the `ui/` row of the module map with:

```markdown
| [ui/](senseless/ui) | Touchscreen app (Tkinter, 800×480): `app.py` (views, controls, dialogs), `controller.py` (one worker process per mode), `state.py` (pure sentence/transcript/text-size/speech-to-sign logic). Workers: [sign/worker.py](senseless/sign/worker.py), [asr/worker.py](senseless/asr/worker.py); events in [common/events.py](senseless/common/events.py). | ✅ |
```

and update the test-count cell to the new total printed by pytest. In `README.md`, add after the PI_Instructions sentence: `Run the touchscreen app with python -m senseless.ui (see PI_Instructions.md §17).`

- [ ] **Step 4: Verify**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q`, ruff, black. All green.

- [ ] **Step 5: Commit** (when the user asks)

```bash
git add deploy PI_Instructions.md ARCHITECTURE.md README.md
git update-index --chmod=+x deploy/senseless-ui.sh
git commit -m "Autostart the touchscreen app on the Pi; document setup and acceptance checklist"
```

---

# Stage 2: speech → sign figure

### Task 8: Sign library (one representative take per word)

**Files:**
- Create: `senseless/sign/library.py`
- Test: `senseless/tests/test_library.py`

**Interfaces:**
- Consumes: `dataset.list_labels(data_dir)`, `dataset.load_label(label, data_dir) -> (N, 45, 153)`, `dataset.save_window(window, label, data_dir)` (tests), `SIGN.idle_label`, `PATHS.sign_library`.
- Produces: `medoid_index(windows: np.ndarray) -> int`, `build_library(data_dir=DATA_DIR, exclude=(SIGN.idle_label,)) -> dict[str, np.ndarray]` (each `(45, 153)` float32), `save_library(library, path=PATHS.sign_library) -> Path`, `load_library(path=PATHS.sign_library) -> dict[str, np.ndarray]`, CLI `python -m senseless.sign.library [--data-dir DIR] [--out PATH]`.

- [ ] **Step 1: Write the failing test**

```python
# senseless/tests/test_library.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_library.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'senseless.sign.library'`.

- [ ] **Step 3: Implement**

```python
# senseless/sign/library.py
"""Sign library: one representative recorded take per word (speech -> sign figure).

Averaging a word's takes would blur the motion, so each word keeps its medoid:
the take with the smallest total distance to that word's other takes, i.e. the
most typical one. IDLE is not a sign and is left out. Saved as
``models/sign_library.npz`` ({label: (45, 153) float32}) and copied to the Pi
with the other models.

    python -m senseless.sign.library            # data/ -> models/sign_library.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from senseless.collect import dataset
from senseless.common.config import DATA_DIR, PATHS, SIGN


def medoid_index(windows: np.ndarray) -> int:
    """Index of the take with the smallest summed Euclidean distance to the others."""
    flat = windows.reshape(len(windows), -1).astype(np.float64)
    sq = (flat**2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T, 0.0)
    return int(np.argmin(np.sqrt(d2).sum(axis=1)))


def build_library(
    data_dir: Path | str = DATA_DIR, exclude: tuple[str, ...] = (SIGN.idle_label,)
) -> dict[str, np.ndarray]:
    library: dict[str, np.ndarray] = {}
    for label in dataset.list_labels(data_dir):
        if label in exclude:
            continue
        windows = dataset.load_label(label, data_dir)
        if len(windows) == 0:
            continue
        library[label] = windows[medoid_index(windows)].astype(np.float32)
    return library


def save_library(library: dict[str, np.ndarray], path: Path | str = PATHS.sign_library) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **library)
    return path


def load_library(path: Path | str = PATHS.sign_library) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the sign library for the Speech figure.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out", default=str(PATHS.sign_library))
    args = parser.parse_args()
    library = build_library(args.data_dir)
    path = save_library(library, args.out)
    print(f"saved {path} with {len(library)} words: {', '.join(sorted(library))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then build the real library**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_library.py`
Expected: `3 passed`.
Run: `.venv\Scripts\python -m senseless.sign.library`
Expected: `saved ...models\sign_library.npz with 16 words: BOOK, CAT, ...` (IDLE absent). Full suite, ruff, black.

- [ ] **Step 5: Commit** (when the user asks)

```bash
git add senseless/sign/library.py senseless/tests/test_library.py
git commit -m "Add the sign library builder: one representative take per word"
```

---

### Task 9: Figure geometry and playback timing

**Files:**
- Create: `senseless/ui/figure.py`
- Test: `senseless/tests/test_figure.py`

**Interfaces:**
- Consumes: `landmark_schema` offsets, `UI.figure_extent`, `SIGN.window_length`, `SIGN.reference_fps`.
- Produces:
  - `Segment(x0: float, y0: float, x1: float, y1: float, kind: str)` with kind in {"body", "left", "right"}; `Dot(x: float, y: float, r: float, kind: str)` with kind in {"head", "joint"}.
  - `figure_geometry(vec: np.ndarray, box: tuple[float, float, float, float], extent=UI.figure_extent) -> tuple[list[Segment], list[Dot]]` (box = x, y, width, height on the canvas).
  - `frame_index(elapsed_s: float, n_frames: int = SIGN.window_length, fps: float = SIGN.reference_fps) -> int | None` (None once the take is over).
  - `rest_frame(library: dict[str, np.ndarray]) -> np.ndarray | None` (first frame of the first word, hands zeroed).

- [ ] **Step 1: Write the failing test**

```python
# senseless/tests/test_figure.py
"""Stick-figure geometry for the Speech-mode signing figure (senseless/ui/figure.py)."""

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.ui.figure import frame_index, figure_geometry, rest_frame

BOX = (10.0, 20.0, 200.0, 300.0)  # x, y, width, height
EXTENT = (-2.0, 2.0, -2.0, 2.0)


def _frame(with_hands: bool) -> np.ndarray:
    v = np.zeros(ls.FEATURE_DIM, dtype=np.float32)
    pose = np.array(
        [[0, -0.8, 0], [-0.5, 0, 0], [0.5, 0, 0], [-0.7, 0.8, 0], [0.7, 0.8, 0],
         [-0.8, 1.5, 0], [0.8, 1.5, 0], [-0.3, 2.0, 0], [0.3, 2.0, 0]],
        dtype=np.float32,
    )
    v[ls.POSE_START : ls.POSE_END] = pose.reshape(-1)
    if with_hands:
        v[ls.LEFT_HAND_START : ls.LEFT_HAND_END] = np.tile([-0.8, 1.5, 0], 21) + 0.05
    return v


def _inside(x: float, y: float) -> bool:
    bx, by, bw, bh = BOX
    return bx - 1e-6 <= x <= bx + bw + 1e-6 and by - 1e-6 <= y <= by + bh + 1e-6


def test_an_empty_frame_draws_nothing() -> None:
    assert figure_geometry(np.zeros(ls.FEATURE_DIM, np.float32), BOX, EXTENT) == ([], [])


def test_body_only_frame_has_body_lines_a_head_and_no_hand_lines() -> None:
    segments, dots = figure_geometry(_frame(with_hands=False), BOX, EXTENT)
    assert segments and all(s.kind == "body" for s in segments)
    assert [d.kind for d in dots].count("head") == 1


def test_a_present_hand_adds_its_lines() -> None:
    segments, _ = figure_geometry(_frame(with_hands=True), BOX, EXTENT)
    assert any(s.kind == "left" for s in segments)
    assert not any(s.kind == "right" for s in segments)


def test_the_figure_fits_inside_the_box_and_the_origin_is_centred() -> None:
    segments, dots = figure_geometry(_frame(with_hands=True), BOX, EXTENT)
    assert all(_inside(s.x0, s.y0) and _inside(s.x1, s.y1) for s in segments)
    head = next(d for d in dots if d.kind == "head")
    bx, by, bw, bh = BOX
    assert np.isclose(head.x, bx + bw / 2)  # nose at x' = 0 -> horizontal centre


def test_frame_index_walks_the_take_then_finishes() -> None:
    assert frame_index(0.0, 45, 30.0) == 0
    assert frame_index(44 / 30.0, 45, 30.0) == 44
    assert frame_index(45 / 30.0, 45, 30.0) is None


def test_rest_frame_is_a_body_without_hands() -> None:
    lib = {"HELLO": np.stack([_frame(with_hands=True)] * 45)}
    rest = rest_frame(lib)
    assert np.all(rest[ls.LEFT_HAND_START : ls.RIGHT_HAND_END] == 0.0)
    assert np.any(rest[ls.POSE_START : ls.POSE_END] != 0.0)
    assert rest_frame({}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_figure.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'senseless.ui.figure'`.

- [ ] **Step 3: Implement**

```python
# senseless/ui/figure.py
"""Stick-figure geometry for the Speech-mode signing figure (pure, no Tk).

One 153-value frame (normalized: shoulder-midpoint origin, shoulder-width units,
y down) becomes line segments and dots in canvas coordinates. A fixed view
window (``UI.figure_extent``) is fitted into the box with one uniform scale and
centred, so the figure doesn't jump in size between frames or words. Hands that
are absent (all-zero blocks) are not drawn; the face is not in the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import SIGN, UI

# Pose subset order: nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip.
POSE_EDGES = ((1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8))
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)  # fmt: skip
HEAD_RADIUS = 0.22  # shoulder widths


@dataclass(frozen=True)
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str  # "body" | "left" | "right"


@dataclass(frozen=True)
class Dot:
    x: float
    y: float
    r: float
    kind: str  # "head" | "joint"


def figure_geometry(
    vec: np.ndarray,
    box: tuple[float, float, float, float],
    extent: tuple[float, float, float, float] = UI.figure_extent,
) -> tuple[list[Segment], list[Dot]]:
    pose = np.asarray(vec[ls.POSE_START : ls.POSE_END], dtype=np.float64).reshape(-1, 3)
    if not np.any(pose):
        return [], []
    bx, by, bw, bh = box
    x_min, x_max, y_min, y_max = extent
    scale = min(bw / (x_max - x_min), bh / (y_max - y_min))
    ox = bx + bw / 2 - scale * (x_min + x_max) / 2
    oy = by + bh / 2 - scale * (y_min + y_max) / 2

    def to_canvas(p: np.ndarray) -> tuple[float, float]:
        x = float(np.clip(p[0], x_min, x_max))
        y = float(np.clip(p[1], y_min, y_max))
        return ox + scale * x, oy + scale * y

    segments = [Segment(*to_canvas(pose[a]), *to_canvas(pose[b]), "body") for a, b in POSE_EDGES]
    dots = [Dot(*to_canvas(pose[0]), HEAD_RADIUS * scale, "head")]
    dots += [Dot(*to_canvas(pose[i]), 3.0, "joint") for i in (1, 2, 3, 4)]
    for start, end, kind in (
        (ls.LEFT_HAND_START, ls.LEFT_HAND_END, "left"),
        (ls.RIGHT_HAND_START, ls.RIGHT_HAND_END, "right"),
    ):
        hand = np.asarray(vec[start:end], dtype=np.float64).reshape(-1, 3)
        if np.any(hand):
            segments += [
                Segment(*to_canvas(hand[a]), *to_canvas(hand[b]), kind) for a, b in HAND_EDGES
            ]
    return segments, dots


def frame_index(
    elapsed_s: float, n_frames: int = SIGN.window_length, fps: float = SIGN.reference_fps
) -> int | None:
    """Frame to show ``elapsed_s`` into a take, or None once the take is over."""
    index = int(elapsed_s * fps + 1e-9)
    return index if index < n_frames else None


def rest_frame(library: dict[str, np.ndarray]) -> np.ndarray | None:
    """A neutral pose for the idle figure: the first word's first frame, hands removed."""
    if not library:
        return None
    frame = np.array(library[sorted(library)[0]][0], dtype=np.float32, copy=True)
    frame[ls.LEFT_HAND_START : ls.RIGHT_HAND_END] = 0.0
    return frame
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_figure.py`
Expected: `6 passed`. Full suite, ruff, black.

- [ ] **Step 5: Commit** (when the user asks)

```bash
git add senseless/ui/figure.py senseless/tests/test_figure.py
git commit -m "Add stick-figure geometry and playback timing for the signing figure"
```

---

### Task 10: Figure playback and highlighting in the app

**Files:**
- Modify: `senseless/ui/app.py`
- Modify: `senseless/tests/test_ui_app.py`

**Interfaces:**
- Consumes: `load_library` (Task 8), `figure_geometry`, `frame_index`, `rest_frame` (Task 9), `words_to_sign`, `highlight_spans` (Task 2), `UI.figure_tick_ms`.
- Produces: `SenselessApp(..., library: dict[str, np.ndarray] | None = None, load_library_file: bool = True)`; attributes `sign_queue: collections.deque[str]` (bounded, `maxlen=UI.sign_queue_max`, drop-oldest), `figure_caption: tk.StringVar`; transcript tag `"signed"`.

- [ ] **Step 1: Add the failing tests** (append to `senseless/tests/test_ui_app.py`)

```python
import numpy as np  # noqa: E402

from senseless.common import landmark_schema as ls  # noqa: E402


def _library() -> dict:
    frame = np.zeros(ls.FEATURE_DIM, dtype=np.float32)
    frame[ls.POSE_START : ls.POSE_END] = 0.3
    frame[ls.LEFT_HAND_START : ls.LEFT_HAND_END] = 0.5
    return {"HELLO": np.stack([frame] * 45), "THANKYOU": np.stack([frame] * 45)}


@pytest.fixture
def speech_app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    application = SenselessApp(
        root,
        controller=FakeController(),
        start_workers=False,
        fullscreen=False,
        initial_mode="speech",
        library=_library(),
    )
    yield application
    root.destroy()


def test_a_final_line_queues_its_vocabulary_words_for_the_figure(speech_app) -> None:
    speech_app.handle_event(SpeechText("hello and thank you", True))
    queued = list(speech_app.sign_queue) + [speech_app._playing]
    assert "HELLO" in queued and "THANKYOU" in queued


def test_partials_do_not_trigger_signing(speech_app) -> None:
    speech_app.handle_event(SpeechText("hello", False))
    assert not speech_app.sign_queue and speech_app._playing is None


def test_signed_words_are_highlighted(speech_app) -> None:
    speech_app.handle_event(SpeechText("well hello there", True))
    ranges = speech_app.transcript_text.tag_ranges("signed")
    assert ranges
    assert speech_app.transcript_text.get(ranges[0], ranges[1]) == "hello"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_ui_app.py`
Expected: the 3 new tests FAIL (`TypeError: __init__() got an unexpected keyword argument 'library'`).

- [ ] **Step 3: Implement the changes in `senseless/ui/app.py`**

Add imports at the top:

```python
import time
from collections import deque

from senseless.ui.figure import figure_geometry, frame_index, rest_frame
from senseless.ui.state import highlight_spans, words_to_sign
```

Change the constructor signature and load the library (add after `self._photo = None`):

```python
    def __init__(
        self,
        root: tk.Tk,
        controller=None,
        start_workers: bool = True,
        fullscreen: bool = True,
        initial_mode: str = "sign",
        library: dict[str, np.ndarray] | None = None,
        load_library_file: bool = True,
    ) -> None:
```

```python
        if library is None and load_library_file:
            try:
                from senseless.sign.library import load_library

                library = load_library()
            except (OSError, ValueError):
                library = None
        self.library: dict[str, np.ndarray] = library or {}
        self._rest = rest_frame(self.library)
        self.sign_queue: deque[str] = deque(maxlen=UI.sign_queue_max)  # drop-oldest
        self._playing: str | None = None
        self._play_t0 = 0.0
        self._rest_drawn = False
        self.figure_caption = tk.StringVar(value="")
```

Also add `root.after(UI.figure_tick_ms, self._play_tick)` right after `root.after(UI.poll_ms, self._tick)`.

In `_build_speech_view`, after `self.transcript_text.configure(state="disabled")`, add the inset:

```python
        self.figure_panel = tk.Frame(view, bg=UI.panel)
        self.figure_canvas = tk.Canvas(
            self.figure_panel, width=220, height=210, bg=UI.panel, highlightthickness=0
        )
        self.figure_canvas.pack(side="top")
        tk.Label(
            self.figure_panel,
            textvariable=self.figure_caption,
            bg=UI.panel,
            fg=UI.word,
            font=(FONT, 13, "bold"),
        ).pack(side="top", pady=(0, 6))
        if self.library:
            self.figure_panel.place(relx=1.0, x=-10, y=10, anchor="ne")
        else:
            tk.Label(
                view,
                text="No sign library: run python -m senseless.sign.library",
                bg=UI.bg,
                fg=UI.muted,
            ).place(relx=1.0, x=-10, y=10, anchor="ne")
```

Note: `self.figure_caption` and `self.library` must exist before `_build()` runs, so place the library-loading block **before** `self._build()` in `__init__`.

In `_apply_fonts`, add the highlight tag:

```python
        self.transcript_text.tag_configure(
            "signed", foreground=UI.word, font=(FONT, size, "bold")
        )
```

Replace `on_speech` and `_render_transcript` with:

```python
    def on_speech(self, event: SpeechText) -> None:
        self.transcript.add(event.text, event.is_final)
        if event.is_final and event.text and self.library:
            self.sign_queue.extend(words_to_sign(event.text, self.library.keys()))
        self._render_transcript()

    def _render_transcript(self) -> None:
        t = self.transcript_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        lines = self.transcript.lines
        for i, line in enumerate(lines):
            start = t.index("end-1c")
            t.insert("end", line + "\n", "new" if i == len(lines) - 1 else "old")
            for s, e in highlight_spans(line, self.library.keys()):
                t.tag_add("signed", f"{start}+{s}c", f"{start}+{e}c")
        if self.transcript.partial:
            t.insert("end", self.transcript.partial, "partial")
        t.configure(state="disabled")
        t.see("end")
```

Add the playback loop and drawing:

```python
    # ------------------------------------------------------------ the figure
    def _play_tick(self) -> None:
        if not self._alive:
            return
        now = time.perf_counter()
        if self._playing is None and self.sign_queue:
            self._playing = self.sign_queue.popleft()
            self._play_t0 = now
            self.figure_caption.set(self._playing)
        if self._playing is not None:
            index = frame_index(now - self._play_t0)
            if index is None:
                self._playing = None
                self.figure_caption.set("")
                self._rest_drawn = False
            else:
                self._draw_figure(self.library[self._playing][index])
        if self._playing is None and not self._rest_drawn and self._rest is not None:
            self._draw_figure(self._rest)
            self._rest_drawn = True
        self.root.after(UI.figure_tick_ms, self._play_tick)

    def _draw_figure(self, frame: np.ndarray) -> None:
        c = self.figure_canvas
        c.delete("all")
        colors = {"body": "#CFE6E8", "left": "#50D28C", "right": "#F2A33A"}
        widths = {"body": 3, "left": 2, "right": 2}
        segments, dots = figure_geometry(frame, (8, 8, 204, 194))
        for s in segments:
            c.create_line(s.x0, s.y0, s.x1, s.y1, fill=colors[s.kind], width=widths[s.kind])
        for d in dots:
            if d.kind == "head":
                c.create_oval(d.x - d.r, d.y - d.r, d.x + d.r, d.y + d.r, outline="#CFE6E8", width=2)
            else:
                c.create_oval(d.x - d.r, d.y - d.r, d.x + d.r, d.y + d.r, fill=UI.coral, width=0)
```

(`on_speech` queues words; `_play_tick` pops the first one on its next tick, which is why the test checks `sign_queue` together with `_playing`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest -o addopts="" -q senseless/tests/test_ui_app.py`
Expected: `7 passed` (or skipped without a display). Full suite, ruff, black.

- [ ] **Step 5: Manual check on the dev PC**

Run: `.venv\Scripts\python -m senseless.ui --windowed --mode speech`
Say "hello" and pause: the line appears with "hello" highlighted and the figure in the top-right inset plays HELLO for ~1.5 s, then returns to rest.

- [ ] **Step 6: Commit** (when the user asks)

```bash
git add senseless/ui/app.py senseless/tests/test_ui_app.py
git commit -m "Speech mode: sign spoken vocabulary words with the stick figure"
```

---

## After all tasks

- Copy `models/sign_library.npz` to the Pi with the other models: `scp models\sign_library.npz admin@senseless.local:~/senseless/models/`.
- On the Pi: `git pull`, the one-time setup from PI_Instructions §17, reboot, then run the acceptance checklist.
