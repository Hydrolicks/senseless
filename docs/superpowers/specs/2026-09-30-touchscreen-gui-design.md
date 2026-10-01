# Senseless touchscreen app: design

Date: 2026-09-30. Status: approved in brainstorming, awaiting spec review.

## Goal

A simple, full-screen Tkinter app for the Raspberry Pi 4 on a 7" 800×480 touch
screen, replacing the separate `sign.demo` / `asr.mic_test` tools for live use.

- **Sign mode:** camera → onset-mode sign recognition → big word + running sentence.
- **Speech mode:** microphone → Vosk live transcript, plus a stick figure that signs
  spoken words that are in the vocabulary (speech → sign, a new direction).
- One mode runs at a time (each gets the whole CPU), switched by a tap.
- Boots straight into the app, full screen. Hardware: the Logitech USB webcam for
  both camera and microphone.

Out of scope: running both channels at once, realistic avatars or facial
expressions, signing words not in the recorded vocabulary, persisting settings.

## Screens

### Top bar (both modes)

`[ Sign | Speech ]` segmented switch (active tab teal in Sign, coral in Speech) ·
status in the centre (hands ✓ · FPS, or ● listening) · `A−` `A+` · `⏻`.

### Sign mode (layout A)

- Left (~42% width): 320×240 camera preview with hand dots drawn on it; below it a
  capture progress bar and a hint ("rest hands out of view, then sign" / "capturing
  sign" / "lower your hands for the next sign").
- Right: the latest word, large, with its confidence. A rejected sign shows
  "?" with the best guess and is not added to the sentence.
- Bottom: the running sentence (last 8 words) with `Undo` and `Clear` buttons.

### Speech mode (scrolling transcript, figure inset)

- Full-width transcript, newest line at the bottom: finished lines in white, older
  lines slightly dimmed, the in-progress partial line in grey italics. Words that
  were signed are highlighted. Keeps the last 50 lines. `Clear` button.
- Top-right inset: the signing stick figure (hands, arms, head position; no face)
  with the word being signed underneath. Rests in a neutral pose when idle.

### Power dialog

`⏻` opens "Close Senseless?" with `Cancel` · `Exit app` · `Power off`.

## Architecture

The GUI is the main process. It runs **one worker process** for the active mode
(spawned, as in `sign/parallel.py`). Switching modes stops one worker and starts
the other. Each mode keeps its own text across switches.

Two queues from the worker to the GUI, deliberately separate:

- `frames` (`DropOldestQueue`, max 2): preview images. Dropping stale ones is fine.
- `events` (`DropOldestQueue`, max `RUNTIME.transcript_queue_maxsize` = 32): words,
  status, speech text, errors. These must not be lost; the GUI drains the queue
  every 40 ms, so it never fills. Sharing one drop-oldest queue with previews could
  evict a recognized word.

### New modules

| Module | Responsibility |
|---|---|
| `common/events.py` | Picklable event dataclasses: `SignStatus(state, progress, hands, fps)`, `SignResult(word, best, confidence)`, `SpeechText(text, is_final)`, `WorkerError(message)`. |
| `sign/worker.py` | `run_sign_worker(frames, events, stop, parts=None)`: `LatestFrameGrabber` → lite `ParallelBackend` → `OnsetSegmenter` → classifier + `interpret`. Draws hand dots on a 320×240 preview in the worker. Parts are injectable factories for tests. |
| `asr/worker.py` | `run_speech_worker(events, stop, parts=None)`: `MicrophoneSource` → `VoskTranscriber` → `SpeechText`. Injectable parts. |
| `sign/library.py` | `build_library(data_dir) -> dict[label, (45, 153)]`: per label the medoid take (smallest mean distance to the label's other takes); IDLE excluded. `save_library` / `load_library` (`models/sign_library.npz`). CLI: `python -m senseless.sign.library`. |
| `ui/state.py` | Pure: `SignSentence` (add, undo, clear, cap 8), `SpeechTranscript` (partial replaced by the next partial or cleared by a final; finals appended; cap 50), `TextScale` (4 steps, bounded), `words_to_sign(text, available, aliases)` (case/punctuation-insensitive, multi-word aliases, only words the library has). |
| `ui/figure.py` | Pure: `figure_geometry(vec, box) -> (lines, dots)` from one 153-value frame to canvas coordinates (fits and centres, keeps aspect; absent hands draw nothing; pose-only resting frame); `frame_index(elapsed_s)` for 45 frames over the window span. |
| `ui/app.py` | Tkinter app: full screen 800×480, top bar, the two views, power dialog, 40 ms polling loop, worker lifecycle (`ModeController`), figure playback queue. |
| `ui/__main__.py` | `python -m senseless.ui` (`--windowed` for development on a PC). |
| `config.py` → `UIConfig` | Screen size, fonts and text-size steps, colours (deck palette: ink `0E3A44`, teal `0E7C86`, coral `E4572E`), preview size, poll interval, library path, aliases (e.g. "thank you" → THANKYOU, "hi" → HELLO, "bye" → GOODBYE). |
| `deploy/` | Desktop-autostart entry + launcher script (logs to `~/senseless.log`). Pi guide section. |

Existing tools (`sign.demo`, `asr.mic_test`, `collect`) stay unchanged.

## Data flow

**Startup:** open full screen, load the sign library, start Sign mode; the preview
shows "Starting camera…" until the first frame (MediaPipe loads for a few seconds).

**Sign:** each frame the worker puts a preview on `frames` and a `SignStatus` on
`events`; when the segmenter emits a window it classifies once and puts a
`SignResult`. The GUI shows the newest preview, updates progress and hint, and on an
accepted result shows the word and appends it to the sentence.

**Speech:** the worker feeds 0.5 s audio blocks to Vosk and puts `SpeechText`. The
GUI replaces the partial line on a partial; on a final it appends a white line, runs
`words_to_sign`, highlights the matches and queues them for the figure.

**Figure playback:** each queued word plays its library take (45 frames at
`SIGN.reference_fps`, ~1.5 s) on a canvas via Tk `after` ticks, one word after
another; an empty queue returns to the resting pose. Triggered by **finals only**,
since partials still change.

**Mode switch:** tap → "Switching…" → set the worker's stop event → wait up to 5 s
without blocking the UI (polling) → terminate if still alive → drain both queues →
start the other worker. The worker's exit releases the camera or microphone.

**Controls:** Undo removes the last signed word; Clear empties the active view;
A−/A+ step text sizes; Exit stops the worker and closes; Power off also runs
`sudo systemctl poweroff`.

## Error handling

- Workers catch everything at the top level and send `WorkerError(message)` before
  exiting (camera not found, mic won't open, model files missing). The GUI shows a
  banner in that mode's view with a **Retry** button that restarts the worker.
- A worker that dies without a message (killed, native crash) is detected by the
  GUI's liveness check and gets the same banner with its exit code.
- No sign library: Speech mode works and hides the figure with a short note on how
  to build it. A spoken word the library lacks is not highlighted or signed.
- A worker that won't stop within 5 s is terminated.
- A failed power-off shows its error in the dialog.
- The launcher logs to `~/senseless.log`; the screen shows only short messages.

## Testing

Test-first (pure):

- `ui/state.py`: sentence add/undo/clear/cap; transcript partial/final/cap; text
  scale bounds; `words_to_sign` (case, punctuation, multi-word aliases, unknown
  words, availability filter).
- `ui/figure.py`: absent hands → no hand lines; pose-only frame; fit/centre/aspect
  in a box; `frame_index` over 45 frames and the span.
- `sign/library.py`: medoid is the most central take (an outlier is never picked);
  IDLE excluded; save/load round-trip.
- `common/events.py`: events pickle round-trip.

With fakes: each worker runs its real loop with injected fake parts; it emits the
expected events, stops on the stop event, and turns a failing part into a
`WorkerError`.

GUI smoke test: build the app hidden with workers disabled, feed synthetic events,
check the displayed text; skipped when no display is available.

On-Pi acceptance checklist (added to `PI_Instructions.md`): boots into the app full
screen; Sign mode recognizes words; switch to Speech and back; transcript scrolls;
the figure signs vocabulary words; Undo, Clear, A−/A+; unplug the camera → banner →
re-plug → Retry recovers; Exit and Power off.

## Build order

One spec, implemented in two stages, each working on its own:

1. The app with both modes (events, workers, state, Tk views, autostart).
2. Speech → sign: library builder, figure geometry and playback, transcript
   highlighting.
