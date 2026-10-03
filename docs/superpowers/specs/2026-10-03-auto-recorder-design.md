# Hands-free recording: design

Date: 2026-10-03. Status: approved in brainstorming.

## Goal

Collect sign samples faster, without pressing a key before every take. The signer
hides both hands behind their back between takes. A take starts when hands come into
frame. The next take is armed once the hands have left the frame again.

## Current behaviour (unchanged without `--auto`)

`python -m senseless.collect --label X` (`senseless/collect/recorder.py`):

- SPACE arms one take.
- Capture starts on the first frame with a hand, the onset.
- It runs for one window span (`default_span_s()`, about 1.47 s).
- It is resampled to 45 steps and saved with `dataset.save_window`.
- The recorder then returns to idle.

## New behaviour

### `--auto` flag

The take cycle with `--auto`:

1. **armed:** waits for a hand. The first frame with a hand starts the take.
2. **recording:** records one span and saves the take.
3. **waiting:** waits for the hands to leave. The recorder re-arms only after no hand
   has been seen for `SIGN.collect_clear_s` (0.5 s) without a break. A hand frame
   restarts the clock. A single missed frame is not enough, because the tracker loses
   hands for a few frames during fast signs.
4. Back to armed.

Other details:

- Auto mode starts armed.
- SPACE toggles a **paused** state.
  - While paused, nothing arms or records.
  - Pausing during a recording discards that take.
  - Resuming goes to `waiting`, so hands already in view do not start a take at once.
- Reaching `--samples` does not stop recording. The count is shown, as today.

### Undo (both modes)

BACKSPACE deletes the most recently saved take of this session. Pressing it again
deletes the take before that.

- Only files saved in the current run can be deleted.
- Nothing happens when there are none.
- Takes that were never saved are not affected.
- The saved count drops by one per undo, and the console prints `deleted 0042.npy`.

### Overlay

The status line shows one of the following:

- `SPACE = arm a take`
- `armed - waiting for a hand...`
- `REC 0.8/1.5 s`
- `hands down to re-arm`
- `PAUSED - SPACE to resume`

The controls line lists BACKSPACE = undo.

## Code

- **`senseless/common/config.py`:** adds `SignConfig.collect_clear_s: float = 0.5`.
- **`senseless/collect/recorder.py`:** a new pure class `TakeMachine(span_s, auto,
  clear_s, length)`. It holds no camera and no cv2.
  - `step(stamp, vec, hands_present) -> np.ndarray | None` advances the state and
    returns a finished (45, 153) take when one completes.
  - `arm()` arms a take in manual mode, from idle only.
  - `toggle_pause()` pauses and resumes in auto mode.
  - `state` is one of `"idle"`, `"armed"`, `"recording"`, `"waiting"` or `"paused"`.
  - `onset` holds the onset time.
  - The camera loop in `main()` calls `step` and saves what it returns. It keeps a list
    of the paths saved this session for undo.
- **`senseless/collect/dataset.py`:** no change. Undo uses `Path.unlink`.
  `next_sample_index` already tolerates gaps and reuses the freed top index.

## Testing

`senseless/tests/test_recorder_machine.py`, with synthetic timestamps and no camera:

- in manual mode, a take starts on the first hand frame after `arm()`, completes after
  the span, and the state returns to idle;
- `step` before `arm()` never records in manual mode;
- in auto mode, after a take the machine waits and re-arms only after `clear_s` of
  continuous no-hand frames;
- a no-hand gap shorter than `clear_s` (0.3 s) followed by a hand does not re-arm;
- pausing blocks arming; resuming goes to waiting; pausing mid-recording returns no take;
- the take shape is (45, 153), and the take covers `[onset, onset + span]`.

Undo is tested through a small helper, `undo_last(saved: list[Path]) -> Path | None`,
that pops the list and unlinks the file.

## Out of scope

- auto-discarding empty takes;
- stopping at `--samples`;
- a countdown;
- any change to the training data format.
