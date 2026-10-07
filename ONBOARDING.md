# Senseless: developer onboarding

Welcome. This guide gets you from an empty PC to running, testing and changing
Senseless, and points to the deeper docs when you need them. Read it top to bottom
once (about 20 minutes). After that, use it as a reference.

## 1. What you are working on

Senseless is an offline Raspberry Pi 4 device with a 7" touchscreen. It has two modes,
one active at a time:

- **Sign mode:** the camera feeds MediaPipe hand and pose landmarks to a small GRU, which
  outputs an ASL word. Words build up into a sentence on screen.
- **Speech mode:** the microphone feeds Vosk, which streams a live transcript. When a
  spoken word has a sign in the figure's library, it is highlighted and a stick figure
  signs it. The library has 140 words: our own 43, plus 97 built from the MS-ASL
  dataset.

**Current state (October 2026):**

- The vocabulary has 43 words plus `IDLE`.
  - 2,210 takes were recorded on the PC by several signers (`data/`).
  - 420 takes were recorded on the Pi itself, about 10 per word (`data_pi/`).
- The deployed model reaches 98.5% on held-out PC samples (98.7% before INT8 export),
  and 97.3% when the samples are simulated at 10 FPS.
- On held-out Pi takes, the recipe reaches about 90%, against about 45% without the Pi
  takes. The Pi recognizes the vocabulary reliably in live use.
- The Pi runs the sign channel at 6.5-7.5 FPS and boots straight into the touchscreen app.
- **Next steps:**
  - test the model on signers it has never seen;
  - record more Pi takes;
  - run both channels at the same time.

**The rules that are not up for debate** live in [CLAUDE.md](CLAUDE.md). The ones you will
hit first:

- 4 cores are the whole budget. CPU-heavy stages run as **processes**, not threads.
- All queues are bounded and **drop the oldest** item. There are no unbounded queues.
- Everything runs offline on the Pi, and the Pi only runs inference. **Training happens
  on the PC or Colab.**
- The sign and speech channels share only `senseless/common/`.

## 2. Day 1: set up your PC (Windows)

### 2.1 Before you start

- **Get GitHub access.** The repo is private: `Hydrolicks/senseless`. Ask Asaf to add you
  as a collaborator.
- **Install Python 3.11.** Use exactly this version, because it matches the Pi. Run
  `py -3.11 --version` to check.
- **Use a folder path with only English letters**, for example `C:\Senseless`.
  MediaPipe's C++ code cannot open model files under a path with Hebrew (or any
  non-ASCII) characters. You get a confusing `FileNotFoundError` for
  `pose_landmarker_lite.task` even though the file is there.

### 2.2 Clone and install

```powershell
git clone https://github.com/Hydrolicks/senseless.git C:\Senseless
cd C:\Senseless
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e . -r requirements-dev.txt
.venv\Scripts\python -m pip install mediapipe vosk sounddevice
.venv\Scripts\pre-commit install
```

- **Training** needs `tensorflow scikit-learn matplotlib` on top of this. Install them
  only when you train.
- **Never copy someone else's `.venv` folder.** A virtual environment contains absolute
  paths to the machine it was made on, so it does not work anywhere else. Always create
  your own.
- **Don't install `opencv-python`.** MediaPipe already brings its own OpenCV build, and
  the two overwrite each other.

### 2.3 Get the models and the data

`models/` and `data/` are not in git: they are large binaries and are listed in
`.gitignore`. Ask Asaf for a zip of both folders and unpack it into the repo root:

| Path | What it is | How to rebuild it yourself |
|---|---|---|
| `models/hand_landmarker.task`, `models/pose_landmarker_lite.task` | MediaPipe model bundles | Download (`instructions.md` §3.2) |
| `models/vosk-model-small-en-us-0.15/` | Speech model | Download (`instructions.md` §4.2) |
| `models/sign_gru_int8.tflite`, `models/sign_labels.txt` | The trained sign classifier | `python -m senseless.notebooks.train_gru --pi-data data_pi --pi-test-frac 0` |
| `models/sign_library.npz` | One take per word for the signing figure: ours, plus MS-ASL takes for other words | `python -m senseless.sign.library` |
| `models/sign_library_extra.npz` | MS-ASL takes for words we did not record (credit: MS-ASL, C-UDA). Not in git; back it up | `python -m senseless.sign.extra_library build` |
| `data/<WORD>/NNNN.npy` | Takes recorded on the PC: one (45, 153) window per take, about 50 per word | Record with `python -m senseless.collect --auto` |
| `data_pi/<WORD>/NNNN.npy` | Takes recorded on the Pi, about 10 per word, same format. The Pi's `~/senseless/data` is the master copy | Record on the Pi (section 7) and copy it over |

### 2.4 Check that everything works

```powershell
.venv\Scripts\python -m pytest              # 230 tests in ~10 s; all must pass
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m black --check .
```

On Windows, a Tk test sometimes skips with "no display available". This comes from a
`tcl_findLibrary` flake on Windows and is not a real failure. Re-run the tests, and the
skip disappears.

Then try the real thing:

```powershell
.venv\Scripts\python -m senseless.ui --windowed      # the touchscreen app in an 800x480 window
.venv\Scripts\python -m senseless.sign.demo          # live sign recognition (onset mode)
.venv\Scripts\python -m senseless.asr.mic_test       # live speech transcript
.venv\Scripts\python -m senseless.sign.preview       # landmarks only, to check your camera
```

**How to sign so the model recognizes you:**

1. Rest your hands out of view.
2. Make the sign.
3. Lower your hands. The word appears.

The model only knows the few signers who recorded it, in one room, so expect it to work
less well for you at first. That is a known limitation, not a bug.

## 3. How the code is organised

```
senseless/
  common/     config.py (ALL tunables), landmark_schema.py (the 153-value vector),
              queue.py (DropOldestQueue), events.py, devices.py, process.py
  sign/       capture.py, landmarks.py (backends), parallel.py, window.py,
              segmenter.py (onset mode), augment.py, classifier.py, library.py,
              extra_library.py (MS-ASL figure takes), worker.py (Sign mode of the
              app), demo.py, preview.py
  asr/        transcriber.py, audio.py, worker.py (Speech mode of the app), mic_test.py
  collect/    recorder.py + dataset.py: the data-collection tool;
              msasl.py: downloads MS-ASL clips for the signing figure
  notebooks/  train_gru.py: split, augment, train, evaluate, export INT8 TFLite
  ui/         app.py (Tkinter), controller.py (one worker process at a time),
              state.py (pure rules), figure.py (stick figure), __main__.py
  eval/       bench_perception.py: FPS/CPU of each perception backend
  tests/      pytest suite
deploy/       Pi launcher script + desktop autostart entry
docs/         vocabulary reference; superpowers/ = app design spec + implementation plan
```

**Reading order for the code:**

1. `common/config.py` and `common/landmark_schema.py`
2. `sign/landmarks.py` (the `frame_landmarks_to_vector` function)
3. `sign/window.py`
4. `sign/segmenter.py`
5. `sign/classifier.py`
6. `ui/controller.py`
7. `ui/app.py`

[ARCHITECTURE.md](ARCHITECTURE.md) has the data-flow diagrams and module map. The
project book (`Assigment Submission Files/Senseless_Project_Book.pdf`) explains every
design decision and the measurements behind it.

## 4. Ideas you need before changing things

- **One config, no magic numbers.** Every tunable is a field of a frozen dataclass in
  `common/config.py`: `SIGN`, `UI`, `AUDIO` and so on. To change behaviour, change it
  there and import it.
- **The feature vector is a contract.** A frame becomes 153 floats, laid out like this:
  - 63 values for the left hand, then 63 for the right hand;
  - 27 values for 9 pose points;
  - everything is normalized to the shoulders, and a missing hand is all zeros.
  
  The recorder, the training script, the live classifier and the stick figure all use
  the same function. **If you change the layout or the normalization, every recorded
  sample becomes invalid.**
- **Windows are time-based.** One sign is 1.47 s of motion, resampled to 45 steps
  (`sign/window.py`), so the model sees the same speed at 7 FPS on the Pi as at 30 FPS on
  the PC.
- **Onset mode.** The model was trained on windows that start the moment a hand appears.
  `OnsetSegmenter` cuts live windows the same way: it classifies exactly once per sign,
  then waits for the hands to leave the frame. Classifying a sliding window flickers.
- **Perception backends:**
  - `tasks` is the default on the PC. It is the only backend MediaPipe 0.10.35 supports,
    and all the data was recorded with it.
  - `lite` uses the old `mp.solutions` API with complexity 0. It is about 3x faster for
    hands and runs on the Pi with MediaPipe 0.10.18.
  - `--parallel` runs pose and hands in two processes.
  - The model copes with the `lite`/`tasks` difference because it is also trained on
    takes recorded on the Pi (`data_pi/`) and with augmentation that imitates the Pi
    (`sign/augment.py`). Augmentation alone was not enough.
- **The app runs one worker process at a time.** The Tk window does no heavy work.
  `ModeController` spawns the sign or speech worker, polls its queues every 40 ms, and
  stops it without freezing the screen. Workers report through event dataclasses
  (`common/events.py`), and errors become a banner with **Retry**.
- **The Pi environment is pinned.** Raspberry Pi OS **Bookworm** (Python 3.11) with
  `mediapipe==0.10.18`, `ai-edge-litert==1.4.0` and `numpy<2`, as listed in
  `requirements-pi.txt`. **Do not upgrade these casually.** Newer MediaPipe and LiteRT
  builds crash on the Pi 4 with "compiled with AES enabled… Illegal instruction", which
  is why the Pi does not run Trixie.

## 5. Daily workflow

1. **Branch from `main`.** Use a short name, such as `add-water-sign` or `fix-banner-text`.
2. **Write the test first** for logic that needs no hardware: pure functions, state
   machines, workers with fake parts (look at `tests/ui_fakes.py` and the worker
   tests). ML training code is checked with a held-out evaluation instead of unit tests.
3. **Keep it green** before every commit:
   - `pytest`, `ruff check .` and `black --check .` all pass.
   - The pre-commit hook runs ruff and black for you.
4. **Stage files by name** (`git add path/to/file`), never `git add -A`. The repo
   folder holds untracked deck and report files that must stay out of git.
5. **Push and open a PR into `main`** on GitHub. Ask Asaf to review before merging.
   Never push large files: models, data, `.npz`, videos.

**Commit messages:** an imperative subject line that says what changed for the user or
system, for example "Keep the newest transcript line in view after a text-size change".

## 6. Common tasks

### Add a new word (or retrain with more data)

[senseless_retraining.md](senseless_retraining.md) has every command. In short:

1. **Choose one form of the sign.** Look it up in `docs/vocab_reference.md`, then record
   only that form.
2. **Record about 50 takes on the PC** with the same camera, distance and lighting as the
   existing data:
   `python -m senseless.collect --label WATER --samples 50 --auto`
   - Raise your hands and sign. Capture starts by itself when a hand appears.
   - Hide your hands for at least 0.5 s between takes; the next take then arms itself.
   - BACKSPACE deletes the last take if you fumbled it, and SPACE pauses.
3. **Record about 10 takes on the Pi** (section 7) and copy the Pi's `~/senseless/data` to
   `data_pi/`. A word without Pi takes is recognized much less reliably on the Pi.
4. **Back up** `data/`, `data_pi/` and `models/`. The trainer overwrites the deployed
   model.
5. **Measuring run:** `python -m senseless.notebooks.train_gru --pi-data data_pi`.
   - Check the test accuracy (including the "simulated 10 FPS" number) and the
     confusion matrix.
   - Check the **Pi test accuracy** per word, measured on Pi takes the model did not
     train on.
6. **Deploy run:** `python -m senseless.notebooks.train_gru --pi-data data_pi --pi-test-frac 0`.
   This trains on every Pi take; it is the model to ship.
7. **Rebuild the figure library:** `python -m senseless.sign.library`. Words you did not
   record keep their MS-ASL takes from `models/sign_library_extra.npz`, and a word you
   record replaces its MS-ASL take.
8. **If people say the word differently** (e.g. "thanks" for THANKYOU), add the spoken
   form to `UI.sign_aliases` in `config.py`.
9. **Copy the new model files to the Pi** (section 7).

### Change a threshold or a timing

Edit the field in `common/config.py`, for example:

- `SIGN.min_confidence` (0.6)
- `SIGN.release_s` (0.3 s)
- `UI.switch_timeout_s`

Run the tests, then try the change live on the PC before you take it to the Pi.

### Work on the touchscreen app

Run `python -m senseless.ui --windowed --mode speech` (or `--mode sign`). To test
without a camera or a microphone, follow the hidden-window tests in
`tests/test_ui_app.py`: they drive the app with a fake controller and synthetic events.

## 7. Working with the Pi

The Pi is `admin@senseless.local`, with the repo at `/home/admin/senseless`. Setup, the
autostart and the acceptance checklist are in [PI_Instructions.md](PI_Instructions.md);
§17 covers the touchscreen app.

```bash
ssh admin@senseless.local
cd ~/senseless && git switch main && git pull          # get new code
tail -n 50 ~/senseless.log                             # the app's log
pkill -f senseless.ui                                  # stop the app
DISPLAY=:0 ~/senseless/deploy/senseless-ui.sh &        # start it on the touchscreen
.venv/bin/python -m senseless.sign.demo --backend lite --parallel --camera opencv --headless
```

Record takes on the Pi (stop the app first, because it holds the camera):

```bash
DISPLAY=:0 .venv/bin/python -m senseless.collect --label WATER --samples 10 --auto --backend lite --parallel --camera opencv
```

Copy the Pi takes to the PC (in PowerShell on the PC). Delete the old `data_pi` first;
otherwise scp nests the copy inside it:

```powershell
scp -r admin@senseless.local:~/senseless/data data_pi
```

Copy model files **from the PC** (run `scp` in PowerShell on the PC, not on the Pi):

```powershell
scp models\sign_gru_int8.tflite models\sign_labels.txt models\sign_library.npz models\sign_library_extra.npz admin@senseless.local:~/senseless/models/
```

The Pi's `.venv` has no pytest. Install it there with `.venv/bin/python -m pip install pytest`
if you want to run the tests on the board. Prefix the command with `DISPLAY=:0` to
include the Tk tests.

## 8. Gotchas we already paid for

| Symptom | Cause | Fix |
|---|---|---|
| `No module named .venv` / broken venv after copying the project | Copied venv | Make a fresh venv on each machine |
| `FileNotFoundError` for a `.task` file that exists | Non-ASCII (e.g. Hebrew) folder path | Move the project to an ASCII path |
| "Illegal instruction … AES" on the Pi | Trixie / newest MediaPipe or LiteRT | Bookworm + the pinned versions |
| Left and right hands swapped | Mirrored camera, or the `lite` backend's labels | `SIGN.mirror`; `lite` already swaps its labels |
| Words flicker in the demo | Continuous mode | Use `--mode onset` (the default) |
| Recognition poor on the Pi but fine on the PC | The Pi's lite tracker, frame rate and camera differ from the PC recordings | Record about 10 takes per word on the Pi and train with `--pi-data data_pi` |
| A word gets no better on the Pi after retraining | Its Pi takes were skipped: the word is missing from `data/`, or its folder name differs | Check the trainer's `Pi data: skipped` line; folder names must match exactly |
| The recorder can't open the camera on the Pi | The default frame source on the Pi is the camera module | Add `--camera opencv` for the USB webcam |
| Pi takes end up in `data_pi/data/` | `scp -r` into a folder that already exists | Delete `data_pi` before copying |
| Pi app not full screen / power icon a box | labwc and the Pi's fonts | Fixed in PR #5; pull `main` |
| `scp` "file not found" | Ran it on the Pi | Run `scp` on the PC |
| Tk test skipped on Windows | Local Tcl flake | Re-run |

## 9. Docs that are partly out of date

- **[instructions.md](instructions.md)** marks training (§6-7) and "run the full system"
  (§9) as pending:
  - Training now runs locally with `senseless.notebooks.train_gru`.
  - The touchscreen app (`python -m senseless.ui`) is the full system.
- **instructions.md** also says Holistic runs on the Pi's MediaPipe 0.10.14. The Pi now
  uses 0.10.18.
- **[README.md](README.md)** still describes the sign channel as "MediaPipe Holistic".
  The device uses the `lite` backend on the Pi and `tasks` on the PC.

When the code and these docs disagree, trust the code, the project book and this file,
and fix the doc in your PR.

## 10. Where to go next

| You want to… | Read |
|---|---|
| Understand the decisions and measurements | The project book, chapters 4-14 |
| Set up or fix the Pi | `PI_Instructions.md` |
| Retrain with new recordings | `senseless_retraining.md` |
| Review the code in order | `ARCHITECTURE.md` → "Suggested review order" |
| Understand the app's design | `docs/superpowers/specs/2026-09-30-touchscreen-gui-design.md` |
| See open work | The project book, §13.11 (measurements) and §15.2 (future work) |

Questions? Ask Asaf. Anything this guide got wrong, fix it in a PR.
