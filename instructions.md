# Senseless — Setup & Operation Guide

End-to-end instructions: set up the dev PC, test both channels locally, collect
data, train + quantize the GRU on Colab, program the Raspberry Pi, and run the
live system. See [ARCHITECTURE.md](ARCHITECTURE.md) for the module map/data flow
and [CLAUDE.md](CLAUDE.md) for the locked decisions.

> **All commands in this guide are PowerShell**, run from the project root
> (`C:\Users\Asaf Amrani\Desktop\Fourth Year\Senseless`). Not CMD — the setup
> steps use PowerShell-only cmdlets. New to this? See **Running the commands** below.

## Status legend

The project is built incrementally. Each step is tagged:

- ✅ **available now** — the code exists; the commands work today.
- ⬜ **planned** — the module isn't built yet; the command shown is the *intended*
  one and will work once we implement that step. (Tracked in ARCHITECTURE.md.)

| # | Step | Status |
|---|------|--------|
| 2 | PC development setup | ✅ |
| 3 | Test the sign channel on PC (preview + benchmark) | ✅ |
| 4 | Test the speech channel on PC (ASR mic test) | ✅ |
| 5 | Collect training data | ✅ |
| 6 | Train the GRU on Colab | ⬜ `notebooks/` pending |
| 7 | Quantize to INT8 TFLite | ⬜ (part of the notebook) |
| 8 | Program the Raspberry Pi | ✅ doc (Pi-side run needs steps 4–7) |
| 9 | Run the full system | ⬜ orchestrator pending |

---

## Running the commands (PowerShell)

Every command block below is **PowerShell**, run from the project root. To run
anything in this guide:

1. **Open PowerShell in the project folder.** In File Explorer, open the
   `Senseless` folder, click the address bar, type `powershell`, and press Enter
   — it opens already pointed at that folder.
   *(Alternative: Start menu → "Windows PowerShell", then*
   `cd "C:\Users\Asaf Amrani\Desktop\Fourth Year\Senseless"`*.)*

2. **Confirm you're in the right place** — this should list `senseless`,
   `pyproject.toml`, `.venv`, etc.:
   ```powershell
   ls
   ```

3. **Run a command.** No need to "activate" the venv — call its Python by path.
   For example, run the whole test suite:
   ```powershell
   .venv\Scripts\python -m pytest
   ```
   You should see something like `24 passed`. That's the quickest check that
   your setup is healthy.

**Handy to know**
- Stop a running command (e.g. the mic test) with **Ctrl+C**; close a preview
  window by pressing **q**.
- Commands that start with `.venv\Scripts\python` also work in CMD, but the
  `Invoke-WebRequest` / `Expand-Archive` download steps are PowerShell-only — so
  just use PowerShell throughout.

## 1. Prerequisites

**Hardware**
- Dev PC (Windows; this repo is set up here) with an HD webcam + microphone.
- Raspberry Pi 4B (8 GB) + official 5 V/3 A USB-C PSU.
- microSD card (≥ 32 GB, A1/A2) + a way to write it.
- Pi Camera Module 3 (CSI ribbon) and ReSpeaker USB Mic Array.

**Software / accounts**
- Git, Python 3.11, a Google account (for Colab).
- Raspberry Pi Imager (for flashing the SD card).

---

## 2. PC development setup ✅

```powershell
# From the repo root (C:\...\Senseless)

# 2.1 Python 3.11 (skip if `py -3.11 --version` already works)
winget install --id Python.Python.3.11 -e --scope user

# 2.2 Create the virtualenv and install the package (editable) + dev tooling
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e . -r requirements-dev.txt

# 2.3 Confirm everything is green
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
```

Optional: `.venv\Scripts\pre-commit install` to run ruff/black on every commit.

---

## 3. Test the sign channel on your PC ✅

```powershell
# 3.1 Perception runtime deps (MediaPipe brings OpenCV-with-GUI on Windows;
#     do NOT also install opencv-python -- it causes a dual-cv2 conflict).
.venv\Scripts\python -m pip install mediapipe

# 3.2 Download the two MediaPipe model bundles into models\
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" -OutFile models\pose_landmarker_lite.task
Invoke-WebRequest "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile models\hand_landmarker.task

# 3.3 Live preview (draws landmarks + status overlay)
.venv\Scripts\python -m senseless.sign.preview

# 3.4 Throughput sanity (FPS/CPU)
.venv\Scripts\python -m senseless.eval.bench_perception --backend tasks --frames 200
```

**In the preview window:** green dots = left hand, blue = right hand, red = the
9 pose points, yellow cross = shoulder-midpoint (normalization origin). The
overlay shows `feature: ACTIVE nz=.../153` when you're in frame and `ALL-ZERO`
when not. Raise one hand → the other 63-block stays zero (the zero-fill policy).
Press `q`/`Esc` to quit.

**Checks / gotchas**
- **L/R swapped?** (common with front webcams) set `SIGN.mirror = True` in
  [config.py](senseless/common/config.py) and re-run.
- **Wrong camera?** use `--source 1` (or 2). Use a file with `--video clip.mp4`.
- **PC FPS ≫ Pi.** This only proves it *works* on x86; the real budget number
  comes from running the benchmark on the Pi (step 8).
- `--backend holistic` errors on the PC (MediaPipe ≥ 0.10.x dropped
  `mp.solutions`); it only runs on the Pi's pinned 0.10.14. Use `tasks`.

---

## 4. Test the speech channel on your PC ✅

The dedicated PC mic-test step. Install the ASR deps, fetch the Vosk model, then
run the live mic test:

```powershell
# 4.1 ASR deps
.venv\Scripts\python -m pip install vosk sounddevice

# 4.2 Vosk small en-us model into models\
Invoke-WebRequest "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile models\vosk.zip
Expand-Archive models\vosk.zip -DestinationPath models
Remove-Item models\vosk.zip
# leaves models\vosk-model-small-en-us-0.15  (matches config.PATHS.vosk_model_dir)

# 4.3 Live mic test (prints streaming partial + final transcripts)
.venv\Scripts\python -m senseless.asr.mic_test
```

It prints live partials and finalized lines. Use `--list-devices` to see input
devices and `--device N` to pick one (it auto-selects a ReSpeaker by name hint
otherwise). Confirms your mic + Vosk work before the Pi.

---

## 5. Collect training data ✅

The classifier learns from short windows of the **same 153-vector** the runtime
produces, so collection reuses `sign/landmarks.py` — guaranteeing train/inference
parity.

1. **Finalize the vocabulary** (your `vocab_core.txt`, ~50–150 words). It becomes
   `models/sign_labels.txt` (one label per line, in class-index order).
2. **Record samples**, one label at a time, watching the live window:
   ```powershell
   .venv\Scripts\python -m senseless.collect --label HELLO --samples 40
   ```
   Press **SPACE** to record one take (a 45-frame window captured over ~1.5 s).
   The overlay shows `detected: yes/NO`, a red dot + `REC k/30` while capturing,
   and `saved N/40`. Press **q** to quit. Each take is one `(45, 153)` array
   under `data/<label>/`, already normalized. Re-running the same `--label`
   resumes the count, so you can collect across multiple sessions.

**Collection tips**
- Aim for **30–50+ samples per word**, across multiple sessions.
- Vary position, distance, and lighting — normalization handles translation/scale,
  but variety still helps generalization.
- Match real use: same camera height/framing you'll deploy with.
- Record one-handed signs one-handed (the missing hand stays a zero-block).
- Keep a few **"nothing"/idle** samples if you want a negative/garbage class.

---

## 6. Train the GRU on Google Colab ⬜ (`notebooks/` pending)

Training happens on Colab, never on the Pi.

1. Zip `data/` and upload to Google Drive (or push to a Drive-mounted folder).
2. Open `notebooks/train_sign_gru.ipynb` in Colab; mount Drive.
3. The notebook will:
   - Load all `(45, 153)` windows + labels, make a **stratified train/val/test split**.
   - Build a small **GRU** (target **< 1M params**, e.g. 1–2 GRU layers of 96–128
     units + a dense softmax over your vocabulary).
   - Train with early stopping; report **held-out accuracy + a confusion matrix**
     (per the ML working style: evaluation report, not TDD on accuracy).
   - Export a float `.tflite`, then **INT8-quantize** it (step 7).
4. Download `sign_gru_int8.tflite` and `sign_labels.txt` into your local `models/`.

> The model input is `(45, 153)` and output is one logit per vocabulary word;
> `sign_labels.txt` must list the words in the **same index order** the model emits.

---

## 7. Quantize to INT8 TFLite ⬜ (in the notebook)

Done inside the training notebook via `tf.lite.TFLiteConverter`:
- Provide a **representative dataset** (~100–300 real training windows) so the
  converter calibrates activation ranges.
- Target full-integer (`int8`) so it runs fast under **XNNPACK** on the Pi.
- **Verify the accuracy drop is small** vs the float model. RNN/GRU full-INT8
  conversion can be finicky; if it misbehaves, fall back to dynamic-range or
  float16 quantization and note the trade-off. Keep the artifact at
  `models/sign_gru_int8.tflite` (matches `config.PATHS.sign_tflite`).

---

## 8. Program the Raspberry Pi ✅ (doc)

> The Pi runs **inference only**. You can do 8.1–8.5 now; running the sign
> classifier (8.6) needs the trained model from steps 5–7, and ASR needs step 4.

**8.1 Flash the OS.** In Raspberry Pi Imager choose **Raspberry Pi OS (64-bit),
Bookworm**. In the gear/settings: set hostname, enable **SSH**, set username +
password, and Wi-Fi/locale. Write the card, boot the Pi.

**8.2 First boot + system update**
```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

**8.3 System packages** (camera, audio, build basics)
```bash
sudo apt install -y git python3-venv python3-picamera2 libportaudio2
# python3-picamera2 + libcamera are usually preinstalled on Bookworm.
```

**8.4 Verify the camera and mic**
```bash
rpicam-hello -t 2000        # (older images: libcamera-hello) -- a preview = camera OK
arecord -l                  # list capture devices; confirm the ReSpeaker appears
```

**8.5 Clone + install (note the system-site-packages venv)**
```bash
git clone <your-repo-url> senseless && cd senseless
# picamera2/libcamera come from APT, so the venv must see system packages:
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e . -r requirements-pi.txt
```
If a pinned wheel in `requirements-pi.txt` won't install on aarch64, adjust the
pin (see the comments in that file) and note what worked.

**8.6 Copy the models to the Pi** (`models/` is gitignored, so transfer it):
```bash
# from your PC, e.g. with scp:
scp -r models <pi-user>@<pi-host>:~/senseless/
# needs: pose_landmarker_lite.task, hand_landmarker.task,
#        vosk-model-small-en-us-0.15/, sign_gru_int8.tflite, sign_labels.txt
```

**8.7 Confirm perception on the Pi + record the numbers**
```bash
.venv/bin/python -m senseless.eval.bench_perception --frames 300
# Both backends run here (0.10.14 has holistic). Paste the FPS/CPU table into
# senseless/sign/README.md to lock the backend decision with real data.
```
If FPS is too low: drop camera resolution in `config.CAMERA`, keep the lite pose
model, or run pose every N frames. If colors look wrong, flip
`config.CAMERA.pixel_format` to `"BGR888"`.

---

## 9. Run the full system ⬜ (orchestrator pending)

Intended entry point once the queue, classifier runner, UI, and orchestrator are built:
```bash
.venv/bin/python -m senseless.app
```
This will launch the sign and ASR channels as **separate processes** (4-core
budget), feed them through **bounded drop-oldest ring buffers**, and show the
**dual live transcript** (sign words + speech text). Press `q` to quit.

---

## 10. Troubleshooting quick reference

| Symptom | Fix |
|---|---|
| Left/right hands swapped | `SIGN.mirror = True` in config.py |
| Wrong webcam opened | `--source 1` (preview/bench) |
| `holistic` errors on PC | expected — use `tasks` (PC MediaPipe dropped `mp.solutions`) |
| picamera2 colors look wrong | set `config.CAMERA.pixel_format = "BGR888"` |
| `sounddevice` import/PortAudio error on Pi | `sudo apt install -y libportaudio2` |
| ReSpeaker not picked up | check `arecord -l`; adjust `AUDIO.device_name_hints` |
| Low FPS on Pi | lower `CAMERA` resolution; lite models; reduce pose cadence |
| INT8 model accuracy poor | recheck representative dataset; try float16/dynamic-range |

---

## 11. Where things live

- **All tunables:** [config.py](senseless/common/config.py) — change behavior here, not inline.
- **Feature-vector contract (153-dim):** [landmark_schema.py](senseless/common/landmark_schema.py).
- **Same normalization for collect + train + infer:** `frame_landmarks_to_vector`
  in [landmarks.py](senseless/sign/landmarks.py) — the single source of truth.
- **Models** (all gitignored, fetched/trained, placed in `models/`): the two
  `.task` bundles, the Vosk model dir, `sign_gru_int8.tflite`, `sign_labels.txt`.
- **Design rationale:** [sign/README.md](senseless/sign/README.md), [ARCHITECTURE.md](ARCHITECTURE.md).
