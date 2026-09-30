# Senseless

On-device, real-time system for a Raspberry Pi 4B (8GB) that bridges
Deaf↔hearing communication via two independent channels:

1. **Sign → text:** Camera → MediaPipe Holistic landmarks (face excluded) →
   temporal classifier → ASL isolated word.
2. **Speech → text:** Microphone → Vosk → streaming partial transcripts.

See [CLAUDE.md](CLAUDE.md) for the locked technical decisions and constraints,
[ARCHITECTURE.md](ARCHITECTURE.md) for the module map, data flow, and a suggested
code-review order, [instructions.md](instructions.md) for the full
setup/operation guide (PC testing → data collection → Colab training → Pi deploy),
and [PI_Instructions.md](PI_Instructions.md) for the detailed Raspberry Pi bring-up
(camera + mic wiring, configuring both channels, autostart, troubleshooting). Run the touchscreen app with `python -m senseless.ui` (see `PI_Instructions.md` §17).

## Repo layout

```
senseless/
  asr/         Vosk streaming module (Pi runtime)
  sign/        MediaPipe capture, landmark extraction, on-Pi inference
  collect/     data-collection CLI
  notebooks/   Colab training + quantization
  common/      shared: config, landmark schema, queue, device selection
  ui/          dual live-transcript display
  eval/        accuracy/latency harness
  tests/       pytest suite
```

## Environments

Runtime dependencies are environment-specific and split into three files:

| File                    | Where it runs        | Contents                                   |
| ----------------------- | -------------------- | ------------------------------------------ |
| `requirements-pi.txt`   | Raspberry Pi (aarch64) | vosk, mediapipe, opencv, sounddevice, picamera2, litert |
| `requirements-colab.txt`| Google Colab         | tensorflow/keras, numpy, matplotlib, scikit-learn |
| `requirements-dev.txt`  | Dev/host machine     | pytest, ruff, black, pre-commit            |

The Pi is **inference-only**; training/quantization happen on Colab.

## Dev setup (host machine)

Requires Python 3.11 (matching the Pi's Bookworm Python).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e . -r requirements-dev.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
```

`.venv\Scripts\pre-commit install` wires up the lint/format git hook.

## Pi setup (on-device)

```bash
# python3-picamera2 + python3-libcamera come from APT (preinstalled on Bookworm).
# The venv must see them, so create it with system site-packages:
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e . -r requirements-pi.txt
```
