# Senseless — Raspberry Pi setup & operation

A complete guide to bringing up the Pi so both channels — **sign** (camera →
words) and **speech** (mic → text) — run and are programmable. All commands here
run **on the Pi** (directly, or over SSH from your PC).

> **Status note.** The Pi is **inference-only** (no training). Today the two
> channels run as **separate tools** — `senseless.sign.demo` and
> `senseless.asr.mic_test`. The unified app (`senseless.app`) that runs both
> together with shared queues is not built yet; when it is, point the autostart
> service (§13) at it. Everything else below is ready now.

Related: [instructions.md](instructions.md) (whole-project lifecycle),
[ARCHITECTURE.md](ARCHITECTURE.md) (how the code fits together).

---

## 1. Hardware & connections

| Part | Connects to | Notes |
|---|---|---|
| **Raspberry Pi 4B (8 GB)** | — | official 5 V / 3 A USB-C PSU; add a heatsink/fan (§14) |
| **microSD (≥ 32 GB, A1/A2)** | SD slot | holds OS + project + models |
| **Pi Camera Module 3** | **CSI camera port** (ribbon) | see below |
| **ReSpeaker USB Mic Array** | any **USB** port | class-compliant, no driver |
| Monitor (optional) | micro-HDMI | only needed for the GUI preview; headless is fine otherwise |
| Keyboard/mouse (optional) | USB | or run fully over SSH |

**Camera ribbon (do this with the Pi powered off):**
1. On the Pi, lift the black plastic latch on the **CSI camera** connector (not the
   DISPLAY one).
2. Insert the ribbon so the **blue stripe faces the USB/Ethernet ports** side (metal
   contacts face the HDMI side), fully seated and straight.
3. Press the latch back down. Do the same at the camera-board end.

**Mic:** plug the ReSpeaker into a USB **3.0 (blue)** port. That's it — it enumerates
as a USB audio device.

---

## 2. Image the OS

Use **Raspberry Pi Imager** (on your PC):
1. Choose **Raspberry Pi OS (Legacy, 64-bit)** (under "Raspberry Pi OS (other)") —
   Debian 12 Bookworm, Python 3.11. **Do not use the current Trixie release on a Pi 4:**
   the only MediaPipe builds for its Python 3.13 require the ARM AES instructions the
   Pi 4's CPU lacks, and crash with "compiled with aes enabled … Illegal instruction".
   64-bit is required (aarch64 wheels + performance).
2. Click the **gear / "Edit settings"** before writing and set:
   - **hostname** (e.g. `senseless`), so you can reach it at `senseless.local`
   - **Enable SSH** (password or key)
   - **username + password**
   - **Wi-Fi** SSID + password + country, and **locale/timezone**
3. Write the card, insert it into the Pi, connect camera + mic, power on.

---

## 3. First boot, update, SSH

From your PC:
```bash
ssh <user>@senseless.local        # or ssh <user>@<pi-ip>
```
On the Pi:
```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

---

## 4. Enable & verify the **camera**

On Bookworm the camera is auto-detected via **libcamera** (no `raspi-config` toggle
needed for Camera Module 3). Verify:
```bash
rpicam-hello --list-cameras     # should list the imx708 (Camera Module 3)
rpicam-hello -t 3000            # 3 s live preview (needs a display); or omit on headless
```
*(Older images use `libcamera-hello` instead of `rpicam-hello`.)*

If it isn't detected: power off, reseat the ribbon (§1), check it's the **CSI** port
and the stripe orientation. On headless setups `--list-cameras` still works without a
display.

---

## 5. Enable & verify the **microphone** (ReSpeaker)

The ReSpeaker is USB-audio-class — no driver. Verify it's seen:
```bash
lsusb                    # look for "SEEED" / "ReSpeaker" / a generic USB audio device
arecord -l               # list capture cards; note the card name/number
```
Record a 3-second test and play it back (swap in your card from `arecord -l`):
```bash
arecord -d 3 -f S16_LE -r 16000 -c 1 -D plughw:CARD=ArrayUAC10,DEV=0 test.wav
aplay test.wav           # needs speakers/headphones on the Pi's output
```
Notes:
- **Channels:** some ReSpeaker arrays expose multiple capture channels. Our capture
  opens **mono** (`AUDIO.channels = 1`); if opening mono fails, list the device's
  native channels (`arecord -L`) and either select the processed/merged channel or
  bump `AUDIO.channels` and downmix. The Python listing in §11 is the easiest way to
  pick the right device.
- You don't need to set the ReSpeaker as the *default* device — the code selects it
  **by name** (see §11).

---

## 6. System packages (APT)

```bash
sudo apt install -y git python3-venv python3-picamera2 libportaudio2
```
- `python3-picamera2` + `python3-libcamera` — the camera bindings (usually
  preinstalled on Raspberry Pi OS). **These come from APT, not pip** — which is why the venv
  below uses `--system-site-packages`.
- `libportaudio2` — required by `sounddevice` (the mic capture).

---

## 7. Get the project & install

```bash
git clone <your-repo-url> senseless && cd senseless

# The venv MUST see the APT-installed picamera2/libcamera:
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e . -r requirements-pi.txt
```
- `pip install -e .` is an **editable** install: after a `git pull` (or editing a
  file on the Pi), changes take effect immediately — no reinstall. That's what makes
  the Pi "programmable."
- If a pinned wheel in `requirements-pi.txt` won't resolve on aarch64, install the
  closest available version and note it. **MediaPipe** in particular: training used
  0.10.35 (no aarch64 wheel), so the Pi runs mediapipe 0.10.18 — that's fine as long
  as you use the same `.task` bundles (see §8). Don't also `pip install
  opencv-python`: mediapipe already brings `opencv-contrib-python`, and the two clash.

---

## 8. Transfer the models

`models/` is gitignored, so copy it from your PC. **The `.task` bundles must be the
same ones training used** — they (not the MediaPipe version) determine the landmark
values, so copy them rather than re-downloading a different variant.

From your PC:
```bash
scp -r models <user>@senseless.local:~/senseless/
```
`models/` must contain:

| File | For | Source |
|---|---|---|
| `pose_landmarker_lite.task` | sign perception | copy from dev box |
| `hand_landmarker.task` | sign perception | copy from dev box |
| `sign_gru_int8.tflite` | sign classifier | your trained model |
| `sign_labels.txt` | sign classifier | your trained labels |
| `vosk-model-small-en-us-0.15/` | speech | copy, or download on the Pi (below) |

Download the Vosk model on the Pi instead of copying, if you prefer:
```bash
cd ~/senseless/models
curl -L -O https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip && rm vosk-model-small-en-us-0.15.zip
```

---

## 9. Configure — the **sign** channel

All knobs live in [`senseless/common/config.py`](senseless/common/config.py) (edit
on the Pi; the editable install picks it up). The relevant ones:

| Setting | Default | On the Pi |
|---|---|---|
| `CAMERA.width` / `height` | 640 × 480 | lower these if FPS is low (§12) |
| `CAMERA.framerate` | 30 | target rate; real rate depends on load |
| `CAMERA.pixel_format` | `"RGB888"` | **flip to `"BGR888"` if colors look wrong** (picamera2 quirk) |
| `SIGN.perception_backend` | `"tasks"` | keep `tasks` (Holistic isn't on the Pi's MediaPipe) |
| `SIGN.mirror` | `False` | set `True` if left/right hands come out swapped |
| `SIGN.num_threads` | 2 | XNNPACK threads; leave cores for ASR + the classifier |
| `SIGN.min_confidence` | 0.6 | raise to suppress weak predictions in the demo |
| `SIGN.frame_stride` | 1 | raise to 2 to cover more real time per window if FPS is low |
| `PATHS.*` | `models/…` | change only if you moved the model files |

The camera source is auto-selected: on the Pi, `picamera2` imports, so the tools use
`PiCameraSource` automatically. Force it with `--camera picamera` if needed.

---

## 10. Configure — the **speech** channel

| Setting | Default | Note |
|---|---|---|
| `AUDIO.sample_rate_hz` | 16000 | Vosk small en-us expects 16 kHz mono |
| `AUDIO.channels` | 1 | see §5 if the ReSpeaker won't open as mono |
| `AUDIO.block_size` | 8000 | ~0.5 s chunks; lower for snappier partials |
| `AUDIO.device_name_hints` | `("ReSpeaker","ArrayUAC10","USB Audio")` | the mic is picked by matching these substrings |
| `PATHS.vosk_model_dir` | `models/vosk-model-small-en-us-0.15` | must exist (§8) |

If your ReSpeaker shows a different name in `arecord -l`, add that substring to
`AUDIO.device_name_hints`.

---

## 11. Run & test each channel

**Pick the mic (confirms name-based selection works):**
```bash
.venv/bin/python -m senseless.asr.mic_test --list-devices
```

**Speech test** (streams partial + final transcripts):
```bash
.venv/bin/python -m senseless.asr.mic_test          # auto-selects the ReSpeaker
.venv/bin/python -m senseless.asr.mic_test --device 2   # or force an index
```

**Sign — landmarks + color** (GUI; needs a display, or use `--video clip.mp4`):
```bash
.venv/bin/python -m senseless.sign.preview
```
Look for hands/pose dots tracking cleanly, `detected: yes`, correct L/R.

**Sign — throughput / FPS** (the key transfer number, §12):
```bash
.venv/bin/python -m senseless.eval.bench_perception --backend tasks --frames 300
```

**Sign — recognition** (headless prints prediction + FPS, no display needed):
```bash
.venv/bin/python -m senseless.sign.demo --headless
# with a display instead:  .venv/bin/python -m senseless.sign.demo
```

---

## 12. Running both channels together

The two channels are **decoupled** — you can run them as two processes:
```bash
# terminal 1
.venv/bin/python -m senseless.sign.demo --headless
# terminal 2
.venv/bin/python -m senseless.asr.mic_test
```
Mind the **4-core budget**: MediaPipe + the GRU take CPU, and Vosk takes more. If
they contend, lower `CAMERA` resolution or `SIGN.num_threads`. The future
`senseless.app` orchestrator will manage this properly (separate processes + bounded
drop-oldest queues) so latency stays bounded.

**FPS is the make-or-break transfer variable:** the model learned each sign over a
fixed **45-frame** window at your webcam's rate. If the Pi runs much slower, signs are
time-stretched and recognition drops. If §11's benchmark is far below ~30 fps: lower
`CAMERA.width/height`, keep the lite models, or raise `SIGN.frame_stride`, and re-test.

---

## 13. Autostart on boot (systemd)

Make a channel start automatically. Create
`/etc/systemd/system/senseless.service` (replace `<user>`):
```ini
[Unit]
Description=Senseless
After=multi-user.target

[Service]
Type=simple
User=<user>
WorkingDirectory=/home/<user>/senseless
ExecStart=/home/<user>/senseless/.venv/bin/python -m senseless.sign.demo --headless
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```
Enable + control it:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now senseless.service
systemctl status senseless.service
journalctl -u senseless.service -f     # live logs
```
Point `ExecStart` at `senseless.app` once the orchestrator exists.

---

## 14. Performance & thermal

- **Cooling:** the Pi 4 throttles when hot. Use a heatsink + fan; check temp with
  `vcgencmd measure_temp` and throttling with `vcgencmd get_throttled` (`0x0` = fine).
- **Headroom:** close the desktop / run headless (`sudo raspi-config` → Boot → console)
  to free RAM/CPU for the ML stages.
- **If FPS is low:** 640×480 → 480×360 in `CAMERA`, keep the *lite* pose model, and/or
  `SIGN.frame_stride = 2`.

---

## 15. Developing on the Pi

- The editable install means **edit a file (or `git pull`) → it takes effect on the
  next run** — no reinstall.
- Edit over SSH (`nano`/`vim`) or use VS Code **Remote-SSH** from your PC for a full
  editor on the Pi.
- Pull updates:
  ```bash
  cd ~/senseless && git pull
  # only re-run pip if requirements changed:
  .venv/bin/python -m pip install -e . -r requirements-pi.txt
  ```
- Don't run the trainer or the linters/tests as a workflow on the Pi — training is a
  dev-box/Colab job; the Pi is for inference.

---

## 16. Troubleshooting

| Symptom | Fix |
|---|---|
| Camera not detected | power off, reseat CSI ribbon (stripe toward USB side); `rpicam-hello --list-cameras` |
| Preview window won't open | headless has no display — use `demo --headless`, `preview --video clip.mp4`, or VNC |
| Colors look wrong / poor hand detection | `CAMERA.pixel_format = "BGR888"` in config.py |
| Left/right hands swapped | `SIGN.mirror = True` |
| Mic not found | `arecord -l`; add its name substring to `AUDIO.device_name_hints`, or `mic_test --device N` |
| `sounddevice`/PortAudio error | `sudo apt install -y libportaudio2` |
| ReSpeaker won't open as mono | see §5 — pick the right channel/device or bump `AUDIO.channels` |
| `picamera2`/`libcamera` import error | recreate the venv with `--system-site-packages` (§7) |
| Low / stuttering FPS | lower `CAMERA` resolution, `SIGN.frame_stride = 2`, add cooling (§14) |
| Recognition poor despite good FPS + tracking | collect a little data **on the Pi** and retrain including it (deployment-camera mismatch) |
| MediaPipe won't `pip install` | install the closest aarch64 version available; keep the same `.task` bundles (§8) |

---

## Quick reference

```bash
# one-time
sudo apt install -y git python3-venv python3-picamera2 libportaudio2
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e . -r requirements-pi.txt
# copy models/ over from the PC (see §8)

# verify
rpicam-hello --list-cameras
arecord -l

# run
.venv/bin/python -m senseless.asr.mic_test --list-devices
.venv/bin/python -m senseless.sign.preview
.venv/bin/python -m senseless.eval.bench_perception --backend tasks --frames 300
.venv/bin/python -m senseless.sign.demo --headless
```
