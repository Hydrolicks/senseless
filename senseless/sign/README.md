# sign/ — perception & landmark extraction

Camera frame → MediaPipe landmarks → normalized 153-dim feature vector
(`common/landmark_schema.py`). The temporal classifier consumes a
`(window_length, 153)` window of these vectors.

- `capture.py` — `FrameSource` interface; `OpenCVSource` (dev) and
  `PiCameraSource` (Pi) behind `open_frame_source()`.
- `landmarks.py` — the **pure** `frame_landmarks_to_vector` (unit-tested,
  numpy-only) plus the MediaPipe **perception backends**.

## Perception backend: decision

Three backends sit behind `PerceptionBackend`, selectable via
`config.SIGN.perception_backend` or `--backend` on every tool:

| Backend | What it runs | Face cost |
| --- | --- | --- |
| `tasks` (default) | Tasks `HandLandmarker` (full model) + `PoseLandmarker` (lite) | none |
| **`lite`** (**use on the Pi**) | legacy `mp.solutions` Hands + Pose at `model_complexity=0` | none |
| `holistic` | legacy `mp.solutions.Holistic`, face landmarks ignored | full face mesh, then discarded |

**Measured on the Pi 4** (Bookworm, mediapipe 0.10.18, Logitech webcam 640×480,
both hands in view; every model uses about one core):

| Model | ms / frame |
| --- | --- |
| Tasks hands (full) | 265 |
| **solutions hands, complexity 0 (lite)** | **95** |
| solutions hands, complexity 1 | 216 |
| pose lite (Tasks or solutions) | ~100 |
| Holistic, complexity 0 / 1 | 264 / 313 |

So on the Pi: **`lite`** (2.8× faster hands than `tasks`). Holistic is no faster
because it always runs the face mesh. The dev PC's mediapipe `0.10.35` has no
`mp.solutions`, so `lite`/`holistic` raise a clear error there and only `tasks`
runs; the Pi's `0.10.18` runs all three.

> **Handedness:** `mp.solutions` Hands labels handedness as if the image were
> mirrored; the Tasks model doesn't. `LiteBackend` inverts it so a hand lands in the
> same left/right slot as with `tasks` (checked on the same image under both
> mediapipe versions). The feature vector is therefore compatible across backends;
> only precision differs (lite vs full hand model: ~2–5% of hand size per landmark
> on a test image).

## Time-based window

The model learned each sign as 45 frames at ~30 FPS (`SIGN.reference_fps`), a span
of 44/30 s. At the Pi's ~10 FPS, "the last 45 frames" would be 4.5 s of motion, so
the demo and the recorder use `sign/window.py` instead: frames are kept with their
capture timestamps and the newest span is resampled to 45 evenly spaced steps at
the reference rate. Each block (left hand, right hand, pose) is interpolated
linearly only when present in both neighbouring frames; otherwise the nearer frame
is used, so a hand is never blended with the all-zero "absent" block. The
classifier runs at most every `SIGN.inference_interval_s` (0.15 s), not every N
frames.

Simulated on the 128 held-out test samples (recorded at 30 FPS, subsampled with
random phase and timing jitter, then resampled), the model trained without any
low-FPS data keeps most of its accuracy: 97.7% at 30 FPS, 97.3% at 15 and 10 FPS,
96.7% at 7.5 FPS (training seed 1).

## Low-FPS training augmentation

`notebooks/train_gru.py` adds `--lowfps-copies` (default 2) simulated low-frame-rate
versions of every training window, each at a random rate between `--min-fps` (6)
and 30 FPS, built with `window.simulate_capture` (the same resampling as inference).
Validation and test data stay unaugmented; the trainer also reports test accuracy
at a simulated `--eval-fps` (10).

Averaged over 3 training seeds on the same split:

| Test accuracy | 30 FPS | 10 FPS | 7.5 FPS | 6 FPS |
| --- | --- | --- | --- | --- |
| no augmentation | 96.6% | 95.2% | 94.6% | 94.4% |
| low-FPS augmentation ×2 | 96.6% | 96.6% | 96.9% | 95.9% |

No cost at 30 FPS, +1.4 to +2.3 points at Pi frame rates, and much less spread
between seeds (93.0–97.3% vs 95.9–97.7% at 10 FPS). One seed alone is noisy: each
test sample is 0.8%.

## Parallel perception (use on the Pi)

`--parallel` (or `SIGN.parallel_perception = True`) wraps the `tasks` or `lite`
backend in `sign/parallel.py`'s `ParallelBackend`: the pose model and the hands model
each run in their own spawned process, so a frame costs max(pose, hands) instead of
their sum. Frames reach the workers through shared memory; the landmarks come back
over `DropOldestQueue`s. Parallel output is identical to serial output (checked with
real models). Measured speed-up on the dev PC: ×1.5–1.9; on the Pi 4, where pose
(~100 ms) and lite hands (~95 ms) are about equal, close to ×2 is expected.
`SIGN.pose_stride = 2` runs pose on every other frame and reuses it in between.

The live demo reads the camera through `capture.LatestFrameGrabber`, which keeps only
the newest frame, so a ~10 FPS consumer never lags behind a 30 FPS camera's buffer.

```bash
python -m senseless.sign.demo --backend lite --parallel --camera opencv --source 0 --headless
python -m senseless.eval.bench_perception --backend lite --parallel --camera opencv
```

## Normalization (body-anchored)

`p' = (p − shoulder_midpoint) / shoulder_width`, applied to **all** landmarks
(both hands + the 9 pose points). Origin = shoulder midpoint (indices 11/12);
scale = shoulder width in the image (x, y) plane. This is **translation-invariant**
(re-origin removes where the signer is in frame) and **scale-invariant** (÷ width
removes camera distance / body size). Shoulders are the reference because they
stay put while arms move, and one shared transform keeps each hand's position
*relative to the body* — the sign's "location". `z` is scaled by the same factor
but is the weakest channel (hand-z is wrist-relative, pose-z hip-relative).

## Missing-hand / no-detection policy (zero-fill)

Absent hand → its 63-dim block is **exactly zero**. Missing pose / degenerate
shoulder width → the **whole 153-vector is zero** ("no detection"). Stateless, and
the data-collection + Colab training pipeline import the *same*
`frame_landmarks_to_vector`, so train and inference see identical zero-blocks.

## Models (downloaded into `models/`, gitignored)

```bash
mkdir -p models
curl -L -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Paths are set in `config.PATHS` (`pose_landmarker_task`, `hand_landmarker_task`).

## Benchmark (run on the Pi)

```bash
python -m senseless.eval.bench_perception --frames 300                     # every available backend
python -m senseless.eval.bench_perception --backend lite --camera opencv --source 0
```

Prints FPS and CPU% per backend. Keep your hands in view while it runs: tracking
two hands costs more than an empty frame. The `lite` pose model is fetched into
the mediapipe package the first time it runs, so do that once while online.

## Gotchas

- **Handedness:** MediaPipe labels Left/Right from the image's perspective. If
  hands come out swapped for your camera, set `config.SIGN.mirror = True`.
- **picamera2 channel order:** the "RGB888" format can return BGR depending on
  libcamera/version. If colors look wrong on the Pi, flip
  `config.CAMERA.pixel_format` to "BGR888". Verify on-device.
