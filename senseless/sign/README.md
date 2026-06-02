# sign/ — perception & landmark extraction

Camera frame → MediaPipe landmarks → normalized 153-dim feature vector
(`common/landmark_schema.py`). The temporal classifier consumes a
`(window_length, 153)` window of these vectors.

- `capture.py` — `FrameSource` interface; `OpenCVSource` (dev) and
  `PiCameraSource` (Pi) behind `open_frame_source()`.
- `landmarks.py` — the **pure** `frame_landmarks_to_vector` (unit-tested,
  numpy-only) plus the MediaPipe **perception backends**.

## Perception backend: decision

Two backends are benchmarked (per CLAUDE.md); both sit behind `PerceptionBackend`
and are selectable via `config.SIGN.perception_backend`:

| Backend | What it runs | Face cost |
| --- | --- | --- |
| **`tasks`** (default, **recommended**) | Tasks `HandLandmarker` + `PoseLandmarker` (lite), composed | **none** — face never computed |
| `holistic` | legacy `mp.solutions.Holistic`, face landmarks ignored | pays full face-mesh CPU, then discards it |

**Recommendation: `tasks`.** Legacy Holistic runs pose → **face mesh** → hands as
one graph with *no flag to skip the face stage*, so "Holistic ignoring face"
still burns face-mesh CPU we throw away — wasteful on a 4-core Pi where cores are
shared with ASR and the classifier. Tasks composes Hand + Pose with **no face at
all**, uses the supported API, and lets us pick lite models and VIDEO-mode
tracking. The trade-off (two graphs; hands not pose-ROI-guided) is expected to be
outweighed by the face-mesh savings — **confirm with the harness below** and flip
`perception_backend` if the numbers say otherwise.

> **Availability:** `holistic` needs the legacy `mp.solutions` API, present on the
> Pi's pinned mediapipe `0.10.14` but **removed in newer wheels** (e.g. `0.10.35`,
> the current Windows build) — there it raises a clear error and only `tasks`
> runs. `tasks` is supported everywhere, which is another reason it's the default.

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
python -m senseless.eval.bench_perception --frames 300          # both backends
python -m senseless.eval.bench_perception --backend tasks --source 0
```

Prints FPS and CPU% per backend. Record the result here once measured on the Pi.

## Gotchas

- **Handedness:** MediaPipe labels Left/Right from the image's perspective. If
  hands come out swapped for your camera, set `config.SIGN.mirror = True`.
- **picamera2 channel order:** the "RGB888" format can return BGR depending on
  libcamera/version. If colors look wrong on the Pi, flip
  `config.CAMERA.pixel_format` to "BGR888". Verify on-device.
