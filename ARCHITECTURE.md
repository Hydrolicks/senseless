# Senseless — Architecture

On-device, real-time Deaf↔hearing bridge for a Raspberry Pi 4B (8 GB) with two
**independent** channels that meet only at the UI:

1. **Sign → text:** camera → MediaPipe landmarks (face excluded) → temporal
   classifier → ASL isolated word.
2. **Speech → text:** microphone → Vosk → streaming transcript.

The locked technical decisions and hard constraints live in
[CLAUDE.md](CLAUDE.md); this document is the map of *how the code is organized*
and *how data flows* through it.

## Status legend

- ✅ implemented & tested
- 🟡 partial / scaffolded
- ⬜ planned (empty stub or not yet created)

## Data flow

### Sign channel
```
[Pi Camera Module 3 / dev webcam]
        │
        ▼
capture.py   FrameSource.read()                       ✅  RGB uint8 (H, W, 3)
        │
        ▼
landmarks.py PerceptionBackend.extract(frame, ts)     ✅  MediaPipe Tasks: Hand + Pose
        │                                                 → RawLandmarks
        │   left_hand (21,3)|None, right_hand (21,3)|None, pose (33,3)|None
        ▼
landmarks.py frame_landmarks_to_vector(raw)           ✅  body-anchored normalize
        │                                                 + zero-fill missing hands
        │   np.float32[153]   (the shared feature vector)
        ▼
common/ bounded ring buffer (window = 30)             ⬜  drop-oldest, multiprocessing
        │   (30, 153)
        ▼
sign classifier  GRU → INT8 TFLite (XNNPACK)          ⬜  word + confidence
        │
        ▼
ui/   dual live transcript                            ⬜
```

### Speech channel
```
[ReSpeaker USB Mic Array / dev mic]
        │
        ▼
asr/  Vosk streaming recognizer                       ✅  partial + final text
        │
        ▼
ui/   dual live transcript                            ⬜
```

The two channels are decoupled; they will share only `common/` (config + the
bounded queue). Each CPU-bound stage is intended to run in its own process
(4-core budget, GIL).

## Module map

| Path | Role | Status |
| --- | --- | --- |
| [common/config.py](senseless/common/config.py) | All tunables (camera, audio, sign windowing, confidences, model paths). No magic numbers elsewhere. | ✅ |
| [common/landmark_schema.py](senseless/common/landmark_schema.py) | The 153-dim per-frame feature-vector contract (offsets, pose indices, shoulder refs). | ✅ |
| [sign/capture.py](senseless/sign/capture.py) | `FrameSource` interface → `OpenCVSource` (dev) + `PiCameraSource` (Pi). | ✅ |
| [sign/landmarks.py](senseless/sign/landmarks.py) | Pure normalization (`frame_landmarks_to_vector`) + perception backends (`TasksBackend`, `HolisticBackend`). | ✅ |
| [sign/preview.py](senseless/sign/preview.py) | Dev-only OpenCV visualizer of the live pipeline. | ✅ |
| [sign/README.md](senseless/sign/README.md) | Sign-channel design doc (backend decision, normalization, policy). | ✅ |
| [eval/bench_perception.py](senseless/eval/bench_perception.py) | On-Pi FPS/CPU benchmark of the two backends. | ✅ |
| `common/` queue + device selection | Bounded drop-oldest ring buffer; device hints (hints partly in config). | ⬜ |
| sign classifier / inference | GRU → INT8 TFLite runner over a (30, 153) window. | ⬜ |
| [asr/transcriber.py](senseless/asr/transcriber.py) + [audio.py](senseless/asr/audio.py) + [mic_test.py](senseless/asr/mic_test.py) | Vosk streaming STT: pure result parser + recognizer wrapper + mic source + PC mic test. | ✅ |
| [collect/](senseless/collect/__init__.py) | Data-collection CLI (records landmark windows per word). | ⬜ |
| [ui/](senseless/ui/__init__.py) | Dual live-transcript display. | ⬜ |
| [notebooks/](senseless/notebooks/README.md) | Colab training + INT8 quantization. | ⬜ |
| [tests/](senseless/tests) | pytest suite (14 tests). | ✅ |

## Key contract: the feature vector

Defined once in `common/landmark_schema.py` so capture, training (Colab), and
on-Pi inference all agree. Face mesh is excluded by design.

```
[  0 :  63)  left hand    21 landmarks × (x, y, z)
[ 63 : 126)  right hand   21 landmarks × (x, y, z)
[126 : 153)  pose subset   9 landmarks × (x, y, z)   indices [0,11,12,13,14,15,16,23,24]
-------------------------------------------------------------------------------
FEATURE_DIM = 153
```

- **Normalization** (in `landmarks.py`): `p' = (p − shoulder_midpoint) / shoulder_width`
  applied to *all* landmarks → translation- and scale-invariant while preserving
  each hand's position relative to the body.
- **Missing-hand / no-detection policy:** an absent hand → its 63-dim block is
  zero; missing pose / degenerate shoulder width → the whole vector is zero. The
  same pure function is used at data-collection and inference time, so the model
  trains on identical zero-blocks.

See [sign/README.md](senseless/sign/README.md) for the full rationale, the
`tasks` vs `holistic` backend decision, and the on-Pi benchmark instructions.

## Suggested review order

Shared contracts first, then the logic that builds on them, then the tools, then
the tests that pin behavior.

1. [CLAUDE.md](CLAUDE.md) — locked decisions & constraints (the "why").
2. [common/config.py](senseless/common/config.py) — the tunables everything reads.
3. [common/landmark_schema.py](senseless/common/landmark_schema.py) — the data model.
4. [sign/README.md](senseless/sign/README.md) — design rationale for the sign channel.
5. [sign/landmarks.py](senseless/sign/landmarks.py) — **the core logic.** Read in two
   passes: (A) `RawLandmarks` + `frame_landmarks_to_vector` (pure), then
   (B) `PerceptionBackend` / `TasksBackend` / `HolisticBackend`.
6. [sign/capture.py](senseless/sign/capture.py) — camera I/O abstraction.
7. [sign/preview.py](senseless/sign/preview.py) — the whole pipeline wired together.
8. [eval/bench_perception.py](senseless/eval/bench_perception.py) — the same flow, instrumented.
9. [tests/test_landmarks.py](senseless/tests/test_landmarks.py) — the executable spec for
   the normalization invariants and missing-hand policy.

**Short on time?** `config.py` → `landmark_schema.py` → `landmarks.py` is ~90% of
the current logic.

## Environments

| Environment | Python | Role | Notes |
| --- | --- | --- | --- |
| Raspberry Pi 4B | 3.11 | inference only | `requirements-pi.txt`; MediaPipe `0.10.14` (has legacy `holistic`), NumPy 1.26.x |
| Dev PC (Windows) | 3.11 | code + tests + on-PC perception preview | `requirements-dev.txt`; MediaPipe `0.10.35` (no `mp.solutions`, only `tasks`), NumPy 2.x |
| Google Colab | — | training + INT8 quantization | `requirements-colab.txt`; never runs on the Pi |

The dev/Pi NumPy major-version divergence is harmless for the pure landmark math
(verified by the tests under both). The `holistic` backend only runs where the
legacy `mp.solutions` API exists (the Pi), and raises a clear error elsewhere.

## Testing & tooling

- `pytest` — 14 tests; `landmarks.py`'s pure path is TDD'd, the rest are import/interface smoke tests.
- `ruff` + `black` (config in [pyproject.toml](pyproject.toml)); hooks in
  [.pre-commit-config.yaml](.pre-commit-config.yaml).
- The MediaPipe backends and camera/GUI are not unit-tested (hardware/integration);
  they are validated via `preview.py` on the PC and `bench_perception.py` on the Pi.
