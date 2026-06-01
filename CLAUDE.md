# Senseless — project context for Claude Code

## Goal
On-device, real-time system on a Raspberry Pi 4B (8GB) that bridges Deaf↔hearing
communication via two independent channels:
1. Camera → MediaPipe Holistic landmarks → temporal classifier → ASL isolated WORD as text.
2. Microphone → Vosk → streaming speech-to-text.

## Locked technical decisions — do not relitigate these
- Target: isolated ASL words over a CURATED ~50–150 word vocabulary (not open-vocabulary).
- OS: Raspberry Pi OS 64-bit (Bookworm), STOCK kernel (no PREEMPT_RT). Soft real-time.
- Sign perception: MediaPipe Holistic with the FACE MESH EXCLUDED from features
  (hands + upper-body pose only). Benchmark Holistic-ignore-face vs Tasks Hand+Pose.
- Sign classifier: lightweight TEMPORAL model (GRU baseline) over a fixed ~30-frame
  landmark window. Target <1M params. Quantize to INT8 TFLite, run via XNNPACK.
- ASR: VOSK (small en-us model), streaming partial results for live display.
- Camera: Pi Camera Module 3 (CSI, picamera2). Mic: ReSpeaker USB Mic Array.
- Training happens on GOOGLE COLAB, never on the Pi. The Pi is inference-only.

## Hard constraints
- 4 CPU cores total — that is the parallelism budget. Use multiprocessing, not threads,
  for CPU-bound ML stages (GIL).
- Everything must run fully offline on the Pi. Models must be free / permissively licensed.
- Bound end-to-end latency: bounded ring buffers with a DROP-OLDEST policy, never
  unbounded queues.

## Repo layout
senseless/
  asr/            # Vosk streaming module
  sign/           # MediaPipe capture, landmark extraction, on-Pi inference
  collect/        # data-collection CLI
  notebooks/      # Colab training + quantization
  common/         # shared: queue, config, landmark schema, device selection
  ui/             # dual live-transcript display
  eval/           # accuracy/latency harness
  tests/

## Working style
- Use the superpowers brainstorm→plan→TDD flow for plumbing.
- For ML training code: smoke-test + held-out evaluation report instead of TDD on accuracy.
- Keep ASR and sign channels decoupled; they share only the common/ queue + config.
