"""Central configuration for Senseless.

ALL tunables live here. There should be no magic numbers anywhere else in the
codebase -- import the relevant frozen-dataclass singleton from this module
(AUDIO, CAMERA, SIGN, RUNTIME) or a path from PATHS instead.

Values are grouped by concern. Everything is a frozen dataclass so configuration
is immutable at runtime; override by constructing a new instance in a test or
entry point rather than mutating these.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Project paths ---------------------------------------------------------
# config.py lives at senseless/common/config.py, so the repo root is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"  # gitignored; populated on-device / from Colab
DATA_DIR = PROJECT_ROOT / "data"  # gitignored; recorded landmark windows / clips


@dataclass(frozen=True)
class AudioConfig:
    """Microphone capture + Vosk feed settings."""

    sample_rate_hz: int = 16_000  # Vosk small en-us expects 16 kHz mono
    channels: int = 1
    block_size: int = 8_000  # frames per audio callback (~0.5 s at 16 kHz)
    dtype: str = "int16"  # Vosk consumes 16-bit little-endian PCM
    # The ReSpeaker USB Mic Array enumerates under one of these name substrings;
    # device selection matches these case-insensitively against the device list.
    device_name_hints: tuple[str, ...] = ("ReSpeaker", "ArrayUAC10", "USB Audio")


@dataclass(frozen=True)
class CameraConfig:
    """Pi Camera Module 3 capture settings (picamera2)."""

    width: int = 640
    height: int = 480
    framerate: int = 30  # frames per second; window_length is sized against this
    pixel_format: str = "RGB888"  # picamera2 main-stream format fed to MediaPipe


@dataclass(frozen=True)
class SignConfig:
    """Temporal sign classifier windowing + inference settings."""

    window_length: int = 30  # frames per classification window (~1 s at 30 fps)
    frame_stride: int = 1  # frames advanced between consecutive captured frames
    inference_stride: int = 5  # re-run the classifier every N frames once full
    min_confidence: float = 0.6  # min softmax prob to emit a word
    num_threads: int = 2  # TFLite/XNNPACK threads (leave cores for camera + ASR)


@dataclass(frozen=True)
class RuntimeConfig:
    """Bounded ring-buffer sizes. DROP-OLDEST policy; never unbounded queues."""

    audio_queue_maxsize: int = 8
    frame_queue_maxsize: int = 4
    landmark_queue_maxsize: int = 4
    transcript_queue_maxsize: int = 32


@dataclass(frozen=True)
class ModelPaths:
    """On-disk locations of models/labels (all under MODELS_DIR, gitignored)."""

    vosk_model_dir: Path = MODELS_DIR / "vosk-model-small-en-us-0.15"
    sign_tflite: Path = MODELS_DIR / "sign_gru_int8.tflite"
    sign_labels: Path = MODELS_DIR / "sign_labels.txt"


# Module-level singletons: import these elsewhere.
AUDIO = AudioConfig()
CAMERA = CameraConfig()
SIGN = SignConfig()
RUNTIME = RuntimeConfig()
PATHS = ModelPaths()
