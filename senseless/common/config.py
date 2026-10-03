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

    window_length: int = 45  # model input steps per window (~1.5 s at reference_fps)
    # Frame rate the training windows were recorded at. Windows are time-based
    # (sign/window.py): whatever the live FPS, the last (window_length - 1) /
    # reference_fps seconds are resampled to window_length steps at this rate.
    reference_fps: float = 30.0
    inference_interval_s: float = 0.15  # re-run the classifier at most this often
    # Live demo "onset" mode (sign/segmenter.py): one classification per sign.
    onset_confirm_frames: int = 2  # consecutive hand frames needed to start a capture
    release_s: float = 0.3  # hands must be out of view this long before the next sign
    min_confidence: float = 0.6  # min softmax prob to emit a word
    idle_label: str = "IDLE"  # the "not a sign" class: never shown as a word
    num_threads: int = 2  # TFLite/XNNPACK threads (leave cores for camera + ASR)
    # Normalization: minimum shoulder width (in normalized image units). Below
    # this the body frame is degenerate and the frame is emitted as all-zeros.
    normalization_eps: float = 1e-6
    # Perception backend (see sign/landmarks.py): "tasks" (Tasks Hand + Pose, full
    # hand model; the only one on the dev PC's mediapipe 0.10.35), "lite" (legacy
    # mp.solutions Hands + Pose, lite models; 2.8x faster hands on a Pi 4, needs
    # mediapipe 0.10.18), or "holistic" (legacy, runs the face mesh; benchmark only).
    perception_backend: str = "tasks"
    lite_model_complexity: int = 0  # "lite" backend: 0 = lite models (fast), 1 = full
    # Run the pose and hands models in separate worker processes, one core each
    # (sign/parallel.py). On the Pi 4 a frame then costs max(pose, hands) instead of
    # their sum. Only "tasks" and "lite" can be split. Tools: --parallel.
    parallel_perception: bool = False
    pose_stride: int = 1  # parallel only: run pose every Nth frame, reuse it in between
    min_hand_detection_confidence: float = 0.5
    min_pose_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Flip if left/right hands come out swapped for your camera (MediaPipe labels
    # handedness from the image's perspective).
    mirror: bool = False
    holistic_model_complexity: int = 1  # Holistic backend only: 0 (fast)..2 (accurate)


@dataclass(frozen=True)
class RuntimeConfig:
    """Bounded ring-buffer sizes. DROP-OLDEST policy; never unbounded queues."""

    audio_queue_maxsize: int = 8
    frame_queue_maxsize: int = 4
    landmark_queue_maxsize: int = 4
    transcript_queue_maxsize: int = 32


@dataclass(frozen=True)
class UIConfig:
    """Touchscreen app (senseless.ui): Tkinter on the Pi's 7" 800x480 display."""

    width: int = 800
    height: int = 480
    poll_ms: int = 40  # the GUI drains the worker queues this often
    preview_size: tuple[int, int] = (320, 240)  # (width, height) of the camera preview
    frames_queue_maxsize: int = 2  # preview images: newest wins
    switch_timeout_s: float = 5.0  # force-stop a worker that won't exit
    sentence_max_words: int = 8
    transcript_max_lines: int = 50
    text_scales: tuple[float, ...] = (0.8, 1.0, 1.25, 1.5)
    default_text_scale: int = 1  # index into text_scales
    # Sign worker: the Logitech webcam, lite models in parallel (fastest on a Pi 4).
    # Where MediaPipe has no mp.solutions (the dev PC) the app uses "tasks" instead.
    camera: str = "opencv"
    camera_source: int = 0
    perception_backend: str = "lite"
    parallel_perception: bool = True
    # Spoken phrase -> vocabulary label, for the signing figure.
    sign_aliases: tuple[tuple[str, str], ...] = (
        ("thank you", "THANKYOU"),
        ("thanks", "THANKYOU"),
        ("hi", "HELLO"),
        ("bye", "GOODBYE"),
        ("good bye", "GOODBYE"),
    )
    # Figure view window in normalized units (shoulder widths): x_min, x_max, y_min, y_max.
    figure_extent: tuple[float, float, float, float] = (-1.8, 1.8, -2.0, 2.8)
    figure_tick_ms: int = 33  # playback frame interval (~30 FPS)
    # Words waiting for the figure (~1.5 s each): room for a whole spoken sentence, signed
    # in order with repeats; beyond this the oldest words (older sentences) are dropped.
    sign_queue_max: int = 12
    # Palette (deck colours).
    bg: str = "#0F2A31"
    panel: str = "#0B2227"
    text: str = "#EAF3F4"
    muted: str = "#8FB3B8"
    teal: str = "#0E7C86"
    coral: str = "#E4572E"
    word: str = "#7FD6C8"


@dataclass(frozen=True)
class ModelPaths:
    """On-disk locations of models/labels (all under MODELS_DIR, gitignored)."""

    vosk_model_dir: Path = MODELS_DIR / "vosk-model-small-en-us-0.15"
    sign_tflite: Path = MODELS_DIR / "sign_gru_int8.tflite"
    sign_labels: Path = MODELS_DIR / "sign_labels.txt"
    # MediaPipe Tasks model bundles (downloaded into MODELS_DIR; see sign/README).
    pose_landmarker_task: Path = MODELS_DIR / "pose_landmarker_lite.task"
    hand_landmarker_task: Path = MODELS_DIR / "hand_landmarker.task"
    # One representative take per word for the Speech-mode signing figure
    # (built by `python -m senseless.sign.library`).
    sign_library: Path = MODELS_DIR / "sign_library.npz"


# Module-level singletons: import these elsewhere.
AUDIO = AudioConfig()
CAMERA = CameraConfig()
SIGN = SignConfig()
RUNTIME = RuntimeConfig()
PATHS = ModelPaths()
UI = UIConfig()
