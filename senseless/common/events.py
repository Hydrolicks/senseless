"""Events sent from a mode worker process to the touchscreen GUI.

Small frozen dataclasses: they travel through a multiprocessing queue, so they
must pickle. Camera preview images are not events; they go on a separate
drop-oldest queue so a burst of frames can never evict a recognized word.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerReady:
    """The worker's devices and models are up; ``mode`` is "sign" or "speech"."""

    mode: str


@dataclass(frozen=True)
class WorkerError:
    """A failure the user should see (camera missing, model file missing, ...)."""

    message: str


@dataclass(frozen=True)
class SignStatus:
    """Per-frame sign pipeline status: segmenter state, capture progress 0..1, hands, FPS."""

    state: str
    progress: float
    hands: bool
    fps: float


@dataclass(frozen=True)
class SignResult:
    """One classified sign. ``word`` is None when rejected (low confidence or IDLE)."""

    word: str | None
    best: str
    confidence: float


@dataclass(frozen=True)
class SpeechText:
    """A Vosk result: a changing partial or a final line."""

    text: str
    is_final: bool
