"""Speech-mode worker for the touchscreen app (runs in its own process).

Microphone -> Vosk -> ``SpeechText`` events: a partial only when it changed,
and each final line. Errors become ``WorkerError`` events the GUI shows with a
Retry button. Parts are injectable factories so tests run the real loop.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from senseless.asr import audio
from senseless.common.events import SpeechText, WorkerError, WorkerReady
from senseless.common.process import parent_alive


@dataclass
class SpeechParts:
    """Factories for the microphone and the recognizer; swapped for fakes in tests."""

    source: Callable[[], audio.AudioSource]
    transcriber: Callable[[], Any]  # -> object with .accept(pcm) -> Transcript


def default_speech_parts() -> SpeechParts:
    from senseless.asr.transcriber import VoskTranscriber

    return SpeechParts(
        source=lambda: audio.MicrophoneSource(device=audio.find_input_device()),
        transcriber=VoskTranscriber,
    )


def run_speech_worker(events, stop, parts: SpeechParts | None = None) -> None:
    """Run microphone -> Vosk until ``stop`` is set or something fails."""
    try:
        parts = parts if parts is not None else default_speech_parts()
        transcriber = parts.transcriber()
    except BaseException as exc:  # noqa: BLE001 -- shown to the user
        traceback.print_exc()
        events.put(WorkerError(f"Speech model could not load: {exc}"))
        return
    try:
        mic = parts.source()
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        events.put(
            WorkerError(f"Microphone could not open. Check the USB cable, then tap Retry. ({exc})")
        )
        return

    shown_partial = ""
    try:
        with mic:
            events.put(WorkerReady("speech"))
            while not stop.is_set() and parent_alive():  # parent gone: don't hold the mic
                block = mic.read()
                if block is None:
                    events.put(WorkerError("Microphone stream ended."))
                    return
                result = transcriber.accept(block)
                if result.is_final:
                    if result.text or shown_partial:
                        events.put(SpeechText(result.text, True))
                    shown_partial = ""
                elif result.text != shown_partial:
                    events.put(SpeechText(result.text, False))
                    shown_partial = result.text
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        events.put(WorkerError(f"Speech engine error: {exc}"))
