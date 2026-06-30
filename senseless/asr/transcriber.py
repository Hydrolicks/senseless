"""Streaming speech-to-text via Vosk.

Two layers, mirroring the sign channel:

* ``Transcript`` + ``parse_vosk_json`` -- a PURE parser (no Vosk import) that
  turns Vosk's JSON output into a small dataclass. Unit-tested.
* ``VoskTranscriber`` -- a thin wrapper around ``vosk.KaldiRecognizer`` that
  lazy-imports Vosk, so importing this module (and the parser tests) never needs
  Vosk installed. Integration code; validated via ``asr/mic_test.py``.

Vosk emits ``{"partial": "..."}`` while an utterance is in progress and
``{"text": "..."}`` when it finalizes one, so the presence of the ``text`` key is
what marks a result final.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from senseless.common.config import AUDIO, PATHS


@dataclass(frozen=True)
class Transcript:
    """One recognizer result: the text and whether the utterance is finalized."""

    text: str
    is_final: bool


def parse_vosk_json(payload: str) -> Transcript:
    """Parse a Vosk result/partial JSON string into a ``Transcript``.

    Robust to malformed input: anything unparseable becomes an empty partial.
    """
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return Transcript("", is_final=False)
    if isinstance(data, dict) and "text" in data:
        return Transcript(str(data["text"]).strip(), is_final=True)
    partial = data.get("partial", "") if isinstance(data, dict) else ""
    return Transcript(str(partial).strip(), is_final=False)


class VoskTranscriber:
    """Feeds PCM audio to a Vosk recognizer and returns ``Transcript`` results."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        sample_rate_hz: int | None = None,
        log_level: int = -1,
    ) -> None:
        import vosk

        vosk.SetLogLevel(log_level)  # -1 silences Vosk's verbose stderr logging
        model_dir = Path(model_dir or PATHS.vosk_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Vosk model not found at {model_dir}. Download the small en-us "
                "model and unzip it there (see instructions.md, step 4)."
            )
        self._model = vosk.Model(str(model_dir))
        self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate_hz or AUDIO.sample_rate_hz)

    def accept(self, pcm: bytes) -> Transcript:
        """Feed one block of 16-bit mono PCM; return a final or partial result."""
        if self._recognizer.AcceptWaveform(pcm):
            return parse_vosk_json(self._recognizer.Result())
        return parse_vosk_json(self._recognizer.PartialResult())

    def final(self) -> Transcript:
        """Flush and return the final transcript for the current utterance."""
        return parse_vosk_json(self._recognizer.FinalResult())
