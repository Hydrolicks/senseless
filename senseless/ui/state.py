"""Pure state for the touchscreen app: no Tk, no processes, fully unit-tested.

* ``SignSentence``   - recognized sign words (add / undo / clear, capped).
* ``SpeechTranscript`` - Vosk finals as lines plus the live partial line.
* ``TextScale``      - bounded text-size steps for the A- / A+ buttons.
* ``words_to_sign`` / ``highlight_spans`` - which spoken words the signing figure
  can show (vocabulary words and aliases such as "thank you" -> THANKYOU).
"""

from __future__ import annotations

import re
from collections.abc import Collection

from senseless.common.config import UI

_TOKEN = re.compile(r"[a-z']+")


class SignSentence:
    """The running sentence of recognized sign words."""

    def __init__(self, max_words: int | None = None) -> None:
        self.max_words = UI.sentence_max_words if max_words is None else max_words
        self.words: list[str] = []

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def add(self, word: str) -> None:
        self.words = (self.words + [word])[-self.max_words :]

    def undo(self) -> str | None:
        return self.words.pop() if self.words else None

    def clear(self) -> None:
        self.words = []


class SpeechTranscript:
    """Final lines plus the current partial (which the next result replaces)."""

    def __init__(self, max_lines: int | None = None) -> None:
        self.max_lines = UI.transcript_max_lines if max_lines is None else max_lines
        self.lines: list[str] = []
        self.partial = ""

    def add(self, text: str, is_final: bool) -> None:
        if not is_final:
            self.partial = text
            return
        self.partial = ""
        if text:
            self.lines = (self.lines + [text])[-self.max_lines :]

    def clear(self) -> None:
        self.lines = []
        self.partial = ""


class TextScale:
    """Text-size factor, stepped by A- / A+ within ``steps``."""

    def __init__(self, steps: tuple[float, ...] | None = None, index: int | None = None) -> None:
        self.steps = UI.text_scales if steps is None else steps
        self.index = UI.default_text_scale if index is None else index

    @property
    def factor(self) -> float:
        return self.steps[self.index]

    def bigger(self) -> float:
        self.index = min(self.index + 1, len(self.steps) - 1)
        return self.factor

    def smaller(self) -> float:
        self.index = max(self.index - 1, 0)
        return self.factor


def _match(
    text: str, available: Collection[str], aliases: tuple[tuple[str, str], ...]
) -> list[tuple[str, int, int]]:
    """(label, start, end) for each signable word or alias phrase, left to right."""
    tokens = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text.lower())]
    phrases = sorted(
        ((tuple(phrase.split()), label) for phrase, label in aliases), key=lambda p: -len(p[0])
    )
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(tokens):
        for words, label in phrases:
            n = len(words)
            if tuple(tok for tok, _, _ in tokens[i : i + n]) == words:
                if label in available:
                    out.append((label, tokens[i][1], tokens[i + n - 1][2]))
                i += n
                break
        else:
            word, start, end = tokens[i]
            label = word.replace("'", "").upper()
            if label in available:
                out.append((label, start, end))
            i += 1
    return out


def words_to_sign(
    text: str, available: Collection[str], aliases: tuple[tuple[str, str], ...] = UI.sign_aliases
) -> list[str]:
    """Vocabulary labels to sign for a spoken line (case/punctuation-insensitive)."""
    return [label for label, _, _ in _match(text, available, aliases)]


def highlight_spans(
    text: str, available: Collection[str], aliases: tuple[tuple[str, str], ...] = UI.sign_aliases
) -> list[tuple[int, int]]:
    """Character spans in ``text`` of the words that ``words_to_sign`` would sign."""
    return [(start, end) for _, start, end in _match(text, available, aliases)]
