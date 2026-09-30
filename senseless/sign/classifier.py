"""On-Pi sign classifier: run the INT8 TFLite GRU over a landmark window.

``load_labels`` + ``decode`` are pure (unit-tested). ``SignClassifier`` wraps the
TFLite interpreter and is lazy about it: it prefers ``ai_edge_litert`` (LiteRT,
the Pi runtime) and falls back to ``tensorflow.lite`` on the dev box. The model
and labels are produced by ``notebooks/train_gru.py`` into ``models/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from senseless.common.config import PATHS, SIGN


def load_labels(path: Path | str) -> list[str]:
    """Read a label-per-line file (blank lines skipped, whitespace stripped)."""
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def decode(probs: np.ndarray, labels: list[str], min_confidence: float) -> tuple[str | None, float]:
    """Turn class probabilities into ``(word, confidence)``.

    Returns the arg-max label when its probability meets ``min_confidence``,
    otherwise ``(None, confidence)`` so callers can suppress low-confidence hits.
    """
    index = int(np.argmax(probs))
    confidence = float(probs[index])
    word = labels[index] if confidence >= min_confidence else None
    return word, confidence


def interpret(
    probs: np.ndarray, labels: list[str], min_confidence: float, idle_label: str
) -> tuple[str | None, str, float]:
    """Decide what to show for one sign: ``(accepted_word | None, best_label, confidence)``.

    A word is accepted when it clears ``min_confidence`` and isn't the "not a sign"
    class; the best label is returned either way, for display as a rejected guess.
    """
    index = int(np.argmax(probs))
    best, confidence = labels[index], float(probs[index])
    accepted = best if confidence >= min_confidence and best != idle_label else None
    return accepted, best, confidence


def _make_interpreter(model_path: Path, num_threads: int):
    """Create a TFLite interpreter: LiteRT on the Pi, tensorflow.lite on dev."""
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter(model_path=str(model_path), num_threads=num_threads)
    except ImportError:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=str(model_path), num_threads=num_threads)


class SignClassifier:
    """Classifies one ``(window_length, FEATURE_DIM)`` window into a vocabulary word."""

    def __init__(
        self,
        model_path: Path | str = PATHS.sign_tflite,
        labels_path: Path | str = PATHS.sign_labels,
        min_confidence: float = SIGN.min_confidence,
        num_threads: int = SIGN.num_threads,
    ) -> None:
        self.labels = load_labels(labels_path)
        self.min_confidence = min_confidence
        self._interp = _make_interpreter(Path(model_path), num_threads)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]

    def probabilities(self, window: np.ndarray) -> np.ndarray:
        """Class probabilities (in ``labels`` order) for one landmark window."""
        x = np.asarray(window, dtype=self._in["dtype"])[None]
        self._interp.set_tensor(self._in["index"], x)
        self._interp.invoke()
        return self._interp.get_tensor(self._out["index"])[0]

    def predict(self, window: np.ndarray) -> tuple[str | None, float]:
        """Return ``(word, confidence)`` for one landmark window."""
        return decode(self.probabilities(window), self.labels, self.min_confidence)
