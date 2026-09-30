"""TDD specs for the pure inference-decoding logic in sign/classifier.py.

The TFLite interpreter is integration (validated against the real exported model
separately); only label loading + probability decoding are unit-tested here.
"""

import numpy as np

from senseless.sign.classifier import decode, load_labels


def test_decode_returns_argmax_label_and_confidence() -> None:
    word, conf = decode(np.array([0.1, 0.7, 0.2], dtype=np.float32), ["HELLO", "YES", "NO"], 0.6)
    assert word == "YES"
    assert abs(conf - 0.7) < 1e-6


def test_decode_below_threshold_returns_none() -> None:
    word, conf = decode(np.array([0.34, 0.33, 0.33], dtype=np.float32), ["HELLO", "YES", "NO"], 0.6)
    assert word is None
    assert abs(conf - 0.34) < 1e-6


def test_decode_at_threshold_is_accepted() -> None:
    word, _conf = decode(np.array([0.6, 0.4], dtype=np.float32), ["A", "B"], 0.6)
    assert word == "A"


def test_load_labels_strips_and_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text("HELLO\nYES\n\n  NO  \n", encoding="utf-8")
    assert load_labels(path) == ["HELLO", "YES", "NO"]


# --- interpret: one decision per onset-mode sign ---
from senseless.sign.classifier import interpret  # noqa: E402

LABELS = ["HELLO", "IDLE", "YES"]


def test_interpret_accepts_a_confident_word() -> None:
    assert interpret(np.array([0.9, 0.05, 0.05]), LABELS, 0.6, "IDLE") == ("HELLO", "HELLO", 0.9)


def test_interpret_rejects_low_confidence_but_keeps_the_best_guess() -> None:
    word, best, conf = interpret(np.array([0.3, 0.2, 0.5]), LABELS, 0.6, "IDLE")
    assert word is None
    assert best == "YES"
    assert np.isclose(conf, 0.5)


def test_interpret_never_accepts_the_idle_class() -> None:
    word, best, conf = interpret(np.array([0.02, 0.95, 0.03]), LABELS, 0.6, "IDLE")
    assert word is None
    assert best == "IDLE"
