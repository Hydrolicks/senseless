"""TDD specs for the touchscreen app's pure state (senseless/ui/state.py)."""

from senseless.ui.state import (
    SignSentence,
    SpeechTranscript,
    TextScale,
    highlight_spans,
    words_to_sign,
)

ALIASES = (("thank you", "THANKYOU"), ("thanks", "THANKYOU"), ("hi", "HELLO"))


def test_sentence_adds_undoes_and_clears() -> None:
    s = SignSentence(max_words=8)
    s.add("HELLO")
    s.add("THANKYOU")
    assert s.text == "HELLO THANKYOU"
    assert s.undo() == "THANKYOU"
    assert s.words == ["HELLO"]
    s.clear()
    assert s.text == ""
    assert s.undo() is None


def test_sentence_keeps_only_the_newest_words() -> None:
    s = SignSentence(max_words=3)
    for w in ("A", "B", "C", "D"):
        s.add(w)
    assert s.words == ["B", "C", "D"]


def test_transcript_partial_is_replaced_and_cleared_by_a_final() -> None:
    t = SpeechTranscript(max_lines=50)
    t.add("hel", False)
    t.add("hello th", False)
    assert t.partial == "hello th" and t.lines == []
    t.add("hello there", True)
    assert t.lines == ["hello there"] and t.partial == ""


def test_transcript_empty_final_only_clears_the_partial() -> None:
    t = SpeechTranscript(max_lines=50)
    t.add("uh", False)
    t.add("", True)
    assert t.lines == [] and t.partial == ""


def test_transcript_keeps_the_newest_lines_and_clears() -> None:
    t = SpeechTranscript(max_lines=2)
    for line in ("one", "two", "three"):
        t.add(line, True)
    assert t.lines == ["two", "three"]
    t.add("par", False)
    t.clear()
    assert t.lines == [] and t.partial == ""


def test_text_scale_steps_within_bounds() -> None:
    s = TextScale(steps=(0.8, 1.0, 1.25, 1.5), index=1)
    assert s.factor == 1.0
    assert s.bigger() == 1.25
    assert s.bigger() == 1.5
    assert s.bigger() == 1.5  # capped
    for _ in range(5):
        s.smaller()
    assert s.factor == 0.8  # floored


def test_words_to_sign_matches_words_and_multiword_aliases() -> None:
    avail = {"HELLO", "HOW", "THANKYOU", "YES"}
    assert words_to_sign("Hello there", avail, ALIASES) == ["HELLO"]
    assert words_to_sign("thank you so much", avail, ALIASES) == ["THANKYOU"]
    assert words_to_sign("hi, how are you?", avail, ALIASES) == ["HELLO", "HOW"]
    assert words_to_sign("yes no", avail, ALIASES) == ["YES"]  # only words the library has


def test_words_to_sign_does_not_guess_inflections() -> None:
    assert words_to_sign("cats and dogs", {"CAT", "DOG"}, ALIASES) == []


def test_highlight_spans_point_at_the_matched_text() -> None:
    text = "well thank you and hello"
    spans = highlight_spans(text, {"THANKYOU", "HELLO"}, ALIASES)
    assert [text[s:e] for s, e in spans] == ["thank you", "hello"]
