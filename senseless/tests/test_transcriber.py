"""TDD specs for the pure Vosk-result parsing in asr/transcriber.py.

Only the framework-agnostic JSON parsing is unit-tested here. The Vosk recognizer
and the microphone are integration code, validated via asr/mic_test.py on the PC.
"""

from senseless.asr.transcriber import Transcript, parse_vosk_json


def test_final_result_is_marked_final() -> None:
    assert parse_vosk_json('{"text": "hello world"}') == Transcript("hello world", is_final=True)


def test_partial_result_is_marked_not_final() -> None:
    assert parse_vosk_json('{"partial": "hel"}') == Transcript("hel", is_final=False)


def test_empty_final_text() -> None:
    assert parse_vosk_json('{"text": ""}') == Transcript("", is_final=True)


def test_empty_partial() -> None:
    assert parse_vosk_json('{"partial": ""}') == Transcript("", is_final=False)


def test_text_key_wins_and_carries_word_details() -> None:
    # Vosk final results also include a "result" array; the "text" key marks final.
    payload = '{"result": [{"word": "hi"}], "text": "hi"}'
    assert parse_vosk_json(payload) == Transcript("hi", is_final=True)


def test_text_is_stripped() -> None:
    assert parse_vosk_json('{"text": "  hi there  "}') == Transcript("hi there", is_final=True)


def test_empty_object_is_empty_partial() -> None:
    assert parse_vosk_json("{}") == Transcript("", is_final=False)


def test_malformed_json_is_empty_partial() -> None:
    assert parse_vosk_json("not json at all") == Transcript("", is_final=False)
