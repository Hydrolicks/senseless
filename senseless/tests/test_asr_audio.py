"""Light import/interface checks for the ASR audio source + mic-test script.

sounddevice and Vosk are lazy-imported, so these modules must import without them.
"""

import pytest

from senseless.asr import audio, mic_test


def test_modules_import_without_audio_deps() -> None:
    assert hasattr(audio, "MicrophoneSource")
    assert hasattr(audio, "find_input_device")
    assert hasattr(mic_test, "main")


def test_audio_source_is_abstract() -> None:
    with pytest.raises(TypeError):
        audio.AudioSource()  # type: ignore[abstract]
