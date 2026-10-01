"""Worker -> GUI events must survive a trip between processes (pickle)."""

import pickle

from senseless.common.config import PATHS, UI
from senseless.common.events import SignResult, SignStatus, SpeechText, WorkerError, WorkerReady


def test_events_round_trip_through_pickle() -> None:
    events = [
        SignStatus("capturing", 0.5, True, 7.2),
        SignResult("HELLO", "HELLO", 0.93),
        SignResult(None, "WANT", 0.41),
        SpeechText("hello there", True),
        WorkerReady("sign"),
        WorkerError("Camera not found."),
    ]
    for event in events:
        assert pickle.loads(pickle.dumps(event)) == event


def test_ui_config_values() -> None:
    assert (UI.width, UI.height) == (800, 480)
    assert UI.frames_queue_maxsize == 2
    assert UI.text_scales[UI.default_text_scale] == 1.0
    assert ("thank you", "THANKYOU") in UI.sign_aliases
    assert PATHS.sign_library.name == "sign_library.npz"
