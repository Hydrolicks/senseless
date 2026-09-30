# senseless/tests/test_ui_app.py
"""Smoke test of the Tk app: hidden window, no worker processes, synthetic events."""

import pytest

tk = pytest.importorskip("tkinter")

from senseless.common.events import (  # noqa: E402
    SignResult,
    SignStatus,
    SpeechText,
    WorkerError,
    WorkerReady,
)
from senseless.ui.app import SenselessApp  # noqa: E402


class FakeController:
    """Stands in for ModeController: records calls, runs nothing."""

    def __init__(self) -> None:
        self.mode = None
        self.started: list[str] = []

    def start(self, mode):
        self.mode = mode
        self.started.append(mode)

    def request_stop(self):
        pass

    def poll_stopped(self):
        self.mode = None
        return True

    def has_exited(self):
        return False

    def exitcode(self):
        return None

    def drain_events(self):
        return []

    def latest_frame(self):
        return None

    def shutdown(self, timeout=None):
        pass


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    application = SenselessApp(
        root, controller=FakeController(), start_workers=False, fullscreen=False
    )
    yield application
    root.destroy()


def test_sign_results_build_the_sentence_and_undo_removes_a_word(app) -> None:
    app.handle_event(WorkerReady("sign"))
    app.handle_event(SignStatus("capturing", 0.5, True, 7.0))
    app.handle_event(SignResult("HELLO", "HELLO", 0.93))
    app.handle_event(SignResult("THANKYOU", "THANKYOU", 0.88))
    assert app.word_var.get() == "THANKYOU"
    assert app.sentence_var.get() == "HELLO THANKYOU"
    app.undo()
    assert app.sentence_var.get() == "HELLO"


def test_a_rejected_sign_shows_the_best_guess_and_is_not_added(app) -> None:
    app.handle_event(SignResult(None, "WANT", 0.41))
    assert app.word_var.get() == "?"
    assert "WANT" in app.conf_var.get()
    assert app.sentence_var.get() == ""


def test_speech_text_appears_in_the_transcript(app) -> None:
    app.handle_event(SpeechText("hello th", False))
    app.handle_event(SpeechText("hello there", True))
    app.handle_event(SpeechText("how are", False))
    text = app.transcript_text.get("1.0", "end")
    assert "hello there" in text and "how are" in text


def test_a_worker_error_shows_a_banner_and_switching_modes_restarts_a_worker(app) -> None:
    app.handle_event(WorkerError("Camera not found."))
    assert app.banner_var.get() == "Camera not found."
    app.set_mode("speech")
    app._tick()  # one GUI tick: the (fake) worker has stopped -> the new mode starts
    assert app.ctl.started[-1] == "speech"
    assert app.banner_var.get() == ""
