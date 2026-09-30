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
        self.shutdowns = 0
        self.exited = False
        self.code: int | None = None
        self.drain_error: Exception | None = None  # raised once by drain_events
        self.events: list = []  # returned (once) by drain_events

    def start(self, mode):
        self.mode = mode
        self.started.append(mode)

    def request_stop(self):
        pass

    def poll_stopped(self):
        self.mode = None
        return True

    def has_exited(self):
        return self.exited

    def exitcode(self):
        return self.code

    def drain_events(self):
        if self.drain_error is not None:
            error, self.drain_error = self.drain_error, None
            raise error
        events, self.events = self.events, []
        return events

    def latest_frame(self):
        return None

    def shutdown(self, timeout=None):
        self.shutdowns += 1


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    application = SenselessApp(
        root,
        controller=FakeController(),
        start_workers=False,
        fullscreen=False,
        load_library_file=False,
    )
    yield application
    try:
        root.destroy()
    except tk.TclError:  # a test already closed the window
        pass


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


def test_an_exception_in_a_tick_shows_a_banner_and_the_loop_keeps_running(app, capsys) -> None:
    app.ctl.drain_error = RuntimeError("boom")
    scheduled = []
    app.root.after = lambda ms, fn: scheduled.append(fn)  # capture the reschedule
    app._tick()
    assert "boom" in app.banner_var.get()
    assert "boom" in capsys.readouterr().err  # traceback logged
    assert scheduled == [app._tick]  # still rescheduled despite the exception
    app.hide_error()
    app._tick()  # the next tick runs normally
    assert app.banner_var.get() == ""
    assert len(scheduled) == 2


def test_an_unexpected_worker_exit_shows_a_banner_with_the_exit_code(app) -> None:
    app.ctl.exited = True
    app.ctl.code = 3
    app._tick()
    assert "exit code 3" in app.banner_var.get()


def test_closing_the_window_runs_exit_app_and_stops_the_worker(app) -> None:
    command = app.root.protocol("WM_DELETE_WINDOW")
    assert command
    app.root.tk.call(command)  # what the window manager does on close
    assert app.ctl.shutdowns == 1
    assert not app._alive


import numpy as np  # noqa: E402

from senseless.common import landmark_schema as ls  # noqa: E402


def _library() -> dict:
    frame = np.zeros(ls.FEATURE_DIM, dtype=np.float32)
    frame[ls.POSE_START : ls.POSE_END] = 0.3
    frame[ls.LEFT_HAND_START : ls.LEFT_HAND_END] = 0.5
    return {"HELLO": np.stack([frame] * 45), "THANKYOU": np.stack([frame] * 45)}


@pytest.fixture
def speech_app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    application = SenselessApp(
        root,
        controller=FakeController(),
        start_workers=False,
        fullscreen=False,
        initial_mode="speech",
        library=_library(),
    )
    yield application
    try:
        root.destroy()
    except tk.TclError:
        pass


def test_a_final_line_queues_its_vocabulary_words_for_the_figure(speech_app) -> None:
    speech_app.handle_event(SpeechText("hello and thank you", True))
    queued = list(speech_app.sign_queue) + [speech_app._playing]
    assert "HELLO" in queued and "THANKYOU" in queued


def test_partials_do_not_trigger_signing(speech_app) -> None:
    speech_app.handle_event(SpeechText("hello", False))
    assert not speech_app.sign_queue and speech_app._playing is None


def test_signed_words_are_highlighted(speech_app) -> None:
    speech_app.handle_event(SpeechText("well hello there", True))
    ranges = speech_app.transcript_text.tag_ranges("signed")
    assert ranges
    assert speech_app.transcript_text.get(ranges[0], ranges[1]) == "hello"


def test_the_figure_plays_queued_words_in_order_then_rests(speech_app) -> None:
    speech_app.root.after = lambda ms, fn: None  # drive the animation by hand
    speech_app.handle_event(SpeechText("hello thank you", True))
    speech_app._play_tick()
    assert speech_app._playing == "HELLO"
    assert speech_app.figure_caption.get() == "HELLO"
    assert speech_app.figure_canvas.find_all()  # a frame was drawn
    speech_app._play_t0 -= 10.0  # the take is long over
    speech_app._play_tick()  # HELLO finishes
    assert speech_app._playing is None
    speech_app._play_tick()  # the next word starts on the following tick
    assert speech_app._playing == "THANKYOU"
    speech_app._play_t0 -= 10.0
    speech_app._play_tick()
    assert speech_app._playing is None and speech_app.figure_caption.get() == ""
    assert speech_app.figure_canvas.find_all()  # resting pose is drawn


def test_without_a_library_the_figure_is_hidden_and_speech_still_works(app) -> None:
    app.handle_event(SpeechText("hello there", True))
    assert not app.sign_queue and not app.transcript_text.tag_ranges("signed")
    assert "hello there" in app.transcript_text.get("1.0", "end")
    assert not app.figure_panel.winfo_manager()  # the inset is not shown
    assert "senseless.sign.library" in app.library_note.cget("text")


def test_a_missing_library_file_is_tolerated(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError("no sign_library.npz")

    monkeypatch.setattr("senseless.sign.library.load_library", missing)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    try:
        application = SenselessApp(
            root, controller=FakeController(), start_workers=False, fullscreen=False
        )
        assert application.library == {}
    finally:
        root.destroy()


def test_a_bad_figure_frame_does_not_kill_the_animation(speech_app, capsys) -> None:
    scheduled = []
    speech_app.root.after = lambda ms, fn: scheduled.append(fn)
    speech_app.library["HELLO"] = None  # a malformed take
    speech_app.sign_queue.append("HELLO")
    speech_app._play_tick()
    assert "Traceback" in capsys.readouterr().err
    assert scheduled == [speech_app._play_tick]  # still rescheduled
    assert speech_app._playing is None  # the bad word is dropped, not retried forever
    speech_app._play_tick()  # the next frame runs normally
    assert len(scheduled) == 2


def test_the_figure_animation_stops_after_exit(speech_app) -> None:
    scheduled = []
    speech_app.root.after = lambda ms, fn: scheduled.append(fn)
    speech_app._alive = False
    speech_app._play_tick()
    assert scheduled == []


def test_speech_layout_keeps_the_clear_row_and_reflows_the_transcript(speech_app) -> None:
    view = speech_app.speech_view
    speech_app.root.deiconify()  # a withdrawn window is never laid out
    speech_app.root.update()
    speech_app.root.withdraw()
    order = view.pack_slaves()
    row = order[0]
    assert row.pack_info()["side"] == "bottom"  # packed first, so it always gets its height
    assert speech_app.figure_panel.pack_info()["side"] == "right"
    assert speech_app.figure_panel.pack_info()["anchor"] == "n"
    assert speech_app.transcript_text.pack_info()["side"] == "left"
    assert order.index(speech_app.figure_panel) < order.index(speech_app.transcript_text)
    clear = row.winfo_children()[0]
    assert row.winfo_height() >= row.winfo_reqheight()
    assert clear.winfo_height() >= clear.winfo_reqheight()
    # the transcript no longer runs underneath the figure
    text, panel = speech_app.transcript_text, speech_app.figure_panel
    assert text.winfo_rootx() + text.winfo_width() <= panel.winfo_rootx()


def test_without_a_library_the_transcript_is_full_width_with_the_note(app) -> None:
    app.root.update_idletasks()
    assert not app.figure_panel.winfo_manager()
    assert app.library_note.winfo_manager()
    assert app.transcript_text.pack_info()["side"] == "left"


def test_the_figure_queue_is_bounded_and_drops_the_oldest_word(speech_app) -> None:
    from senseless.common.config import UI

    assert speech_app.sign_queue.maxlen == UI.sign_queue_max == 3
    speech_app.handle_event(SpeechText("hello thank you hello thank you", True))
    assert list(speech_app.sign_queue) == ["THANKYOU", "HELLO", "THANKYOU"]  # first HELLO dropped


def test_clear_in_speech_mode_stops_the_figure_and_empties_its_queue(speech_app) -> None:
    speech_app.root.after = lambda ms, fn: None
    speech_app.handle_event(SpeechText("hello thank you", True))
    speech_app._play_tick()
    assert speech_app._playing == "HELLO" and speech_app.sign_queue
    speech_app.clear()
    assert speech_app._playing is None and not speech_app.sign_queue
    assert speech_app.figure_caption.get() == ""
    assert speech_app._rest_drawn is False  # the resting pose is redrawn on the next tick
    speech_app._play_tick()
    assert speech_app._playing is None and speech_app._rest_drawn


def test_switching_to_sign_mode_resets_the_figure(speech_app) -> None:
    speech_app.root.after = lambda ms, fn: None
    speech_app.handle_event(SpeechText("hello thank you", True))
    speech_app._play_tick()
    speech_app.set_mode("sign")
    speech_app._tick()  # the fake worker has stopped -> the switch completes
    assert speech_app.mode == "sign"
    assert speech_app._playing is None and not speech_app.sign_queue
    assert speech_app.figure_caption.get() == ""


def test_the_figure_does_not_redraw_while_sign_mode_is_showing(speech_app) -> None:
    speech_app.root.after = lambda ms, fn: None
    speech_app.set_mode("sign")
    speech_app._tick()
    speech_app.figure_canvas.delete("all")
    speech_app._play_tick()
    assert not speech_app.figure_canvas.find_all()


def _build_with_library_file(monkeypatch, path, capsys):
    from senseless.sign import library as sign_library

    original = sign_library.load_library
    monkeypatch.setattr(sign_library, "load_library", lambda: original(path))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    try:
        application = SenselessApp(
            root, controller=FakeController(), start_workers=False, fullscreen=False
        )
        assert application.library == {}
        assert application._rest is None
        assert application.library_note.winfo_manager()  # the note replaces the figure
        assert "Traceback" in capsys.readouterr().err  # logged, not swallowed silently
    finally:
        root.destroy()


def test_a_garbage_library_file_does_not_stop_the_app_starting(
    monkeypatch, tmp_path, capsys
) -> None:
    bad = tmp_path / "sign_library.npz"
    bad.write_bytes(b"this is not a zip archive at all" * 10)
    _build_with_library_file(monkeypatch, bad, capsys)


def test_a_truncated_library_file_does_not_stop_the_app_starting(
    monkeypatch, tmp_path, capsys
) -> None:
    good = tmp_path / "good.npz"
    np.savez(good, HELLO=np.zeros((45, 153), dtype=np.float32))
    cut = tmp_path / "sign_library.npz"
    cut.write_bytes(good.read_bytes()[:40])
    _build_with_library_file(monkeypatch, cut, capsys)


def test_a_library_with_an_empty_take_does_not_stop_the_app_starting(
    monkeypatch, tmp_path, capsys
) -> None:
    odd = tmp_path / "sign_library.npz"
    np.savez(odd, HELLO=np.zeros((0, 153), dtype=np.float32))  # rest_frame can't use this
    _build_with_library_file(monkeypatch, odd, capsys)


def test_show_frame_puts_the_camera_image_on_the_preview_without_imagetk(app, monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "PIL.ImageTk", None)  # as on a Pi without python3-pil.imagetk
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (10, 200, 30)
    frame[0, 0] = (255, 0, 0)
    app.show_frame(frame)
    photo = app._photo
    assert str(app.preview.cget("image")) == str(photo)
    assert (photo.width(), photo.height()) == (320, 240)
    assert photo.get(0, 0) == (255, 0, 0)
    assert photo.get(100, 100) == (10, 200, 30)
    app.show_frame(np.full((240, 320, 3), 77, dtype=np.uint8))  # later frames replace the image
    assert str(app.preview.cget("image")) == str(app._photo)
    assert app._photo.get(100, 100) == (77, 77, 77)


def test_stale_events_from_the_old_mode_are_dropped_when_a_switch_completes(app) -> None:
    app.set_mode("speech")
    assert app.status_var.get() == "Switching..."  # visible from both views
    app.ctl.events = [
        SignStatus("capturing", 0.5, True, 7.0),
        WorkerReady("sign"),
        WorkerError("old camera error"),
        SignResult("HELLO", "HELLO", 0.9),  # recognised just before the stop: kept
    ]
    app._tick()
    assert app.status_var.get() == ""  # the stale SignStatus did not set the speech status
    assert app.banner_var.get() == ""  # nor did the stale WorkerError raise a banner
    assert app.sentence_var.get() == "HELLO"
    assert app.ctl.started[-1] == "speech"


def test_a_speech_final_that_arrives_during_a_switch_is_kept(speech_app) -> None:
    speech_app.set_mode("sign")
    speech_app.ctl.events = [
        SpeechText("late partial", False),
        SpeechText("see you soon", True),
    ]
    speech_app._tick()
    assert "see you soon" in speech_app.transcript_text.get("1.0", "end")
    assert "late partial" not in speech_app.transcript_text.get("1.0", "end")


def test_exit_always_destroys_the_window_even_if_the_worker_will_not_stop(app) -> None:
    def failing_shutdown(timeout=None):
        raise RuntimeError("worker would not stop")

    app.ctl.shutdown = failing_shutdown
    with pytest.raises(RuntimeError):
        app.exit_app()
    with pytest.raises(tk.TclError):
        app.root.winfo_exists()  # destroyed


def test_a_failing_rest_frame_is_logged_once_not_every_tick(speech_app, capsys) -> None:
    speech_app.root.after = lambda ms, fn: None
    speech_app._rest = "not a frame"
    speech_app._rest_drawn = False
    speech_app._play_tick()
    assert "Traceback" in capsys.readouterr().err
    speech_app._play_tick()
    speech_app._play_tick()
    assert capsys.readouterr().err == ""


def test_the_mouse_cursor_is_hidden_only_in_fullscreen() -> None:
    results = {}
    for fullscreen in (True, False):
        try:
            root = tk.Tk()
        except tk.TclError:
            pytest.skip("no display available")
        root.withdraw()
        try:
            SenselessApp(
                root,
                controller=FakeController(),
                start_workers=False,
                fullscreen=fullscreen,
                load_library_file=False,
            )
            results[fullscreen] = str(root.cget("cursor"))
        finally:
            root.destroy()
    assert results == {True: "none", False: ""}
