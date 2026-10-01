"""The python -m senseless.ui entry point: cheap to import, stops cleanly on SIGTERM."""

import signal
import subprocess
import sys


def test_importing_the_entry_point_does_not_load_the_gui_stack() -> None:
    # The entry point stays cheap to import; the GUI stack loads inside main().
    code = (
        "import sys, senseless.ui.__main__\n"
        "assert 'senseless.ui.app' not in sys.modules\n"
        "assert 'tkinter' not in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_sigterm_asks_the_app_to_exit_through_the_tk_loop() -> None:
    from senseless.ui.__main__ import install_sigterm_handler

    calls = []

    class FakeRoot:
        def after(self, ms, fn):
            calls.append((ms, fn))

    class FakeApp:
        def exit_app(self):
            pass

    app = FakeApp()
    previous = signal.getsignal(signal.SIGTERM)
    try:
        install_sigterm_handler(FakeRoot(), app)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)  # what the OS would trigger
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert calls == [(0, app.exit_app)]


def test_a_sigterm_after_the_window_is_gone_is_ignored() -> None:
    import tkinter as tk

    from senseless.ui.__main__ import install_sigterm_handler

    class DestroyedRoot:
        def after(self, ms, fn):
            raise tk.TclError("application has been destroyed")

    class FakeApp:
        def exit_app(self):
            pass

    previous = signal.getsignal(signal.SIGTERM)
    try:
        install_sigterm_handler(DestroyedRoot(), FakeApp())
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)  # must not raise
    finally:
        signal.signal(signal.SIGTERM, previous)
