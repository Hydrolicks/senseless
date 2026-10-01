"""The python -m senseless.ui entry point: cheap to import, stops cleanly on SIGTERM."""

import signal
import subprocess
import sys


def test_importing_the_entry_point_does_not_load_the_gui_stack() -> None:
    # Every spawned worker re-imports the main module, so it must stay light.
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
