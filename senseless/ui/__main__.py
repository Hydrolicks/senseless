"""Run the touchscreen app: ``python -m senseless.ui`` (full screen on the Pi).

``--windowed`` opens an 800x480 window instead (development on a PC).
"""

from __future__ import annotations

import argparse
import signal


def install_sigterm_handler(root, app) -> None:
    """Turn ``kill``/systemd stop into the same clean exit as the power dialog."""
    import tkinter as tk

    def on_sigterm(*_) -> None:
        try:
            root.after(0, app.exit_app)
        except tk.TclError:
            pass  # window already destroyed: main()'s finally is stopping the worker

    signal.signal(signal.SIGTERM, on_sigterm)


def main() -> None:
    # Imported here, not at module level, so importing the entry point stays cheap.
    import tkinter as tk

    from senseless.ui.app import SenselessApp

    parser = argparse.ArgumentParser(description="Senseless touchscreen app.")
    parser.add_argument("--windowed", action="store_true", help="800x480 window, not full screen.")
    parser.add_argument("--mode", choices=["sign", "speech"], default="sign")
    args = parser.parse_args()
    root = tk.Tk()
    app = SenselessApp(root, fullscreen=not args.windowed, initial_mode=args.mode)
    install_sigterm_handler(root, app)
    try:
        root.mainloop()
    finally:
        app.ctl.shutdown()  # no-op if exit_app already stopped the worker


if __name__ == "__main__":
    main()
