"""Run the touchscreen app: ``python -m senseless.ui`` (full screen on the Pi).

``--windowed`` opens an 800x480 window instead (development on a PC).
"""

from __future__ import annotations

import argparse
import signal


def install_sigterm_handler(root, app) -> None:
    """Turn ``kill``/systemd stop into the same clean exit as the power dialog."""
    signal.signal(signal.SIGTERM, lambda *_: root.after(0, app.exit_app))


def main() -> None:
    # Imported here, not at module level: every spawned worker process re-imports this
    # module as __mp_main__, and must not pay for Tk and the GUI code.
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
