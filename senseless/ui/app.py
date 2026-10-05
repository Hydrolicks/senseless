# senseless/ui/app.py
"""Senseless touchscreen app: Tkinter on the Pi's 7" 800x480 display.

Two modes, one at a time (each gets the whole CPU): Sign (camera preview, the
recognized word, a running sentence) and Speech (a scrolling live transcript).
The heavy work runs in a worker process managed by ``ModeController``; this
class only drains its queues every ``UI.poll_ms`` and draws.
"""

from __future__ import annotations

import subprocess
import time
import tkinter as tk
import traceback
from collections import deque

import numpy as np

from senseless.common.config import UI
from senseless.common.events import SignResult, SignStatus, SpeechText, WorkerError, WorkerReady
from senseless.ui.figure import figure_geometry, frame_index, rest_frame
from senseless.ui.state import (
    SignSentence,
    SpeechTranscript,
    TextScale,
    highlight_spans,
    words_to_sign,
)

FONT = "DejaVu Sans"
HINTS = {
    "idle": "Rest hands out of view, then sign",
    "pending": "Hand detected...",
    "capturing": "Capturing sign...",
    "holding": "Lower your hands for the next sign",
}
MODE_COLOR = {"sign": UI.teal, "speech": UI.coral}


def _button(parent, text, command, fg=UI.text, width=None) -> tk.Button:
    return tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        bg=UI.panel,
        fg=fg,
        activebackground=UI.teal,
        activeforeground=UI.text,
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground="#2E5A62",
        font=(FONT, 13),
        padx=10,
        pady=6,
    )


def _power_icon(parent, command) -> tk.Canvas:
    """A drawn power symbol: the Pi's fonts have no U+23FB glyph (it shows as a box)."""
    icon = tk.Canvas(parent, width=40, height=36, bg=UI.panel, highlightthickness=0, bd=0)
    icon.create_arc(10, 8, 30, 28, start=120, extent=300, style="arc", outline=UI.coral, width=3)
    icon.create_line(20, 5, 20, 17, fill=UI.coral, width=3, capstyle="round")
    icon.bind("<Button-1>", lambda _event: command())
    return icon


class SenselessApp:
    """The touchscreen GUI. See the module docstring."""

    def __init__(
        self,
        root: tk.Tk,
        controller=None,
        start_workers: bool = True,
        fullscreen: bool = True,
        initial_mode: str = "sign",
        library: dict[str, np.ndarray] | None = None,
        load_library_file: bool = True,
    ) -> None:
        if controller is None:
            from senseless.ui.controller import ModeController

            controller = ModeController()
        self.root = root
        self.ctl = controller
        self.mode = initial_mode
        self.sentence = SignSentence()
        self.transcript = SpeechTranscript()
        self.scale = TextScale()
        self._pending_mode: str | None = None
        self._error: str | None = None
        self._alive = True
        self._photo = None  # keep a reference, or Tk drops the image

        # The sign library drives the Speech-mode figure; without it Speech mode
        # still works and the figure is replaced by a note (see _build_speech_view).
        self._rest: np.ndarray | None = None
        if library is None and load_library_file:
            try:
                from senseless.sign.library import load_library

                library = load_library()
            except FileNotFoundError:
                library = None  # not built yet: the normal "no figure" case
            except Exception:  # corrupt/truncated file: Speech mode must still start
                traceback.print_exc()
                library = None
        self.library: dict[str, np.ndarray] = library or {}
        try:
            self._rest = rest_frame(self.library)
        except Exception:  # a malformed take: drop the library rather than fail to start
            traceback.print_exc()
            self.library, self._rest = {}, None
        self.sign_queue: deque[str] = deque(maxlen=UI.sign_queue_max)  # drop-oldest
        self._playing: str | None = None
        self._play_t0 = 0.0
        self._rest_drawn = False

        self.word_var = tk.StringVar(value="...")
        self.conf_var = tk.StringVar(value="")
        self.hint_var = tk.StringVar(value="Starting camera...")
        self.sentence_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.banner_var = tk.StringVar(value="")
        self.dialog_error_var = tk.StringVar(value="")
        self.figure_caption = tk.StringVar(value="")

        root.title("Senseless")
        root.configure(bg=UI.bg)
        if fullscreen:
            self._enter_fullscreen()
            # Again once the window is on screen: Wayland compositors (labwc on the Pi,
            # through XWayland) ignore a fullscreen request made before it is mapped.
            root.after(500, self._enter_fullscreen)
            root.after(2000, self._enter_fullscreen)
            root.config(cursor="none")  # a touchscreen has no pointer to show
        else:
            root.geometry(f"{UI.width}x{UI.height}")
        self._build()
        self._apply_fonts()
        self._show_view(self.mode)
        if start_workers:
            self.ctl.start(self.mode)
        root.protocol("WM_DELETE_WINDOW", self.exit_app)
        root.after(UI.poll_ms, self._tick)
        root.after(UI.figure_tick_ms, self._play_tick)

    def _enter_fullscreen(self) -> None:
        root = self.root
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
        root.attributes("-fullscreen", True)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        bar = tk.Frame(self.root, bg=UI.panel, height=52)
        bar.pack(side="top", fill="x")
        self.mode_buttons = {
            "sign": _button(bar, "Sign", lambda: self.set_mode("sign"), width=7),
            "speech": _button(bar, "Speech", lambda: self.set_mode("speech"), width=7),
        }
        self.mode_buttons["sign"].pack(side="left", padx=(8, 0), pady=6)
        self.mode_buttons["speech"].pack(side="left", pady=6)
        self.power_button = _power_icon(bar, self.open_power_dialog)
        self.power_button.pack(side="right", padx=8)
        _button(bar, "A+", self.bigger).pack(side="right", padx=2)
        _button(bar, "A−", self.smaller).pack(side="right", padx=2)
        self.status_label = tk.Label(bar, textvariable=self.status_var, bg=UI.panel, fg=UI.muted)
        self.status_label.pack(side="left", expand=True)

        self.content = tk.Frame(self.root, bg=UI.bg)
        self.content.pack(side="top", fill="both", expand=True)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)
        self.sign_view = self._build_sign_view(self.content)
        self.speech_view = self._build_speech_view(self.content)
        for view in (self.sign_view, self.speech_view):
            view.grid(row=0, column=0, sticky="nsew")

        self.banner = tk.Frame(self.content, bg=UI.panel, padx=16, pady=12)
        self.banner_label = tk.Label(
            self.banner, textvariable=self.banner_var, bg=UI.panel, fg=UI.text, wraplength=560
        )
        self.banner_label.pack(side="top", pady=(0, 8))
        _button(self.banner, "Retry", self.retry).pack(side="top")

        self.dialog = tk.Frame(self.root, bg="#16373F", padx=20, pady=16)
        tk.Label(
            self.dialog, text="Close Senseless?", bg="#16373F", fg=UI.text, font=(FONT, 16, "bold")
        ).pack(side="top", pady=(0, 10))
        tk.Label(self.dialog, textvariable=self.dialog_error_var, bg="#16373F", fg=UI.coral).pack(
            side="top"
        )
        row = tk.Frame(self.dialog, bg="#16373F")
        row.pack(side="top")
        _button(row, "Cancel", self.close_power_dialog).pack(side="left", padx=4)
        _button(row, "Exit app", self.exit_app).pack(side="left", padx=4)
        _button(row, "Power off", self.power_off, fg=UI.coral).pack(side="left", padx=4)

    def _build_sign_view(self, parent) -> tk.Frame:
        view = tk.Frame(parent, bg=UI.bg)
        left = tk.Frame(view, bg=UI.bg)
        left.grid(row=0, column=0, sticky="nw", padx=(10, 6), pady=10)
        blank = tk.PhotoImage(width=UI.preview_size[0], height=UI.preview_size[1])
        self._blank = blank
        self.preview = tk.Label(left, image=blank, bg="#33474C", bd=0)
        self.preview.pack(side="top")
        self.progress = tk.Canvas(
            left, width=UI.preview_size[0], height=8, bg="#23434A", highlightthickness=0
        )
        self.progress.pack(side="top", pady=(6, 2))
        self._progress_bar = self.progress.create_rectangle(0, 0, 0, 8, fill=UI.coral, width=0)
        self.hint_label = tk.Label(left, textvariable=self.hint_var, bg=UI.bg, fg=UI.muted)
        self.hint_label.pack(side="top", anchor="w")

        right = tk.Frame(view, bg=UI.bg)
        right.grid(row=0, column=1, sticky="nsew", pady=10)
        view.grid_columnconfigure(1, weight=1)
        view.grid_rowconfigure(0, weight=1)
        self.word_label = tk.Label(right, textvariable=self.word_var, bg=UI.bg, fg=UI.word)
        self.word_label.pack(side="top", expand=True)
        self.conf_label = tk.Label(right, textvariable=self.conf_var, bg=UI.bg, fg=UI.muted)
        self.conf_label.pack(side="top")

        bottom = tk.Frame(view, bg=UI.bg)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        bottom.grid_columnconfigure(0, weight=1)
        self.sentence_label = tk.Label(
            bottom,
            textvariable=self.sentence_var,
            bg=UI.panel,
            fg=UI.text,
            anchor="w",
            padx=10,
            pady=8,
        )
        self.sentence_label.grid(row=0, column=0, sticky="ew")
        _button(bottom, "Undo", self.undo).grid(row=0, column=1, padx=(6, 0))
        _button(bottom, "Clear", self.clear).grid(row=0, column=2, padx=(6, 0))
        return view

    def _build_speech_view(self, parent) -> tk.Frame:
        view = tk.Frame(parent, bg=UI.bg)
        # Pack order matters: the Clear row first (bottom) so it always keeps its full
        # height, then the figure (right) so the transcript reflows beside it.
        row = tk.Frame(view, bg=UI.bg)
        row.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        _button(row, "Clear", self.clear).pack(side="right")

        self.figure_panel = tk.Frame(view, bg=UI.panel)
        self.figure_canvas = tk.Canvas(
            self.figure_panel, width=220, height=210, bg=UI.panel, highlightthickness=0
        )
        self.figure_canvas.pack(side="top")
        tk.Label(
            self.figure_panel,
            textvariable=self.figure_caption,
            bg=UI.panel,
            fg=UI.word,
            font=(FONT, 13, "bold"),
        ).pack(side="top", pady=(0, 6))
        self.library_note = tk.Label(
            view,
            text="No sign library: run python -m senseless.sign.library",
            bg=UI.bg,
            fg=UI.muted,
        )
        if self.library:
            self.figure_panel.pack(side="right", anchor="n", padx=(6, 10), pady=10)
        else:
            self.library_note.pack(side="top", anchor="e", padx=10, pady=(10, 0))

        self.transcript_text = tk.Text(
            view, bg=UI.bg, fg=UI.text, wrap="word", bd=0, highlightthickness=0, padx=12, pady=8
        )
        self.transcript_text.pack(side="left", fill="both", expand=True)
        self.transcript_text.configure(state="disabled")
        return view

    def _apply_fonts(self) -> None:
        f = self.scale.factor
        self.word_label.configure(font=(FONT, int(54 * f), "bold"))
        self.conf_label.configure(font=(FONT, int(13 * f)))
        self.hint_label.configure(font=(FONT, int(12 * f)))
        self.sentence_label.configure(font=(FONT, int(22 * f), "bold"))
        self.status_label.configure(font=(FONT, 12))
        self.banner_label.configure(font=(FONT, int(15 * f)))
        size = int(20 * f)
        self.transcript_text.tag_configure("old", foreground=UI.muted, font=(FONT, size))
        self.transcript_text.tag_configure("new", foreground=UI.text, font=(FONT, size))
        self.transcript_text.tag_configure(
            "partial", foreground=UI.muted, font=(FONT, size, "italic")
        )
        self.transcript_text.tag_configure("signed", foreground=UI.word, font=(FONT, size, "bold"))
        # Bigger text re-wraps the lines: keep the newest one in view once Tk has laid it out.
        self.transcript_text.after_idle(self.transcript_text.see, "end")

    def _show_view(self, mode: str) -> None:
        (self.sign_view if mode == "sign" else self.speech_view).tkraise()
        for name, btn in self.mode_buttons.items():
            btn.configure(bg=MODE_COLOR[name] if name == mode else UI.panel)

    # ------------------------------------------------------------- the loop
    def _tick(self) -> None:
        if not self._alive:
            return
        try:
            self._tick_body()
        except Exception as exc:  # keep the GUI loop alive whatever a tick hits
            message = f"App error: {exc}"
            if message != self._error:  # log each distinct failure once, not every tick
                traceback.print_exc()
            self.show_error(message)
        finally:
            if self._alive:
                self.root.after(UI.poll_ms, self._tick)

    def _tick_body(self) -> None:
        if self._pending_mode is not None:
            if self.ctl.poll_stopped():
                self._absorb_stale_events()
                self.status_var.set("")  # clear "Switching..." and anything from the old mode
                self.mode, self._pending_mode = self._pending_mode, None
                if self.mode == "sign":
                    self._reset_figure()
                self._show_view(self.mode)
                self.ctl.start(self.mode)
        else:
            for event in self.ctl.drain_events():
                self.handle_event(event)
            frame = self.ctl.latest_frame()
            if frame is not None and self.mode == "sign":
                self.show_frame(frame)
            if self._error is None and self.ctl.has_exited():
                code = self.ctl.exitcode()
                self.show_error(f"The {self.mode} engine stopped unexpectedly (exit code {code}).")

    def _absorb_stale_events(self) -> None:
        """Events the old worker left behind: keep results and finals, drop the rest.

        A status, ready or error from the old mode would be shown as if it came from the
        new one. A sign recognised or a line finalised just before the stop still counts.
        """
        for event in self.ctl.drain_events():
            if isinstance(event, SignResult) or (isinstance(event, SpeechText) and event.is_final):
                self.handle_event(event)

    def handle_event(self, event) -> None:
        if isinstance(event, WorkerReady):
            if event.mode == "sign":
                self.hint_var.set(HINTS["idle"])
            else:
                self.status_var.set("● listening")
        elif isinstance(event, SignStatus):
            self.hint_var.set(HINTS.get(event.state, ""))
            self._set_progress(event.progress)
            hands = "✓" if event.hands else "–"
            self.status_var.set(f"hands {hands}  ·  {event.fps:.0f} FPS")
        elif isinstance(event, SignResult):
            if event.word:
                self.sentence.add(event.word)
                self.word_var.set(event.word)
                self.conf_var.set(f"confidence {event.confidence:.2f}")
            else:
                self.word_var.set("?")
                self.conf_var.set(
                    f"not recognized (best guess {event.best} {event.confidence:.2f})"
                )
            self.sentence_var.set(self.sentence.text)
        elif isinstance(event, SpeechText):
            self.on_speech(event)
        elif isinstance(event, WorkerError):
            self.show_error(event.message)

    def on_speech(self, event: SpeechText) -> None:
        self.transcript.add(event.text, event.is_final)
        if event.is_final and event.text and self.library:
            self.sign_queue.extend(words_to_sign(event.text, self.library.keys()))
        self._render_transcript()

    def _render_transcript(self) -> None:
        t = self.transcript_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        lines = self.transcript.lines
        for i, line in enumerate(lines):
            start = t.index("end-1c")
            t.insert("end", line + "\n", "new" if i == len(lines) - 1 else "old")
            for s, e in highlight_spans(line, self.library.keys()):
                t.tag_add("signed", f"{start}+{s}c", f"{start}+{e}c")
        if self.transcript.partial:
            t.insert("end", self.transcript.partial, "partial")
        t.configure(state="disabled")
        t.see("end")

    # ------------------------------------------------------------ the figure
    def _play_tick(self) -> None:
        if not self._alive:
            return
        try:
            self._play_tick_body()
        except Exception:  # a bad frame must not freeze the figure for good
            traceback.print_exc()
            self._playing = None  # drop the offending word instead of retrying it forever
            self.figure_caption.set("")
        finally:
            if self._alive:
                self.root.after(UI.figure_tick_ms, self._play_tick)

    def _reset_figure(self) -> None:
        """Stop the figure: forget the queued words and redraw the resting pose next tick."""
        self.sign_queue.clear()
        self._playing = None
        self._rest_drawn = False
        self.figure_caption.set("")

    def _play_tick_body(self) -> None:
        if self.mode != "speech":  # the figure is hidden in Sign mode: no drawing, no playback
            return
        now = time.perf_counter()
        if self._playing is None and self.sign_queue:
            self._playing = self.sign_queue.popleft()
            self._play_t0 = now
            self.figure_caption.set(self._playing)
        if self._playing is not None:
            index = frame_index(now - self._play_t0, len(self.library[self._playing]))
            if index is None:
                self._playing = None
                self.figure_caption.set("")
                self._rest_drawn = False
            else:
                self._draw_figure(self.library[self._playing][index])
        if self._playing is None and not self._rest_drawn and self._rest is not None:
            self._rest_drawn = (
                True  # first: a failing draw must not be retried (and logged) at 30 Hz
            )
            self._draw_figure(self._rest)

    def _draw_figure(self, frame: np.ndarray) -> None:
        c = self.figure_canvas
        c.delete("all")
        colors = {"body": "#CFE6E8", "left": "#50D28C", "right": "#F2A33A"}
        widths = {"body": 3, "left": 2, "right": 2}
        segments, dots = figure_geometry(frame, (8, 8, 204, 194))
        for seg in segments:
            c.create_line(
                seg.x0, seg.y0, seg.x1, seg.y1, fill=colors[seg.kind], width=widths[seg.kind]
            )
        for d in dots:
            box = (d.x - d.r, d.y - d.r, d.x + d.r, d.y + d.r)
            if d.kind == "head":
                c.create_oval(*box, outline="#CFE6E8", width=2)
            else:
                c.create_oval(*box, fill=UI.coral, width=0)

    def show_frame(self, frame: np.ndarray) -> None:
        # A binary PPM straight from the array: Tk decodes it itself, so neither PIL's
        # ImageTk (a separate Debian package on the Pi) nor a PIL import is needed.
        rgb = np.ascontiguousarray(frame, dtype=np.uint8)
        height, width = rgb.shape[:2]
        ppm = b"P6 %d %d 255\n" % (width, height) + rgb.tobytes()
        if self._photo is None:
            self._photo = tk.PhotoImage(data=ppm, format="PPM")
            self.preview.configure(image=self._photo)
        else:  # reuse the one image: no per-frame allocation, and Tk keeps the reference
            self._photo.configure(data=ppm, format="PPM")

    def _set_progress(self, fraction: float) -> None:
        self.progress.coords(self._progress_bar, 0, 0, int(UI.preview_size[0] * fraction), 8)

    # ---------------------------------------------------------- errors / modes
    def show_error(self, message: str) -> None:
        self._error = message
        self.banner_var.set(message)
        self.banner.place(relx=0.5, rely=0.5, anchor="center")
        self.banner.tkraise()

    def hide_error(self) -> None:
        self._error = None
        self.banner_var.set("")
        self.banner.place_forget()

    def set_mode(self, mode: str) -> None:
        if mode == self.mode and self._pending_mode is None and self._error is None:
            return
        self.hide_error()
        self._pending_mode = mode
        self.hint_var.set("Switching...")
        self.status_var.set("Switching...")  # the status bar shows on both views
        for name, btn in self.mode_buttons.items():
            btn.configure(bg=MODE_COLOR[name] if name == mode else UI.panel)
        self.ctl.request_stop()

    def retry(self) -> None:
        self.set_mode(self.mode if self._pending_mode is None else self._pending_mode)

    # ---------------------------------------------------------------- controls
    def undo(self) -> None:
        self.sentence.undo()
        self.sentence_var.set(self.sentence.text)

    def clear(self) -> None:
        if self.mode == "sign":
            self.sentence.clear()
            self.sentence_var.set("")
            self.word_var.set("...")
            self.conf_var.set("")
        else:
            self.transcript.clear()
            self._render_transcript()
            self._reset_figure()

    def bigger(self) -> None:
        self.scale.bigger()
        self._apply_fonts()

    def smaller(self) -> None:
        self.scale.smaller()
        self._apply_fonts()

    # ------------------------------------------------------------------- power
    def open_power_dialog(self) -> None:
        self.dialog_error_var.set("")
        self.dialog.place(relx=0.5, rely=0.5, anchor="center")
        self.dialog.tkraise()

    def close_power_dialog(self) -> None:
        self.dialog.place_forget()

    def exit_app(self) -> None:
        self._alive = False
        try:
            self.ctl.shutdown()
        finally:
            self.root.destroy()

    def power_off(self) -> None:
        self.ctl.shutdown()
        try:
            subprocess.run(
                ["sudo", "systemctl", "poweroff"], check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            self.dialog_error_var.set(f"Power off failed: {detail.strip()}")
            self._pending_mode = self.mode  # bring the worker back
