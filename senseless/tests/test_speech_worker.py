"""Speech worker loop with a fake microphone and a scripted fake Vosk."""

import queue
import threading

from senseless.asr.audio import AudioSource
from senseless.asr.transcriber import Transcript
from senseless.asr.worker import SpeechParts, run_speech_worker
from senseless.common.events import SpeechText, WorkerError, WorkerReady
from senseless.common.queue import DropOldestQueue


class FakeMic(AudioSource):
    def __init__(self, n_blocks: int | None) -> None:
        self.n, self.i = n_blocks, 0

    def read(self):
        if self.n is not None and self.i >= self.n:
            return None
        self.i += 1
        return b"\x00\x00" * 8000

    def close(self) -> None:
        pass


class ScriptedVosk:
    SCRIPT = [
        Transcript("hel", False),
        Transcript("hello", False),
        Transcript("hello", False),  # unchanged partial: not re-sent
        Transcript("hello there", True),
        Transcript("", True),  # nothing new
    ]

    def __init__(self) -> None:
        self.i = 0

    def accept(self, pcm: bytes) -> Transcript:
        result = self.SCRIPT[min(self.i, len(self.SCRIPT) - 1)]
        self.i += 1
        return result


def _drain(q) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_partials_are_deduplicated_and_finals_passed_on() -> None:
    events = DropOldestQueue(10_000, backend=queue.Queue)
    run_speech_worker(events, threading.Event(), SpeechParts(lambda: FakeMic(5), ScriptedVosk))
    assert _drain(events) == [
        WorkerReady("speech"),
        SpeechText("hel", False),
        SpeechText("hello", False),
        SpeechText("hello there", True),
        WorkerError("Microphone stream ended."),
    ]


def test_stop_event_ends_the_loop() -> None:
    events = DropOldestQueue(10_000, backend=queue.Queue)
    stop = threading.Event()
    parts = SpeechParts(lambda: FakeMic(None), ScriptedVosk)
    worker = threading.Thread(target=run_speech_worker, args=(events, stop, parts))
    worker.start()
    stop.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()


def test_a_missing_speech_model_is_reported() -> None:
    def no_model():
        raise FileNotFoundError("Vosk model not found at models/vosk-model-small-en-us-0.15")

    events = DropOldestQueue(10, backend=queue.Queue)
    run_speech_worker(events, threading.Event(), SpeechParts(lambda: FakeMic(1), no_model))
    (error,) = _drain(events)
    assert isinstance(error, WorkerError) and "Speech model" in error.message
