"""Microphone audio sources for the ASR channel.

``AudioSource`` mirrors the sign channel's ``FrameSource``: one interface with a
sounddevice-backed ``MicrophoneSource`` behind it. sounddevice is lazy-imported,
so this module imports anywhere. Blocks are raw 16-bit mono PCM bytes at
``config.AUDIO`` settings -- exactly what ``VoskTranscriber.accept`` expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from senseless.common.config import AUDIO


def find_input_device(name_hints: tuple[str, ...] = AUDIO.device_name_hints) -> int | None:
    """Return the index of the first input device whose name matches a hint.

    Matching is case-insensitive substring (e.g. "ReSpeaker"). Returns None (the
    system default) when nothing matches.
    """
    import sounddevice as sd

    hints = tuple(h.lower() for h in name_hints)
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        if any(hint in str(dev["name"]).lower() for hint in hints):
            return index
    return None


def list_input_devices() -> list[tuple[int, str]]:
    """List ``(index, name)`` for every input-capable device."""
    import sounddevice as sd

    return [
        (i, str(d["name"]))
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] >= 1
    ]


class AudioSource(ABC):
    """A source of 16-bit mono PCM blocks. Use as a context manager or via blocks()."""

    @abstractmethod
    def read(self) -> bytes | None:
        """Return the next PCM block, or None when the stream ends."""

    @abstractmethod
    def close(self) -> None:
        """Release the audio device."""

    def __enter__(self) -> AudioSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def blocks(self) -> Iterator[bytes]:
        """Yield PCM blocks until the source ends."""
        while True:
            block = self.read()
            if block is None:
                break
            yield block


class MicrophoneSource(AudioSource):
    """Live microphone capture via sounddevice (raw 16-bit mono PCM)."""

    def __init__(self, device: int | None = None) -> None:
        import sounddevice as sd

        self._stream = sd.RawInputStream(
            samplerate=AUDIO.sample_rate_hz,
            blocksize=AUDIO.block_size,
            dtype=AUDIO.dtype,
            channels=AUDIO.channels,
            device=device,
        )
        self._stream.start()

    def read(self) -> bytes | None:
        data, _overflowed = self._stream.read(AUDIO.block_size)
        return bytes(data)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
