"""Device selection shared across channels.

The matching logic (``find_device_by_name``) is pure -- it takes a list of
device-info mappings as sounddevice/PortAudio return them, so it is unit-tested
without any audio hardware. ``find_audio_input_device`` is the thin
sounddevice-backed wrapper used at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from senseless.common.config import AUDIO


def find_device_by_name(
    devices: Sequence[Mapping[str, object]],
    name_hints: tuple[str, ...],
    *,
    requires_input: bool = True,
) -> int | None:
    """Index of the first device whose name contains a hint (case-insensitive).

    When ``requires_input`` is set, devices with no input channels are skipped.
    Returns None if nothing matches (callers treat that as "use the default").
    """
    hints = tuple(hint.lower() for hint in name_hints)
    for index, device in enumerate(devices):
        if requires_input and int(device.get("max_input_channels", 0)) < 1:
            continue
        name = str(device.get("name", "")).lower()
        if any(hint in name for hint in hints):
            return index
    return None


def find_audio_input_device(name_hints: tuple[str, ...] = AUDIO.device_name_hints) -> int | None:
    """Index of the first matching audio input device, or None for the default."""
    import sounddevice as sd

    return find_device_by_name(list(sd.query_devices()), name_hints, requires_input=True)
