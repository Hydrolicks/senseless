"""TDD specs for the pure device-name matcher (common/devices.py).

The matcher takes a list of device-info mappings (as sounddevice/PortAudio
return) so it can be tested without any audio hardware.
"""

from senseless.common.devices import find_device_by_name

_DEVICES = [
    {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2},
    {"name": "Speakers (Realtek Audio)", "max_input_channels": 0},
    {"name": "ReSpeaker 4 Mic Array (UAC1.0)", "max_input_channels": 6},
    {"name": "Integrated Webcam Mic", "max_input_channels": 1},
]


def test_matches_hint_case_insensitively() -> None:
    assert find_device_by_name(_DEVICES, ("respeaker",)) == 2


def test_skips_output_only_devices_when_input_required() -> None:
    assert find_device_by_name(_DEVICES, ("realtek",)) is None


def test_returns_none_when_nothing_matches() -> None:
    assert find_device_by_name(_DEVICES, ("nonexistent",)) is None


def test_first_input_device_matching_any_hint_wins() -> None:
    assert find_device_by_name(_DEVICES, ("respeaker", "input")) == 0


def test_requires_input_false_considers_output_devices() -> None:
    assert find_device_by_name(_DEVICES, ("realtek",), requires_input=False) == 1
