"""Live microphone speech-to-text test (DEV TOOL, PC + Pi).

Streams your default (or ReSpeaker) microphone through Vosk and prints partial
transcripts live, with finalized utterances on their own line. Use it to confirm
your mic + the Vosk model work before wiring ASR into the full system.

Needs the Vosk model downloaded (see instructions.md, step 4) and `sounddevice`.

Examples
--------
    python -m senseless.asr.mic_test
    python -m senseless.asr.mic_test --list-devices
    python -m senseless.asr.mic_test --device 2

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse

from senseless.asr import audio
from senseless.asr.transcriber import VoskTranscriber
from senseless.common.config import AUDIO


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Vosk microphone test.")
    parser.add_argument("--device", type=int, default=None, help="Input device index.")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        for index, name in audio.list_input_devices():
            print(f"{index:>3}  {name}")
        return

    device = args.device
    if device is None:
        device = audio.find_input_device(AUDIO.device_name_hints)

    transcriber = VoskTranscriber()
    shown = device if device is not None else "(default)"
    print(f"Listening on input device {shown}. Speak; press Ctrl+C to stop.")
    try:
        with audio.MicrophoneSource(device=device) as mic:
            for block in mic.blocks():
                result = transcriber.accept(block)
                if result.is_final:
                    if result.text:
                        print(f"\nFINAL: {result.text}")
                else:
                    print(f"partial: {result.text}", end="\r", flush=True)
    except KeyboardInterrupt:
        final = transcriber.final()
        if final.text:
            print(f"\nFINAL: {final.text}")
        print("\nStopped.")


if __name__ == "__main__":
    main()
