"""MediaPipe-free estimator fakes for the ParallelBackend tests.

They live in their own module (not a ``test_`` file) because the backend builds
its estimators inside spawned worker processes, which must be able to import the
factories by name. Each fake encodes the frame it saw into its output, so a test
can check that a worker processed the right frame through shared memory.
"""

import numpy as np


class FakePose:
    """Returns a (33, 3) pose filled with the frame's first red value."""

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        return np.full((33, 3), float(frame_rgb[0, 0, 0]), dtype=np.float32)

    def close(self) -> None:
        pass


class FakeHands:
    """Returns one "Left" hand filled with the frame's first green value."""

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list:
        return [(np.full((21, 3), float(frame_rgb[0, 0, 1]), dtype=np.float32), "Left")]

    def close(self) -> None:
        pass


class CrashingPose:
    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        raise ValueError("pose model exploded")

    def close(self) -> None:
        pass


def make_fake_pose() -> FakePose:
    return FakePose()


def make_fake_hands() -> FakeHands:
    return FakeHands()


def make_crashing_pose() -> CrashingPose:
    return CrashingPose()
