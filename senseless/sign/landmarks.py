"""Per-frame landmark processing for the sign channel.

Two layers live here:

1. ``RawLandmarks`` + ``frame_landmarks_to_vector`` -- a PURE, framework-agnostic
   numpy path (no MediaPipe import) that turns one frame's raw landmarks into the
   normalized 153-dim feature vector defined in ``common/landmark_schema.py``.
   This is the single source of truth for normalization and the missing-hand
   policy, imported by both on-Pi inference and the data-collection / Colab
   training pipeline so they stay in lockstep.

2. ``PerceptionBackend`` and its two implementations (``TasksBackend``,
   ``HolisticBackend``) -- thin adapters that run MediaPipe on an RGB frame and
   emit ``RawLandmarks``. MediaPipe is lazy-imported inside them, so importing
   this module (and unit-testing layer 1) never requires MediaPipe.

Normalization (body-anchored, translation- and scale-invariant)
---------------------------------------------------------------
All landmarks (both hands and the pose subset) are mapped into one body-centric
frame: re-origined to the shoulder midpoint and divided by the shoulder width::

    p' = (p - shoulder_midpoint) / shoulder_width

Subtracting the shoulder midpoint removes where the signer is in the frame
(translation invariance); dividing by shoulder width removes how far the signer
is from the camera and their body size (scale invariance). Shoulders are the
reference because they barely move while the arms/hands do, keeping the frame
stable, and one shared transform preserves the hand position relative to the
body -- the linguistically meaningful "location" of a sign. Shoulder width uses
the image (x, y) plane only; z is the noisiest, most heterogeneous channel.

Missing-hand / no-detection policy (zero-fill)
----------------------------------------------
An absent hand (one-handed sign or detection dropout) has its 63-dim block set
to exactly zero. If the pose (hence the shoulders) is missing or shoulder width
is degenerate, the whole frame is emitted as zeros ("no detection"). The rule is
stateless and matches the schema; the training pipeline uses this same function,
so the model trains on identical zero-blocks to those seen at inference.

Perception backend
------------------
``config.SIGN.perception_backend`` selects the default ("tasks" recommended).
Both backends sit behind one interface so ``eval/bench_perception.py`` can
measure them on the Pi and you can switch with one config line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import PATHS, SIGN


@dataclass(eq=False)
class RawLandmarks:
    """Framework-agnostic per-frame landmarks, already slotted to left/right.

    Coordinates are MediaPipe-style normalized image coords, shape ``(N, 3)``:
    ``left_hand`` / ``right_hand`` are ``(21, 3)`` or ``None`` when not detected;
    ``pose`` is the full ``(33, 3)`` MediaPipe Pose array or ``None``.
    """

    left_hand: np.ndarray | None
    right_hand: np.ndarray | None
    pose: np.ndarray | None


def frame_landmarks_to_vector(raw: RawLandmarks) -> np.ndarray:
    """Normalize one frame's landmarks into the flat ``FEATURE_DIM`` vector.

    See the module docstring for the normalization and missing-hand policy.
    Returns an all-zero vector when there is no usable body frame.
    """
    vec = np.zeros(ls.FEATURE_DIM, dtype=np.float32)
    if raw.pose is None:
        return vec

    pose = np.asarray(raw.pose, dtype=np.float32)
    left_shoulder = pose[ls.LEFT_SHOULDER_INDEX]
    right_shoulder = pose[ls.RIGHT_SHOULDER_INDEX]
    origin = (left_shoulder + right_shoulder) / 2.0
    scale = float(np.linalg.norm((left_shoulder - right_shoulder)[:2]))
    if scale < SIGN.normalization_eps:
        return vec

    def _normalize(points: np.ndarray) -> np.ndarray:
        return ((np.asarray(points, dtype=np.float32) - origin) / scale).reshape(-1)

    if raw.left_hand is not None:
        vec[ls.LEFT_HAND_START : ls.LEFT_HAND_END] = _normalize(raw.left_hand)
    if raw.right_hand is not None:
        vec[ls.RIGHT_HAND_START : ls.RIGHT_HAND_END] = _normalize(raw.right_hand)
    pose_subset = pose[list(ls.POSE_LANDMARK_INDICES)]
    vec[ls.POSE_START : ls.POSE_END] = _normalize(pose_subset)
    return vec


# --------------------------------------------------------------------------- #
# Perception backends                                                          #
# MediaPipe is lazy-imported here and is NOT exercised by the unit tests.      #
# --------------------------------------------------------------------------- #


def _normalized_landmarks_to_array(landmarks) -> np.ndarray:
    """Convert an iterable of MediaPipe NormalizedLandmark to an (N, 3) array."""
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)


class PerceptionBackend(ABC):
    """Runs a perception model on an RGB frame and returns ``RawLandmarks``."""

    @abstractmethod
    def extract(self, frame_rgb: np.ndarray, timestamp_ms: int) -> RawLandmarks:
        """Return slotted landmarks for one RGB frame (H, W, 3 uint8)."""

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""

    def __enter__(self) -> PerceptionBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TasksBackend(PerceptionBackend):
    """MediaPipe Tasks HandLandmarker + PoseLandmarker, composed (face-free)."""

    def __init__(self) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(PATHS.pose_landmarker_task)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=SIGN.min_pose_detection_confidence,
                min_tracking_confidence=SIGN.min_tracking_confidence,
            )
        )
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(PATHS.hand_landmarker_task)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=SIGN.min_hand_detection_confidence,
                min_tracking_confidence=SIGN.min_tracking_confidence,
            )
        )

    def extract(self, frame_rgb: np.ndarray, timestamp_ms: int) -> RawLandmarks:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        pose_result = self._pose.detect_for_video(image, timestamp_ms)
        hand_result = self._hands.detect_for_video(image, timestamp_ms)

        pose = (
            _normalized_landmarks_to_array(pose_result.pose_landmarks[0])
            if pose_result.pose_landmarks
            else None
        )
        left = right = None
        for marks, handedness in zip(
            hand_result.hand_landmarks, hand_result.handedness, strict=False
        ):
            arr = _normalized_landmarks_to_array(marks)
            # MediaPipe labels handedness from the image; flip with SIGN.mirror so
            # left_hand always means the signer's left hand for your camera.
            is_left = (handedness[0].category_name == "Left") ^ SIGN.mirror
            if is_left and left is None:
                left = arr
            elif not is_left and right is None:
                right = arr
            elif left is None:
                left = arr
            else:
                right = arr
        return RawLandmarks(left_hand=left, right_hand=right, pose=pose)

    def close(self) -> None:
        self._pose.close()
        self._hands.close()


class HolisticBackend(PerceptionBackend):
    """Legacy ``mp.solutions.Holistic``; face landmarks are computed but ignored.

    Kept for the on-Pi benchmark only. Holistic runs the face mesh internally
    with no flag to disable it, so it pays face-mesh CPU we then discard -- which
    is why ``TasksBackend`` is the recommended default.
    """

    def __init__(self) -> None:
        import mediapipe as mp

        if not hasattr(getattr(mp, "solutions", None), "holistic"):
            raise RuntimeError(
                "mp.solutions.holistic is unavailable in MediaPipe "
                f"{getattr(mp, '__version__', '?')}. The Holistic backend needs the "
                "legacy solutions API (present in e.g. mediapipe 0.10.14 on the Pi); "
                "newer wheels removed it. Use perception_backend='tasks' instead."
            )
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=SIGN.holistic_model_complexity,
            refine_face_landmarks=False,
            min_detection_confidence=SIGN.min_pose_detection_confidence,
            min_tracking_confidence=SIGN.min_tracking_confidence,
        )

    def extract(self, frame_rgb: np.ndarray, timestamp_ms: int) -> RawLandmarks:
        # Holistic is an internally stateful stream; timestamp_ms is unused but
        # kept for interface parity with TasksBackend.
        result = self._holistic.process(frame_rgb)
        pose = (
            _normalized_landmarks_to_array(result.pose_landmarks.landmark)
            if result.pose_landmarks
            else None
        )
        left = (
            _normalized_landmarks_to_array(result.left_hand_landmarks.landmark)
            if result.left_hand_landmarks
            else None
        )
        right = (
            _normalized_landmarks_to_array(result.right_hand_landmarks.landmark)
            if result.right_hand_landmarks
            else None
        )
        if SIGN.mirror:
            left, right = right, left
        return RawLandmarks(left_hand=left, right_hand=right, pose=pose)

    def close(self) -> None:
        self._holistic.close()


def create_backend(name: str | None = None) -> PerceptionBackend:
    """Instantiate the configured perception backend ("tasks" or "holistic")."""
    name = (name or SIGN.perception_backend).lower()
    if name == "tasks":
        return TasksBackend()
    if name == "holistic":
        return HolisticBackend()
    raise ValueError(f"Unknown perception backend {name!r} (expected 'tasks' or 'holistic')")
