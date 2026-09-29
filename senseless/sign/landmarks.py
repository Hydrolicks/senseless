"""Per-frame landmark processing for the sign channel.

Two layers live here:

1. ``RawLandmarks`` + ``frame_landmarks_to_vector`` -- a PURE, framework-agnostic
   numpy path (no MediaPipe import) that turns one frame's raw landmarks into the
   normalized 153-dim feature vector defined in ``common/landmark_schema.py``.
   This is the single source of truth for normalization and the missing-hand
   policy, imported by both on-Pi inference and the data-collection / Colab
   training pipeline so they stay in lockstep.

2. ``PerceptionBackend`` and its implementations (``TasksBackend``,
   ``LiteBackend``, ``HolisticBackend``) -- thin adapters that run MediaPipe on an
   RGB frame and emit ``RawLandmarks``. MediaPipe is lazy-imported inside them, so
   importing this module (and unit-testing layer 1) never requires MediaPipe.

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
``config.SIGN.perception_backend`` selects the default. All backends sit behind one
interface so ``eval/bench_perception.py`` can measure them on the Pi and you can
switch with one config line (or ``--backend`` on the tools):

* ``tasks``    -- Tasks HandLandmarker + PoseLandmarker. Full hand model; the
  only backend on mediapipe >= 0.10.35 (dev PC). 265 ms/frame for hands on a Pi 4.
* ``lite``     -- legacy ``mp.solutions`` Hands + Pose at model_complexity=0. Lite
  hand model, no face; 95 ms/frame for hands on a Pi 4 (2.8x faster than tasks).
  Needs a mediapipe that still ships ``mp.solutions`` (0.10.18 on the Pi).
* ``holistic`` -- legacy Holistic; always runs the face mesh, no faster on a Pi 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import PATHS, SIGN

BACKEND_NAMES: tuple[str, ...] = ("tasks", "lite", "holistic")


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


def slot_hands(
    detections: list[tuple[np.ndarray, str]], mirror: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Assign hand detections to the signer's ``(left, right)`` slots.

    ``detections`` are ``(landmarks, label)`` pairs in MediaPipe's order, where
    ``label`` is its handedness ("Left"/"Right"). MediaPipe labels handedness from
    the image, so ``mirror`` flips it to mean the signer's own hands for your
    camera. If both hands get the same label, the second takes the free slot;
    detections beyond two are ignored.
    """
    left = right = None
    for arr, label in detections:
        is_left = (label == "Left") ^ mirror
        if is_left and left is None:
            left = arr
        elif not is_left and right is None:
            right = arr
        elif left is None:
            left = arr
        elif right is None:
            right = arr
    return left, right


def _require_solutions(mp, backend: str) -> None:
    """Raise a clear error if this MediaPipe build no longer ships ``mp.solutions``."""
    if not hasattr(getattr(mp, "solutions", None), "hands"):
        raise RuntimeError(
            f"The {backend!r} backend needs the legacy mp.solutions API, which is missing "
            f"from MediaPipe {getattr(mp, '__version__', '?')} (removed in newer wheels). "
            "Install mediapipe 0.10.18 (as on the Pi), or use perception_backend='tasks'."
        )


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


# --- Estimators: one model each. The serial backends run a pose and a hands
# estimator one after the other; sign/parallel.py runs each in its own process.


class PoseEstimator(Protocol):
    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Return the full (33, 3) pose, or ``None`` if no body is found."""

    def close(self) -> None: ...


class HandsEstimator(Protocol):
    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[tuple[np.ndarray, str]]:
        """Return ``(landmarks (21, 3), label)`` per hand, labels in the Tasks convention."""

    def close(self) -> None: ...


def _readonly(frame_rgb: np.ndarray) -> np.ndarray:
    """A read-only contiguous view (lets mp.solutions skip a copy; caller's array untouched)."""
    view = np.ascontiguousarray(frame_rgb).view()
    view.flags.writeable = False
    return view


class _TasksPose:
    """Tasks PoseLandmarker (lite bundle), VIDEO mode."""

    def __init__(self) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # Bundles are passed as bytes, not paths: once an mp.solutions graph has run in
        # the process (lite/holistic), mediapipe 0.10.18 resolves asset paths against
        # its package dir and mangles absolute Windows paths.
        self._model = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_buffer=PATHS.pose_landmarker_task.read_bytes()
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=SIGN.min_pose_detection_confidence,
                min_tracking_confidence=SIGN.min_tracking_confidence,
            )
        )

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._model.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return _normalized_landmarks_to_array(result.pose_landmarks[0])

    def close(self) -> None:
        self._model.close()


class _TasksHands:
    """Tasks HandLandmarker (full hand model), up to two hands, VIDEO mode."""

    def __init__(self) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._model = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_buffer=PATHS.hand_landmarker_task.read_bytes()
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=SIGN.min_hand_detection_confidence,
                min_tracking_confidence=SIGN.min_tracking_confidence,
            )
        )

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[tuple[np.ndarray, str]]:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._model.detect_for_video(image, timestamp_ms)
        return [
            (_normalized_landmarks_to_array(marks), handedness[0].category_name)
            for marks, handedness in zip(result.hand_landmarks, result.handedness, strict=False)
        ]

    def close(self) -> None:
        self._model.close()


class _LitePose:
    """mp.solutions Pose at ``SIGN.lite_model_complexity`` (tracks across calls)."""

    def __init__(self) -> None:
        import mediapipe as mp

        _require_solutions(mp, "lite")
        self._model = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=SIGN.lite_model_complexity,
            min_detection_confidence=SIGN.min_pose_detection_confidence,
            min_tracking_confidence=SIGN.min_tracking_confidence,
        )

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        result = self._model.process(_readonly(frame_rgb))
        if not result.pose_landmarks:
            return None
        return _normalized_landmarks_to_array(result.pose_landmarks.landmark)

    def close(self) -> None:
        self._model.close()


_SWAP_LABEL = {"Left": "Right", "Right": "Left"}


class _LiteHands:
    """mp.solutions Hands at ``SIGN.lite_model_complexity``, up to two hands."""

    def __init__(self) -> None:
        import mediapipe as mp

        _require_solutions(mp, "lite")
        self._model = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=SIGN.lite_model_complexity,
            max_num_hands=2,
            min_detection_confidence=SIGN.min_hand_detection_confidence,
            min_tracking_confidence=SIGN.min_tracking_confidence,
        )

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[tuple[np.ndarray, str]]:
        result = self._model.process(_readonly(frame_rgb))
        # mp.solutions Hands labels handedness as if the image were mirrored (selfie
        # view); the Tasks HandLandmarker does not. Swap to the Tasks convention so a
        # hand lands in the same slot as in the Tasks-recorded training data (verified
        # on the same image: tasks -> right slot, raw lite -> left slot).
        detections = []
        for marks, handed in zip(
            result.multi_hand_landmarks or [], result.multi_handedness or [], strict=False
        ):
            label = handed.classification[0].label
            detections.append(
                (_normalized_landmarks_to_array(marks.landmark), _SWAP_LABEL.get(label, label))
            )
        return detections

    def close(self) -> None:
        self._model.close()


_POSE_ESTIMATORS = {"tasks": _TasksPose, "lite": _LitePose}
_HANDS_ESTIMATORS = {"tasks": _TasksHands, "lite": _LiteHands}
SPLITTABLE_BACKENDS: tuple[str, ...] = tuple(_POSE_ESTIMATORS)


def make_pose_estimator(family: str) -> PoseEstimator:
    """Build the pose half of the "tasks" or "lite" backend."""
    if family not in _POSE_ESTIMATORS:
        raise ValueError(f"No separate pose estimator for backend {family!r}")
    return _POSE_ESTIMATORS[family]()


def make_hands_estimator(family: str) -> HandsEstimator:
    """Build the hands half of the "tasks" or "lite" backend."""
    if family not in _HANDS_ESTIMATORS:
        raise ValueError(f"No separate hands estimator for backend {family!r}")
    return _HANDS_ESTIMATORS[family]()


class _SerialBackend(PerceptionBackend):
    """Runs a pose estimator and a hands estimator one after the other."""

    family: str

    def __init__(self) -> None:
        self._pose = make_pose_estimator(self.family)
        self._hands = make_hands_estimator(self.family)

    def extract(self, frame_rgb: np.ndarray, timestamp_ms: int) -> RawLandmarks:
        pose = self._pose.process(frame_rgb, timestamp_ms)
        left, right = slot_hands(self._hands.process(frame_rgb, timestamp_ms), SIGN.mirror)
        return RawLandmarks(left_hand=left, right_hand=right, pose=pose)

    def close(self) -> None:
        self._pose.close()
        self._hands.close()


class TasksBackend(_SerialBackend):
    """MediaPipe Tasks HandLandmarker + PoseLandmarker, composed (face-free)."""

    family = "tasks"


class LiteBackend(_SerialBackend):
    """Legacy ``mp.solutions`` Hands + Pose at ``SIGN.lite_model_complexity`` (face-free).

    The fastest models on a Pi 4: the lite hand model takes 95 ms/frame there vs
    265 ms for the Tasks full model. Same 21 hand / 33 pose landmarks, and hands are
    slotted to match ``TasksBackend`` (see ``_LiteHands``), so the feature vector is
    unchanged; the lite model is just a little less precise. Needs a MediaPipe that
    still ships ``mp.solutions`` (0.10.18 on the Pi). The pose lite model is fetched
    into the mediapipe package on first use, so run it once while online.
    """

    family = "lite"


class HolisticBackend(PerceptionBackend):
    """Legacy ``mp.solutions.Holistic``; face landmarks are computed but ignored.

    Kept for the on-Pi benchmark only. Holistic runs the face mesh internally
    with no flag to disable it, so it pays face-mesh CPU we then discard. It is one
    graph, so it cannot be split across processes like tasks/lite.
    """

    def __init__(self) -> None:
        import mediapipe as mp

        _require_solutions(mp, "holistic")
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


def create_backend(name: str | None = None, parallel: bool | None = None) -> PerceptionBackend:
    """Instantiate a perception backend by name (one of ``BACKEND_NAMES``).

    ``parallel`` (default ``SIGN.parallel_perception``) runs the pose and hands
    models in separate worker processes; only "tasks" and "lite" can be split.
    """
    name = (name or SIGN.perception_backend).lower()
    if name not in BACKEND_NAMES:
        raise ValueError(
            f"Unknown perception backend {name!r} (expected one of {', '.join(BACKEND_NAMES)})"
        )
    if SIGN.parallel_perception if parallel is None else parallel:
        if name not in SPLITTABLE_BACKENDS:
            raise ValueError(f"Backend {name!r} can't run in parallel (it is one graph)")
        from senseless.sign.parallel import ParallelBackend

        return ParallelBackend.for_family(name)
    if name == "tasks":
        return TasksBackend()
    if name == "lite":
        return LiteBackend()
    return HolisticBackend()
