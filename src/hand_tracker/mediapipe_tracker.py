"""
MediaPipe Hands Inference Module for HandForge.

This module provides a strictly typed wrapper around MediaPipe Hands, decoupling
downstream code from the MediaPipe API and ensuring data immutability.

Key Responsibilities:
    - Lifecycle Management: Owns the init → process → close lifecycle of the solution.
    - Data Decoupling: Maps raw proto-wrappers to FrameResult/RawHandResult.
    - Dual Coordinate Spaces:
        - position: Normalised image coords (x,y in [0,1], z=depth).
        - world_position: Metric 3D coords in metres (wrist-relative).

Design Decisions:
    - Structural Typing (Protocols): MediaPipe objects are typed via Protocols
      (MPResults, MPHandsSolution, etc.) instead of 'Any'. This ensures 100%
      type safety and enables robust Dependency Injection without requiring
      the heavy MediaPipe library during unit tests.
    - Result Collection: process() always returns a FrameResult. Empty detections
      are represented as an empty 'hands' tuple rather than None, making the
      collection handling consistent for the caller.

Thread Safety & Concurrency:
    - MediaPipeTracker is **NOT** thread-safe. One instance per thread/camera.
    - Output data (FrameResult, RawHandResult) is **Immutable** (using frozen
      dataclasses and tuples) and completely safe for concurrent sharing
      across threads after creation.

Hardware & Performance:
    - CPU Execution: MediaPipe Python runs on CPU (no DirectML/OpenCL support).
    - model_complexity=1: Required for world_landmarks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, NamedTuple, Protocol, runtime_checkable

import cv2
import mediapipe as mp
import numpy as np

from hand_tracker.capture import Frame
from hand_tracker.config import Handedness, MediaPipeConfig, TrackerConfig
from hand_tracker.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocols (MediaPipe Contract)
# ---------------------------------------------------------------------------


@runtime_checkable
class MPLandmark(Protocol):
    """Contract for a single MediaPipe landmark."""

    x: float
    y: float
    z: float


@runtime_checkable
class MPLandmarkList(Protocol):
    """Contract for a list of MediaPipe landmarks."""

    landmark: list[MPLandmark]


@runtime_checkable
class MPClassification(Protocol):
    """Contract for a MediaPipe classification result."""

    label: str
    score: float


@runtime_checkable
class MPCategory(Protocol):
    """Contract for a MediaPipe classification category (Handedness)."""

    classification: list[MPClassification]


@runtime_checkable
class MPResults(Protocol):
    """Contract for MediaPipe Hands solution results."""

    @property
    def multi_hand_landmarks(self) -> list[MPLandmarkList] | None: ...

    @property
    def multi_hand_world_landmarks(self) -> list[MPLandmarkList] | None: ...

    @property
    def multi_handedness(self) -> list[MPCategory] | None: ...


@runtime_checkable
class MPHandsModule(Protocol):
    """Contract for the MediaPipe Hands solution module/factory."""

    def Hands(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> MPHandsSolution: ...


@runtime_checkable
class MPHandsSolution(Protocol):
    """Contract for a MediaPipe Hands solution instance."""

    def process(self, image: np.ndarray) -> MPResults: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Constants & Specifications
# ---------------------------------------------------------------------------

# Time unit conversions
US_PER_SEC: Final[int] = 1_000_000  # microseconds per second
NS_PER_US: Final[int] = 1_000  # nanoseconds per microsecond


class HandLandmark(IntEnum):
    """MediaPipe hand landmark indices."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Derived specifications
LANDMARK_COUNT: Final[int] = len(HandLandmark)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MediaPipeError(Exception):
    """Base exception for all MediaPipe tracker errors."""


class MediaPipeInferenceError(MediaPipeError):
    """Raised when inference results are malformed or inconsistent."""


class MediaPipeConfigurationError(MediaPipeError):
    """Raised when the MediaPipe solution is misconfigured."""


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """
    Single landmark in both coordinate spaces.

    position:
        Normalised image coordinates.
        x, y in [0.0, 1.0] relative to frame width/height.
        z is relative depth (negative = closer to camera).
        Used for Unity IK normalised input.

    world_position:
        Metric 3D coordinates in metres.
        Origin is the wrist (landmark 0).
        Real-world scale; used for physical interaction simulation.
    """

    index: int
    name: str

    # Normalised image space
    x: float
    y: float
    z: float

    # World space (metres)
    wx: float
    wy: float
    wz: float


@dataclass(frozen=True, slots=True)
class RawHandResult:
    """
    Fully typed, MediaPipe-agnostic result for one detected hand.

    landmarks:
        Tuple of exactly LANDMARK_COUNT (21) LandmarkPoints,
        ordered by MediaPipe landmark index.
    handedness:
        Handedness classification (Left or Right).
    confidence:
        Handedness classification confidence in [0.0, 1.0].
    timestamp_us:
        Inherited from the source Frame — acquisition time in microseconds.
    frame_index:
        Inherited from the source Frame.
    inference_time_us:
        Time spent inside mp_hands.process() in microseconds.
    """

    landmarks: tuple[LandmarkPoint, ...]
    handedness: Handedness
    confidence: float
    timestamp_us: int
    frame_index: int
    inference_time_us: int


@dataclass(frozen=True, slots=True)
class FrameResult:
    """
    Collection of all hands detected and filtered in a single frame.

    hands:
        Tuple of RawHandResults for each detected hand.
    timestamp_us:
        Inherited from the source Frame — acquisition time.
    frame_index:
        Inherited from the source Frame.
    inference_time_us:
        Time spent inside mp_hands.process() in microseconds.
    """

    hands: tuple[RawHandResult, ...]
    timestamp_us: int
    frame_index: int
    inference_time_us: int


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------


class _LandmarkLists(NamedTuple):
    """Pair of mediapipe NormalizedLandmarkList and LandmarkList."""

    image: MPLandmarkList
    world: MPLandmarkList


def _extract_landmark_lists(
    results: MPResults,
    hand_index: int,
) -> _LandmarkLists:
    """
    Pull the (image, world) landmark lists for a given hand index.

    Both lists are guaranteed to be present when model_complexity=1
    and a hand is detected.

    Raises
    ------
    MediaPipeInferenceError
        If world_landmarks are missing despite model_complexity=1.
    """
    try:
        # Fetch results
        img_raw = results.multi_hand_landmarks
        wld_raw = results.multi_hand_world_landmarks

        # Type Guard
        if img_raw is None or wld_raw is None:
            raise MediaPipeInferenceError("MediaPipe returned None for landmarks.")

        # Safe Indexing
        return _LandmarkLists(image=img_raw[hand_index], world=wld_raw[hand_index])
    except (IndexError, AttributeError, TypeError) as e:
        raise MediaPipeInferenceError(
            f"Failed to extract landmark lists for hand {hand_index}: {e}"
        ) from e


def _build_landmark_points(lists: _LandmarkLists) -> tuple[LandmarkPoint, ...]:
    """Convert raw mediapipe landmark objects into typed LandmarkPoints."""
    points: list[LandmarkPoint] = []

    image_landmarks = lists.image.landmark
    world_landmarks = lists.world.landmark

    for i in range(LANDMARK_COUNT):
        img = image_landmarks[i]
        wld = world_landmarks[i]

        points.append(
            LandmarkPoint(
                index=i,
                name=HandLandmark(i).name,
                x=float(img.x),
                y=float(img.y),
                z=float(img.z),
                wx=float(wld.x),
                wy=float(wld.y),
                wz=float(wld.z),
            )
        )

    return tuple(points)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class MediaPipeTracker:
    """
    Lifecycle-managed MediaPipe Hands wrapper.

    Usage
    -----
    ::

        cfg_mp = MediaPipeConfig()
        cfg_tr = TrackerConfig()
        with MediaPipeTracker(cfg_mp, cfg_tr) as tracker:
            for frame in capture:
                result = tracker.process(frame)
                if result.hands:
                    handle(result.hands)

    The context manager calls solution.close() on exit, releasing
    MediaPipe's internal TFLite interpreter resources.
    """

    def __init__(
        self,
        mp_cfg: MediaPipeConfig,
        tracker_cfg: TrackerConfig,
        hands_module: MPHandsModule | None = None,
    ) -> None:
        """
        Initialise the tracker.

        Parameters
        ----------
        mp_cfg: MediaPipeConfig
            Inference performance and model settings.
        tracker_cfg: TrackerConfig
            Application-level tracking and filtering settings.
        hands_module: MPHandsModule, optional
            A mock or alternative MediaPipe hands module for dependency
            injection during testing. If None, the standard mediapipe
            library is used.
        """
        self._mp_cfg = mp_cfg
        self._tracker_cfg = tracker_cfg
        self._hands_module = hands_module
        self._hands: MPHandsSolution | None = None
        self._warmup_remaining: int = mp_cfg.warmup_frame_count

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> MediaPipeTracker:
        # Production: Use the real mediapipe library.
        # Testing: Use the injected mock module to avoid loading ML models.
        mp_hands = self._hands_module or mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=self._mp_cfg.static_image_mode,
            max_num_hands=self._mp_cfg.max_num_hands,
            model_complexity=self._mp_cfg.model_complexity,
            min_detection_confidence=self._mp_cfg.min_detection_confidence,
            min_tracking_confidence=self._mp_cfg.min_tracking_confidence,
        )

        log.info(
            "MediaPipe Hands initialised",
            extra={
                "model_complexity": self._mp_cfg.model_complexity,
                "max_num_hands": self._mp_cfg.max_num_hands,
                "min_detection_confidence": self._mp_cfg.min_detection_confidence,
                "min_tracking_confidence": self._mp_cfg.min_tracking_confidence,
                "primary_hand": self._tracker_cfg.primary_hand,
                "warmup_frames": self._mp_cfg.warmup_frame_count,
            },
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None
            log.info("MediaPipe Hands closed")

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(self, frame: Frame) -> FrameResult:
        """
        Run MediaPipe Hands inference on a single frame.

        Parameters
        ----------
        frame:
            Frame from WebcamCapture.  BGR format; converted to RGB
            internally before passing to MediaPipe.

        Returns
        -------
        FrameResult
            Container with zero, one, or multiple RawHandResults,
            depending on detection and primary_hand config.
            Empty during warmup frames or when no hands are detected.

        Raises
        ------
        MediaPipeConfigurationError
            If called outside the context manager.
        """
        if self._hands is None:
            raise MediaPipeConfigurationError(
                "MediaPipeTracker must be used as a context manager before calling process()."
            )

        # Standard return for no-op states (warmup, empty frame)
        empty_result = FrameResult(
            hands=(),
            timestamp_us=frame.timestamp_us,
            frame_index=frame.frame_index,
            inference_time_us=0,
        )

        # Production-grade input validation
        if frame.bgr is None or frame.bgr.size == 0:
            log.warning("received empty frame; skipping inference")
            return empty_result

        # BGR → RGB: MediaPipe expects RGB input
        rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)

        # Marking the array as non-writeable passes a zero-copy reference
        # to MediaPipe and prevents accidental mutation of the frame buffer.
        rgb.flags.writeable = False

        t_start_ns = time.perf_counter_ns()
        results = self._hands.process(rgb)
        inference_time_us = (time.perf_counter_ns() - t_start_ns) // NS_PER_US

        rgb.flags.writeable = True

        # Discard warmup frames
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            log.debug(
                "warmup frame discarded",
                extra={
                    "frame_index": frame.frame_index,
                    "warmup_remaining": self._warmup_remaining,
                    "inference_time_us": inference_time_us,
                },
            )
            return FrameResult(
                hands=(),
                timestamp_us=frame.timestamp_us,
                frame_index=frame.frame_index,
                inference_time_us=inference_time_us,
            )

        if results.multi_hand_landmarks is None:
            log.debug(
                "no hands detected",
                extra={
                    "frame_index": frame.frame_index,
                    "inference_time_us": inference_time_us,
                },
            )
            return FrameResult(
                hands=(),
                timestamp_us=frame.timestamp_us,
                frame_index=frame.frame_index,
                inference_time_us=inference_time_us,
            )

        # Process all detected hands and filter by primary_hand config
        detected_hands: list[RawHandResult] = []
        target_side = self._tracker_cfg.primary_hand

        for idx in range(len(results.multi_hand_landmarks)):
            # Robustness check: Ensure handedness data is present for the detected hand
            if results.multi_handedness is None or idx >= len(results.multi_handedness):
                raise MediaPipeInferenceError(
                    f"Handedness data missing for hand index {idx} despite landmark detection."
                )

            # Extract handedness metadata
            # multi_handedness is guaranteed non-None by the check above
            classification = results.multi_handedness[idx].classification[0]
            handedness = Handedness(classification.label)
            confidence: float = float(classification.score)

            # Filtering logic: Include if "Both" or if it matches requested side
            if target_side != Handedness.BOTH and handedness != target_side:
                continue

            landmark_lists = _extract_landmark_lists(results, idx)
            landmarks = _build_landmark_points(landmark_lists)

            detected_hands.append(
                RawHandResult(
                    landmarks=landmarks,
                    handedness=handedness,
                    confidence=confidence,
                    timestamp_us=frame.timestamp_us,
                    frame_index=frame.frame_index,
                    inference_time_us=inference_time_us,
                )
            )

        log.debug(
            "inference complete",
            extra={
                "frame_index": frame.frame_index,
                "detected_count": len(results.multi_hand_landmarks),
                "filtered_count": len(detected_hands),
                "inference_time_us": inference_time_us,
            },
        )

        return FrameResult(
            hands=tuple(detected_hands),
            timestamp_us=frame.timestamp_us,
            frame_index=frame.frame_index,
            inference_time_us=inference_time_us,
        )
