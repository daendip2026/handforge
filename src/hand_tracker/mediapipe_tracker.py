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

import threading
import time
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_tracker.config import MediaPipeConfig, TrackerConfig
from hand_tracker.logger import get_logger
from hand_tracker.types import (
    LANDMARK_COUNT,
    Frame,
    FrameResult,
    Handedness,
    RawHandResult,
)
from hand_tracker.utils import NS_PER_US, US_PER_MS

log = get_logger(__name__)

# Performance: Pre-cache Enum values to avoid string-lookup overhead in hot path
_HANDEDNESS_MAP: Final[dict[str, Handedness]] = {
    "Right": Handedness.RIGHT,
    "Left": Handedness.LEFT,
    "Both": Handedness.BOTH,
}


# ---------------------------------------------------------------------------
# Protocols (MediaPipe Contract)
# ---------------------------------------------------------------------------


@runtime_checkable
class MPLandmark(Protocol):
    """Contract for a single MediaPipe landmark (Normalized or World)."""

    x: float
    y: float
    z: float


@runtime_checkable
class MPCategory(Protocol):
    """Contract for a MediaPipe classification category."""

    score: float
    index: int
    category_name: str
    display_name: str


@runtime_checkable
class MPHandLandmarkerResult(Protocol):
    """Contract for MediaPipe HandLandmarker inference results."""

    hand_landmarks: list[list[MPLandmark]]
    hand_world_landmarks: list[list[MPLandmark]]
    handedness: list[list[MPCategory]]


@runtime_checkable
class MPHandLandmarker(Protocol):
    """Contract for a MediaPipe HandLandmarker instance."""

    def detect_async(
        self,
        image: mp.Image,
        timestamp_ms: int,
    ) -> None: ...

    def close(self) -> None: ...


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
# Extraction types
# ---------------------------------------------------------------------------


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
        hand_landmarker_factory: type[MPHandLandmarker] | None = None,
    ) -> None:
        """
        Initialise the tracker.

        Parameters
        ----------
        mp_cfg: MediaPipeConfig
            Inference performance and model settings.
        tracker_cfg: TrackerConfig
            Application-level tracking and filtering settings.
        hand_landmarker_factory: type[MPHandLandmarker], optional
            Alternative factory for HandLandmarker (for testing).
        """
        self._mp_cfg = mp_cfg
        self._tracker_cfg = tracker_cfg
        self._factory = hand_landmarker_factory
        self._detector: MPHandLandmarker | None = None
        self._warmup_remaining: int = mp_cfg.warmup_frame_count

        # Async handling state: Lightweight single-slot buffer with lock
        # This replaces queue.Queue which has higher overhead for single-latest-result use cases.
        self._hands_lock = threading.Lock()
        self._latest_hands: tuple[RawHandResult, ...] = ()
        self._latest_inference_time_us: int = 0

        self._last_detected_at_ns: int = 0
        self._frame_count: int = 0
        self._last_processed_timestamp_us: int = -1

        # Zero-Allocation Pool: Pre-allocate target arrays for cv2.cvtColor
        # to strictly prevent massive GC spikes in hot path.
        self._rgb_pool: list[np.ndarray] | None = None
        self._pool_index: int = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> MediaPipeTracker:
        # 1. Validate Model Existence
        model_path = Path(self._mp_cfg.model_path)
        if not model_path.exists():
            raise MediaPipeConfigurationError(
                f"Model file not found at {model_path}. "
                "Please run 'python scripts/download_models.py' first."
            )

        # 2. Setup HandLandmarker
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=self._mp_cfg.max_num_hands,
            min_hand_detection_confidence=self._mp_cfg.min_detection_confidence,
            min_hand_presence_confidence=self._mp_cfg.min_presence_confidence,
            min_tracking_confidence=self._mp_cfg.min_tracking_confidence,
            result_callback=self._on_result,
        )

        if self._factory:
            # Type ignore because the factory is a Protocol mock in tests
            self._detector = self._factory.create_from_options(options)  # type: ignore
        else:
            self._detector = vision.HandLandmarker.create_from_options(options)

        log.info(
            "MediaPipe Tasks HandLandmarker initialised",
            extra={
                "model_path": self._mp_cfg.model_path,
                "max_num_hands": self._mp_cfg.max_num_hands,
                "min_detection_confidence": self._mp_cfg.min_detection_confidence,
                "min_tracking_confidence": self._mp_cfg.min_tracking_confidence,
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
        if self._detector is not None:
            self._detector.close()
            self._detector = None
            log.info("MediaPipe HandLandmarker closed")

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _on_result(
        self,
        results: MPHandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """
        Callback from MediaPipe background thread.

        Extreme-optimized path to minimize Python glue overhead.
        """
        now_ns = time.perf_counter_ns()
        inf_time = (now_ns - self._last_detected_at_ns) // NS_PER_US
        ts_us = timestamp_ms * US_PER_MS

        # Use a generator to build the tuple directly. This avoids the intermediate
        # list allocation ('detected_hands = []'), further reducing GC pressure.
        hands_result: tuple[RawHandResult, ...] = ()

        if results.hand_landmarks:
            # Local attribute caching for hot-path speed
            cfg = self._tracker_cfg
            target_side = cfg.primary_hand

            h_both = Handedness.BOTH

            # Task results lists
            rm_landmarks = results.hand_landmarks
            rm_world = results.hand_world_landmarks
            rm_hands = results.handedness

            # Validation: Ensure all result lists have matching lengths to prevent
            # downstream indexing errors or data misalignment.
            if not (len(rm_landmarks) == len(rm_world) == len(rm_hands)):
                log.error(
                    "MediaPipe inference produced inconsistent result lists",
                    extra={
                        "hand_landmarks": len(rm_landmarks),
                        "hand_world_landmarks": len(rm_world),
                        "handedness": len(rm_hands),
                    },
                )
                raise MediaPipeInferenceError(
                    f"Inconsistent result list lengths: {len(rm_landmarks)}, {len(rm_world)}, {len(rm_hands)}"
                )

            # Validation: Ensure all detected hands have the correct landmark count
            for hand_lms in rm_landmarks:
                if len(hand_lms) != LANDMARK_COUNT:
                    raise MediaPipeInferenceError(
                        f"Expected {LANDMARK_COUNT} landmarks, but got {len(hand_lms)}."
                    )

            hands_result = tuple(
                RawHandResult(
                    # Fast bulk extraction into NumPy arrays to minimize object allocation
                    # and GC pressure. Reshape from flattened list is often faster than nested lists.
                    landmarks=np.array(
                        [[p.x, p.y, p.z] for p in mp_lm], dtype=np.float32
                    ),
                    world_landmarks=np.array(
                        [[w.x, w.y, w.z] for w in mp_wlm], dtype=np.float32
                    ),
                    handedness=_HANDEDNESS_MAP.get(rm_h[0].category_name, h_both),
                    confidence=float(rm_h[0].score),
                    timestamp_us=ts_us,
                    inference_time_us=inf_time,
                )
                for mp_lm, mp_wlm, rm_h in zip(
                    rm_landmarks, rm_world, rm_hands, strict=True
                )
                if target_side == h_both
                or _HANDEDNESS_MAP.get(rm_h[0].category_name, h_both) == target_side
            )

        # Atomic update of the latest hands data
        with self._hands_lock:
            self._latest_hands = hands_result
            self._latest_inference_time_us = inf_time

    def process(self, frame: Frame) -> FrameResult:
        """
        Trigger asynchronous MediaPipe inference.

        This method returns immediately to prevent blocking the webcam loop.
        It returns the latest available result from the internal buffer.

        Parameters
        ----------
        frame:
            Frame from WebcamCapture. BGR format.

        Returns
        -------
        FrameResult
            Contains the MOST RECENT available results (possibly with 1-frame lag).
        """
        if self._detector is None:
            raise MediaPipeConfigurationError(
                "MediaPipeTracker must be used as a context manager before calling process()."
            )

        self._frame_count += 1
        if self._frame_count <= self._mp_cfg.warmup_frame_count:
            return self._make_empty_result(frame)

        # 1. Trigger async inference (Only if this is a new frame)
        if (
            frame.bgr is not None
            and frame.bgr.size > 0
            and frame.timestamp_us > self._last_processed_timestamp_us
        ):
            self._last_processed_timestamp_us = frame.timestamp_us

            # Zero-Allocation Pool: Initialize on first frame reception
            if self._rgb_pool is None or self._rgb_pool[0].shape != frame.bgr.shape:
                # 5 frames pool size is enough for async stream consumer lag
                self._rgb_pool = [np.empty_like(frame.bgr) for _ in range(5)]
                self._pool_index = 0

            # Get next available buffer without allocating memory
            dst_rgb = self._rgb_pool[self._pool_index]
            self._pool_index = (self._pool_index + 1) % 5

            # In-place conversion eliminates 1MB memory allocation overhead
            dst_rgb.flags.writeable = True
            cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB, dst=dst_rgb)
            dst_rgb.flags.writeable = False

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dst_rgb)
            ts_ms = int(frame.timestamp_us / US_PER_MS)

            self._last_detected_at_ns = time.perf_counter_ns()
            self._detector.detect_async(mp_image, ts_ms)

        # 2. Get latest result from buffer (Consumer)
        # We read the latest landmarks and timing under lock, then construct
        # the result object once with the current frame's metadata.
        with self._hands_lock:
            hands = self._latest_hands
            inf_time = self._latest_inference_time_us

        return FrameResult(
            hands=hands,
            timestamp_us=frame.timestamp_us,
            frame_index=frame.frame_index,
            is_mirrored=frame.is_mirrored,
            inference_time_us=inf_time,
        )

    def _make_empty_result(self, frame: Frame) -> FrameResult:
        """Helper to create a consistent empty result for a given frame."""
        return FrameResult(
            hands=(),
            timestamp_us=frame.timestamp_us,
            frame_index=frame.frame_index,
            is_mirrored=frame.is_mirrored,
            inference_time_us=0,
        )
