"""
Landmark Post-Processor for HandForge.

Responsibilities:
- Transform tracker-specific raw data (FrameResult) into canonical pipeline
  objects (ProcessedFrame) consumed by downstream filtering and output stages.
- Estimate stable real-time FPS using a configuration-driven sliding window
  over hardware acquisition timestamps.
- Provide stateless, side-effect-free visualisation utilities for real-time
  terminal monitoring and deep data inspection.

Design Decisions:
- True Capture Rate: FPS is derived from acquisition timestamps (from the
  original Frame) rather than processing timestamps. This measures actual
  sensor throughput, isolating inference latency jitter from capture rate.
- Immutability: ProcessedFrame is frozen to ensure that once a frame is
  processed, its state (landmarks, FPS) cannot be modified by downstream bugs.
- Decoupled Visualisation: console_summary() is a pure function returning
  formatted strings. It has no knowledge of the logger or console, making
  it easily testable and portable to GUIs.
- Stateless Processing: The processor itself is stateless regarding hand
  identity (tracking IDs); it processes whatever collection of hands the
  upstream tracker provides (0, 1, or 2).
"""

from __future__ import annotations

import collections
import logging
import math

from hand_tracker.logger import get_logger
from hand_tracker.types import (
    LANDMARK_COUNT,
    LANDMARK_NAMES,
    FrameResult,
    ProcessedFrame,
    ProcessedHand,
)
from hand_tracker.utils import (
    US_PER_SEC,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# FPS estimator
# ---------------------------------------------------------------------------


class _FpsEstimator:
    """
    Sliding-window FPS estimator based on hardware acquisition timestamps.

    Uses a total-span approach (Last - First / Intervals) over a sliding
    window for O(1) update performance. This method is more robust than
    simple per-frame deltas as it naturally handles occasional hardware
    jitter without over-reacting.

    Thread Safety:
        Not thread-safe. Should be encapsulated within a single processing
        pipeline instance.
    """

    def __init__(self, window_size: int) -> None:
        """
        Initialise the estimator.

        Parameters
        ----------
        window_size:
            Number of recent frames to include in the average.
            Value MUST be injected from TrackerConfig.fps_window_size.
        """
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        self._window_size = window_size
        # Store acquisition timestamps in microseconds
        self._timestamps: collections.deque[int] = collections.deque(maxlen=window_size)

    def update(self, timestamp_us: int) -> float:
        """
        Record a new frame timestamp and return the current FPS estimate.

        Parameters
        ----------
        timestamp_us:
            Acquisition timestamp from Frame.timestamp_us (microseconds).

        Returns
        -------
        float
            FPS estimate, or float('nan') if fewer than 2 samples present.
        """
        # Guard against backwards clock drift or duplicate timestamps
        if self._timestamps and timestamp_us <= self._timestamps[-1]:
            # In a production pipeline, we treat non-monotonic timestamps as
            # invalid for FPS calculation but don't strictly crash.
            return float("nan")

        self._timestamps.append(timestamp_us)

        if len(self._timestamps) < 2:
            return float("nan")

        # Use (Total Span / Num Intervals)
        # This is O(1) as we only access the head and tail of the deque.
        # This approach is statistically identical to mean(deltas).
        total_span_us = self._timestamps[-1] - self._timestamps[0]
        num_intervals = len(self._timestamps) - 1

        if total_span_us <= 0:
            return float("nan")

        mean_delta_s = (total_span_us / num_intervals) / US_PER_SEC
        return 1.0 / mean_delta_s

    def reset(self) -> None:
        """Clear all buffered timestamps. Used after tracker re-init."""
        self._timestamps.clear()


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class LandmarkProcessor:
    """
    Orchestrator for landmark refinement and frame metadata aggregation.

    Primary driver for the post-processing pipeline. It maintains state
    for FPS estimation across frames but remains agnostic to hand identity
    management (IDs are handled by tracker or downstream stages).

    Usage:
        processor = LandmarkProcessor(window_size=config.fps_window_size)
        processed_frame = processor.update(tracker_output)
    """

    def __init__(self, window_size: int) -> None:
        """
        Initialise the processor.

        Parameters
        ----------
        window_size:
            Smoothing window for FPS. MUST be injected from config at runtime.
            Values less than 2 will raise ValueError.
        """
        if window_size < 2:
            raise ValueError(
                f"window_size must be >= 2 for FPS estimation, got {window_size}. "
                "Check your TrackerConfig.fps_window_size setting."
            )
        self._fps = _FpsEstimator(window_size=window_size)
        self._processed_count: int = 0

    def update(self, result: FrameResult) -> ProcessedFrame:
        """
        Convert a FrameResult into a ProcessedFrame.

        Advances the FPS estimator using result.timestamp_us.

        Parameters
        ----------
        result:
            Output from MediaPipeTracker.process().

        Returns
        -------
        ProcessedFrame
            Frozen frame containing zero or more processed hands.
        """
        fps = self._fps.update(result.timestamp_us)
        self._processed_count += 1

        # Generator → tuple directly: avoids intermediate list allocation
        # to reduce GC pressure in the per-frame hot path.
        processed_hands = tuple(
            ProcessedHand(
                landmarks=raw_hand.landmarks,
                world_landmarks=raw_hand.world_landmarks,
                handedness=raw_hand.handedness,
                confidence=raw_hand.confidence,
            )
            for raw_hand in result.hands
        )

        processed = ProcessedFrame(
            hands=processed_hands,
            timestamp_us=result.timestamp_us,
            frame_index=result.frame_index,
            inference_time_us=result.inference_time_us,
            fps=fps,
        )

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "frame processed",
                extra={
                    "frame_index": result.frame_index,
                    "fps": round(fps, 2) if not math.isnan(fps) else None,
                    "hands_count": len(processed_hands),
                    "inference_time_us": result.inference_time_us,
                    "processed_count": self._processed_count,
                },
            )

        return processed

    def reset(self) -> None:
        """Reset FPS estimator and processed frame counter."""
        self._fps.reset()
        self._processed_count = 0
        log.debug("landmark processor reset")

    @property
    def processed_count(self) -> int:
        """Total number of frames processed since creation or last reset."""
        return self._processed_count


# ---------------------------------------------------------------------------
# Debug utilities
# ---------------------------------------------------------------------------


def full_landmark_dump(frame: ProcessedFrame) -> str:
    """
    Return all 21 landmarks for all hands as a multi-line string.
    """
    lines: list[str] = [
        f"  [full dump] frame={frame.frame_index}  ts={frame.timestamp_us}"
    ]

    if not frame.hands:
        lines.append("  [ NO HANDS ]")
        return "\n".join(lines)

    for i, hand in enumerate(frame.hands):
        lines.extend(
            [
                f"  --- HAND #{i + 1} ({hand.handedness}) ---",
                f"  {'IDX':<4} {'NAME':<14}  "
                f"{'pos_x':>7} {'pos_y':>7} {'pos_z':>8}    "
                f"{'wld_x':>8} {'wld_y':>8} {'wld_z':>8}",
            ]
        )
        for j in range(LANDMARK_COUNT):
            p = hand.landmarks[j]
            w = hand.world_landmarks[j]
            lines.append(
                f"  {j:<4} {LANDMARK_NAMES[j]:<14}  "
                f"{p[0]:7.4f} {p[1]:7.4f} {p[2]:8.4f}    "
                f"{w[0]:8.4f} {w[1]:8.4f} {w[2]:8.4f}"
            )
    return "\n".join(lines)
