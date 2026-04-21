"""
Domain models and shared types for HandForge.

This module centralizes all data structures used across the hand tracking pipeline
to ensure consistency, prevent circular dependencies, and provide a clear
interface for downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Frame:
    """
    A single captured video frame with acquisition metadata.
    """

    bgr: npt.NDArray[np.uint8]
    timestamp_us: int
    frame_index: int
    is_mirrored: bool


class Handedness(StrEnum):
    """Enumeration for hand side classification."""

    LEFT = "Left"
    RIGHT = "Right"
    BOTH = "Both"


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


LANDMARK_COUNT: Final[int] = len(HandLandmark)
LANDMARK_NAMES: Final[tuple[str, ...]] = tuple(lm.name for lm in HandLandmark)


@dataclass(frozen=True, slots=True)
class RawHandResult:
    """
    MediaPipe-agnostic Raw Detection Result for one hand.

    This represents the 'Sensor/Link Layer' output directly from the tracker.
    It should be treated as immutable historical evidence of what the model saw.

    landmarks:
        (21, 3) float32 array of (x, y, z) in normalized [0, 1] space.
    world_landmarks:
        (21, 3) float32 array of (x, y, z) in metric metres.
    """

    landmarks: npt.NDArray[np.float32]
    world_landmarks: npt.NDArray[np.float32]
    handedness: Handedness
    confidence: float
    timestamp_us: int
    inference_time_us: int


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Collection of all hands detected in a single frame (Raw)."""

    hands: tuple[RawHandResult, ...]
    timestamp_us: int
    frame_index: int
    is_mirrored: bool
    inference_time_us: int


@dataclass(frozen=True, slots=True)
class ProcessedHand:
    """
    Refined and Filtered data for one hand, ready for application use.

    This represents the 'Application/Network Layer' data.
    Note: 'landmarks' here may contain smoothed values (EMA, One-Euro) or
    coordinate-transformed values (e.g. Unity space) that differ from the raw detection.
    """

    landmarks: npt.NDArray[np.float32]
    world_landmarks: npt.NDArray[np.float32]
    handedness: Handedness
    confidence: float


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """Canonical representation of a processed multi-hand frame."""

    hands: tuple[ProcessedHand, ...]
    timestamp_us: int
    frame_index: int
    is_mirrored: bool
    inference_time_us: int
    fps: float
