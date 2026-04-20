"""
Domain models and shared types for HandForge.

This module centralizes all data structures used across the hand tracking pipeline
to ensure consistency, prevent circular dependencies, and provide a clear
interface for downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
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


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """
    Single landmark in both coordinate spaces.

    position:
        Normalised image coordinates (x, y, z).
    world_position:
        Metric 3D coordinates in metres (wx, wy, wz).
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
    """MediaPipe-agnostic result for one detected hand."""

    landmarks: tuple[LandmarkPoint, ...]
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
    """Processed data for a single hand (filtered/refined)."""

    landmarks: tuple[LandmarkPoint, ...]
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
