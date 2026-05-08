"""
hand_tracker — MediaPipe hand tracking pipeline for HandForge.

Public API surface. Import from this module in application code;
do not import from sub-modules directly except in tests.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("handforge-python")
except PackageNotFoundError:
    # Editable install not yet built, or running from source tree
    __version__ = "0.0.0+unknown"

# Capture
from hand_tracker.capture import CaptureError, WebcamCapture

# Config
from hand_tracker.config import AppConfig, load_config

# Landmark processing
from hand_tracker.landmark_processor import (
    LandmarkProcessor,
    full_landmark_dump,
)

# Logging
from hand_tracker.logger import AsyncLoggerLifecycle, get_logger

# Tracking
from hand_tracker.mediapipe_tracker import MediaPipeTracker

# Types
from hand_tracker.types import (
    LANDMARK_NAMES,
    Frame,
    ProcessedFrame,
    RawHandResult,
)

# Utils
from hand_tracker.utils import console_summary

# Visualization
from hand_tracker.viewer import DebugViewer

__all__: list[str] = [
    "LANDMARK_NAMES",
    "AppConfig",
    "AsyncLoggerLifecycle",
    "CaptureError",
    "DebugViewer",
    "Frame",
    "LandmarkProcessor",
    "MediaPipeTracker",
    "ProcessedFrame",
    "RawHandResult",
    "WebcamCapture",
    "__version__",
    "console_summary",
    "full_landmark_dump",
    "get_logger",
    "load_config",
]
