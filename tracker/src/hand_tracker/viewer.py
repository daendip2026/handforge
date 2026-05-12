"""
Visualization utilities for HandForge.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np

from hand_tracker.types import Frame, FrameResult

# Standard MediaPipe Hand Connections (Pairs of landmark indices)
HAND_CONNECTIONS: Final[list[tuple[int, int]]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (0, 17),  # Palm base
]

# Aesthetics
COLOR_SKELETON: Final[tuple[int, int, int]] = (245, 117, 66)  # Blueish-orange
COLOR_LANDMARK: Final[tuple[int, int, int]] = (66, 245, 117)  # Greenish
COLOR_TEXT: Final[tuple[int, int, int]] = (255, 255, 255)  # White


class DebugViewer:
    """
    OpenCV-based debug window for real-time visualization of hand tracking.
    """

    def __init__(self, window_name: str = "HandForge Debug Viewer") -> None:
        self._window_name = window_name
        self._is_active = False

    def render(self, frame: Frame, result: FrameResult) -> bool:
        """
        Draw landmarks on the frame and display it.
        """
        if frame.bgr is None or frame.bgr.size == 0:
            return True

        if not self._is_active:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            # Potentially helpful for some Windows OpenCV versions to keep GUI responsive
            if hasattr(cv2, "startWindowThread"):
                cv2.startWindowThread()
            self._is_active = True

        canvas = frame.bgr.copy()
        h, w = canvas.shape[:2]

        # Health Check: If the frame is almost pitch black, warn the user
        if np.mean(canvas) < 1.0:
            cv2.putText(
                canvas,
                "WARNING: NO CAMERA SIGNAL (BLACK FRAME)",
                (20, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Check Hardware or Resolution Settings",
                (20, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw detected hands
        for hand in result.hands:
            # Draw skeleton
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = hand.landmarks[start_idx]
                p2 = hand.landmarks[end_idx]
                pt1 = (int(p1[0] * w), int(p1[1] * h))
                pt2 = (int(p2[0] * w), int(p2[1] * h))
                cv2.line(canvas, pt1, pt2, COLOR_SKELETON, 2, cv2.LINE_AA)

            # Draw landmarks
            for i in range(len(hand.landmarks)):
                p = hand.landmarks[i]
                pt = (int(p[0] * w), int(p[1] * h))
                cv2.circle(canvas, pt, 4, COLOR_LANDMARK, -1, cv2.LINE_AA)

            # Draw labels
            wrist = hand.landmarks[0]
            label_pos = (int(wrist[0] * w), int(wrist[1] * h) - 20)
            cv2.putText(
                canvas,
                f"{hand.handedness.name} ({hand.confidence:.2f})",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2,
                cv2.LINE_AA,
            )

        # Dashboard overlay
        status_text = f"Frame: {frame.frame_index} | Hands: {len(result.hands)} | Inf: {result.inference_time_us / 1000:.1f}ms"
        cv2.putText(
            canvas,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(self._window_name, canvas)

        # Increased waitKey slightly for GUI event loop stability
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or 'ESC'
            return False

        # Proper window close detection
        try:
            if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except Exception:
            # On some platforms, this might throw if window is already gone
            return False

        return True

    def close(self) -> None:
        """Close the debug window."""
        if self._is_active:
            cv2.destroyWindow(self._window_name)
            self._is_active = False
