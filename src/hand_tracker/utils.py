"""
Shared utility functions and constants for HandForge.
"""

from __future__ import annotations

import math
from typing import Final

from hand_tracker.types import ProcessedFrame

# Time unit conversions
US_PER_SEC: Final[int] = 1_000_000  # microseconds per second
US_PER_MS: Final[int] = 1_000  # microseconds per millisecond
MS_PER_SEC: Final[int] = 1_000  # milliseconds per second
NS_PER_US: Final[int] = 1_000  # nanoseconds per microsecond


# Console UI Aesthetics
CONSOLE_WIDTH: Final[int] = 100

# Wrist + fingertip indices for summary view
_SUMMARY_INDICES: Final[tuple[int, ...]] = (0, 4, 8, 12, 16, 20)
_SUMMARY_NAMES: Final[dict[int, str]] = {
    0: "WRIST     ",
    4: "THUMB_TIP ",
    8: "INDEX_TIP ",
    12: "MIDDLE_TIP",
    16: "RING_TIP  ",
    20: "PINKY_TIP ",
}


def console_summary(frame: ProcessedFrame) -> str:
    """
    Return a human-readable multi-line summary of a ProcessedFrame.
    """
    fps_str = f"{frame.fps:6.1f}" if not math.isnan(frame.fps) else "   ---"
    lines: list[str] = [
        "═" * CONSOLE_WIDTH,
        (
            f" FRAME: {frame.frame_index:>6}  │  "
            f"TS: {frame.timestamp_us}  │  "
            f"FPS: {fps_str}  │  "
            f"Inference: {frame.inference_time_us / US_PER_MS:5.1f}ms"
        ),
        "─" * CONSOLE_WIDTH,
    ]

    if not frame.hands:
        lines.append("  [!] NO HANDS DETECTED")
        lines.append("═" * CONSOLE_WIDTH)
        return "\n".join(lines)

    for i, hand in enumerate(frame.hands):
        lines.extend(
            [
                "",
                f"  HAND #{i + 1}: {hand.handedness} (conf={hand.confidence:.2f})",
                f"  {'NAME':<12}  "
                f"{'pos_x':>7} {'pos_y':>7} {'pos_z':>8}    "
                f"{'wld_x':>8} {'wld_y':>8} {'wld_z':>8}",
                f"  {'':─<12}  {'':─>7} {'':─>7} {'':─>8}    {'':─>8} {'':─>8} {'':─>8}",
            ]
        )

        for idx in _SUMMARY_INDICES:
            p = hand.landmarks[idx]
            w = hand.world_landmarks[idx]
            name = _SUMMARY_NAMES[idx]
            lines.append(
                f"  {name}  "
                f"{p[0]:7.4f} {p[1]:7.4f} {p[2]:8.4f}    "
                f"{w[0]:8.4f} {w[1]:8.4f} {w[2]:8.4f}"
            )

    lines.append("═" * CONSOLE_WIDTH)
    return "\n".join(lines)
