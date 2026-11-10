"""
Hand position estimate: where the hand is, in metres, relative to the camera.

MediaPipe gives two things per hand and neither is a position: image
landmarks (where the joints appear in the frame) and world landmarks (the
hand's shape in metres with the origin inside the hand). Treating the world
landmarks as the object's shape and the image landmarks as its projection,
the hand's translation T follows from the pinhole model as a linear least
squares problem (two equations per joint, three unknowns). Least squares
returns some T for any input, so each result carries a depth check (every
joint in front of the camera) and the reprojection error for diagnosis.
Camera coordinates: origin at the lens centre, +X right, +Y down, +Z along
the view direction, metres. The frame given to MediaPipe must not be
mirrored, because the world landmarks use the axes of the frame they came
from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hand_tracker.config import CameraConfig
from hand_tracker.types import RawHandResult

_WRIST = 0


@dataclass(frozen=True, slots=True)
class Intrinsics:
    """Pinhole camera parameters in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_from_fov(
    width: int, height: int, diagonal_fov_deg: float, full_frame_aspect: tuple[int, int]
) -> Intrinsics:
    """
    Derive the intrinsics from the frame size and the spec-sheet diagonal FOV.

    The vertical FOV of the full-frame mode is used because every supported
    mode keeps the full vertical extent: fx = fy = (height / 2) / tan(VFOV / 2),
    principal point at the centre. Supported aspect ratios are the full frame
    (16:9 by default) and 4:3 as a centre crop of it; anything else raises.
    """
    aspect = width / height
    full_w, full_h = full_frame_aspect
    full_aspect = full_w / full_h
    if not (
        math.isclose(aspect, full_aspect, rel_tol=0.01)
        or math.isclose(aspect, 4 / 3, rel_tol=0.01)
    ):
        raise ValueError(
            f"{width}x{height}: only the full-frame aspect {full_w}:{full_h} and a "
            "4:3 centre crop with the same vertical extent are supported"
        )
    tan_half_vfov = (
        math.tan(math.radians(diagonal_fov_deg) / 2)
        * full_h
        / math.hypot(full_w, full_h)
    )
    f = (height / 2) / tan_half_vfov
    return Intrinsics(fx=f, fy=f, cx=width / 2, cy=height / 2)


def to_pixels(
    image_landmarks: npt.NDArray[np.floating], width: int, height: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Normalized image landmarks (n, >=2) -> pixel columns (u, v) of the unmirrored frame."""
    xy = np.asarray(image_landmarks, dtype=np.float64)
    return xy[:, 0] * width, xy[:, 1] * height


def build_rows(
    object_points: npt.NDArray[np.floating],
    u: npt.NDArray[np.floating],
    v: npt.NDArray[np.floating],
    intr: Intrinsics,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Two linear equations in T per joint -> A (2n x 3), b (2n)."""
    pts = np.asarray(object_points, dtype=np.float64)
    du = np.asarray(u, dtype=np.float64) - intr.cx
    dv = np.asarray(v, dtype=np.float64) - intr.cy
    n = pts.shape[0]
    a = np.zeros((2 * n, 3), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    a[0::2, 0] = intr.fx
    a[0::2, 2] = -du
    b[0::2] = du * pts[:, 2] - intr.fx * pts[:, 0]
    a[1::2, 1] = intr.fy
    a[1::2, 2] = -dv
    b[1::2] = dv * pts[:, 2] - intr.fy * pts[:, 1]
    return a, b


def solve(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Least-squares T: minimises the sum of squared residuals of A T - b."""
    t, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.asarray(t, dtype=np.float64)


def depths(
    object_points: npt.NDArray[np.floating], t: npt.NDArray[np.floating]
) -> npt.NDArray[np.float64]:
    """Per-joint camera depth Z_i + Tz."""
    z = np.asarray(object_points, dtype=np.float64)[:, 2] + float(t[2])
    return np.asarray(z, dtype=np.float64)


def reprojection_rms_px(
    object_points: npt.NDArray[np.floating],
    t: npt.NDArray[np.floating],
    u: npt.NDArray[np.floating],
    v: npt.NDArray[np.floating],
    intr: Intrinsics,
) -> float:
    """RMS pixel distance between re-projected and measured joints; inf if any depth <= 0."""
    pts = np.asarray(object_points, dtype=np.float64) + np.asarray(t, dtype=np.float64)
    z = pts[:, 2]
    if np.any(z <= 0):
        return math.inf
    du = intr.fx * pts[:, 0] / z + intr.cx - u
    dv = intr.fy * pts[:, 1] / z + intr.cy - v
    return float(np.sqrt(np.mean(du * du + dv * dv)))


def hand_pixel_size(u: npt.NDArray[np.floating], v: npt.NDArray[np.floating]) -> float:
    """Diagonal of the joints' bounding box in pixels; the scale the reprojection error grows with."""
    return float(math.hypot(np.max(u) - np.min(u), np.max(v) - np.min(v)))


@dataclass(frozen=True, slots=True)
class WristEstimate:
    """
    Result of one estimate.

    position is the wrist in camera coordinates (metres), or None when a joint
    lies at or behind the camera. The reprojection fields are diagnostics for
    the log; they do not decide whether the position is kept.
    """

    position: npt.NDArray[np.float32] | None
    depth_ok: bool
    reprojection_px: float
    reprojection_ratio: float


def estimate_wrist(
    image_landmarks: npt.NDArray[np.floating],
    world_landmarks: npt.NDArray[np.floating],
    width: int,
    height: int,
    intr: Intrinsics,
) -> WristEstimate:
    """Solve T from one hand's two landmark sets and check its depth."""
    u, v = to_pixels(image_landmarks, width, height)
    a, b = build_rows(world_landmarks, u, v, intr)
    t = solve(a, b)
    depth_ok = bool(np.all(depths(world_landmarks, t) > 0))
    rms = reprojection_rms_px(world_landmarks, t, u, v, intr)
    size = hand_pixel_size(u, v)
    ratio = rms / size if size > 0 else math.inf
    position = (
        (t + np.asarray(world_landmarks, dtype=np.float64)[_WRIST]).astype(np.float32)
        if depth_ok
        else None
    )
    return WristEstimate(
        position=position,
        depth_ok=depth_ok,
        reprojection_px=rms,
        reprojection_ratio=ratio,
    )


class HandPositionEstimator:
    """Per-session estimator: fixes the frame size and intrinsics once."""

    def __init__(self, width: int, height: int, intr: Intrinsics) -> None:
        self._width = width
        self._height = height
        self._intr = intr

    @classmethod
    def from_config(
        cls, width: int, height: int, cfg: CameraConfig
    ) -> HandPositionEstimator:
        """Build from the actual frame size and the spec-sheet FOV in config."""
        intr = intrinsics_from_fov(
            width, height, cfg.diagonal_fov_deg, cfg.full_frame_aspect
        )
        return cls(width, height, intr)

    @property
    def intrinsics(self) -> Intrinsics:
        return self._intr

    def estimate(self, hand: RawHandResult) -> WristEstimate:
        return estimate_wrist(
            hand.landmarks, hand.world_landmarks, self._width, self._height, self._intr
        )
