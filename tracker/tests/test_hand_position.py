"""Hand position estimate: synthetic projections with a known answer, plus validation behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_tracker.config import CameraConfig
from hand_tracker.hand_position import (
    HandPositionEstimator,
    Intrinsics,
    estimate_wrist,
    hand_pixel_size,
    intrinsics_from_fov,
    reprojection_rms_px,
    to_pixels,
)
from hand_tracker.landmark_processor import LandmarkProcessor
from hand_tracker.types import LANDMARK_COUNT, FrameResult, Handedness, RawHandResult

_W, _H = 640, 480
_INTR = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
_T_TRUE = np.array([0.07, -0.03, 0.55])


def _hand(seed: int = 0) -> np.ndarray:
    """21 object points within +-8 cm of the origin (not a hand shape; the solve does not care)."""
    return np.random.default_rng(seed).uniform(-0.08, 0.08, size=(LANDMARK_COUNT, 3))


def _project(obj: np.ndarray, t: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Pinhole projection -> normalized image landmarks (x, y, 0)."""
    cam = obj + t
    u = intr.fx * cam[:, 0] / cam[:, 2] + intr.cx
    v = intr.fy * cam[:, 1] / cam[:, 2] + intr.cy
    return np.column_stack([u / _W, v / _H, np.zeros(len(obj))])


class TestIntrinsics:
    @pytest.mark.parametrize(
        ("width", "height", "expected_f"),
        [(1280, 720, 906.8), (640, 480, 604.5), (1920, 1080, 1360.2)],
    )
    def test_c922_modes_share_the_vertical_fov(
        self, width: int, height: int, expected_f: float
    ) -> None:
        intr = intrinsics_from_fov(width, height, 78.0, (16, 9))
        assert intr.fx == intr.fy == pytest.approx(expected_f, abs=0.1)
        assert (intr.cx, intr.cy) == (width / 2, height / 2)

    def test_unsupported_aspect_raises(self) -> None:
        with pytest.raises(ValueError):
            intrinsics_from_fov(1280, 1024, 78.0, (16, 9))

    def test_from_config_uses_the_camera_fields(self) -> None:
        est = HandPositionEstimator.from_config(640, 480, CameraConfig())
        assert est.intrinsics.fx == pytest.approx(604.5, abs=0.1)


class TestSolve:
    def test_noise_free_projection_returns_the_true_translation(self) -> None:
        obj = _hand()
        est = estimate_wrist(_project(obj, _T_TRUE, _INTR), obj, _W, _H, _INTR)
        assert est.position is not None
        np.testing.assert_allclose(
            est.position, (_T_TRUE + obj[0]).astype(np.float32), atol=1e-5
        )
        assert est.depth_ok and est.reprojection_px < 1e-6

    def test_one_pixel_noise_keeps_the_position(self) -> None:
        obj = _hand()
        img = _project(obj, _T_TRUE, _INTR)
        noise = np.random.default_rng(1).normal(
            0.0, 1.0, size=(LANDMARK_COUNT, 2)
        ) / np.array([_W, _H])
        img[:, :2] += noise
        est = estimate_wrist(img, obj, _W, _H, _INTR)
        assert est.position is not None
        assert 0.3 < est.reprojection_px < 3.0
        u, v = to_pixels(img, _W, _H)
        assert est.reprojection_ratio == pytest.approx(
            est.reprojection_px / hand_pixel_size(u, v)
        )

    def test_position_is_empty_when_a_joint_lands_behind_the_camera(self) -> None:
        """World X reversed against the image: here the solve puts joints behind the camera."""
        obj = _hand()
        mirrored = obj * np.array([-1.0, 1.0, 1.0])
        est = estimate_wrist(_project(obj, _T_TRUE, _INTR), mirrored, _W, _H, _INTR)
        assert not est.depth_ok
        assert est.position is None

    def test_to_pixels_scales_by_frame_size(self) -> None:
        u, v = to_pixels(np.array([[0.25, 0.5, 0.0], [1.0, 0.0, 0.0]]), _W, _H)
        np.testing.assert_allclose(u, [160.0, 640.0])
        np.testing.assert_allclose(v, [240.0, 0.0])

    def test_reprojection_is_inf_behind_the_camera(self) -> None:
        obj = _hand()
        u, v = to_pixels(_project(obj, _T_TRUE, _INTR), _W, _H)
        assert math.isinf(
            reprojection_rms_px(obj, np.array([0.0, 0.0, -1.0]), u, v, _INTR)
        )


class TestProcessorIntegration:
    def _raw(self, obj: np.ndarray) -> RawHandResult:
        return RawHandResult(
            landmarks=_project(obj, _T_TRUE, _INTR).astype(np.float32),
            world_landmarks=obj.astype(np.float32),
            handedness=Handedness.LEFT,
            confidence=0.9,
            timestamp_us=1_000,
            inference_time_us=100,
        )

    def test_processed_hand_carries_the_wrist_position(self) -> None:
        obj = _hand()
        processor = LandmarkProcessor(
            window_size=2, estimator=HandPositionEstimator(_W, _H, _INTR)
        )
        frame = FrameResult(
            hands=(self._raw(obj),),
            timestamp_us=1_000,
            frame_index=0,
            inference_time_us=100,
        )
        processed = processor.update(frame)
        assert processed.hands[0].wrist_position is not None
        np.testing.assert_allclose(
            processed.hands[0].wrist_position, _T_TRUE + obj[0], atol=1e-4
        )

    def test_without_an_estimator_the_position_is_absent(self) -> None:
        obj = _hand()
        processor = LandmarkProcessor(window_size=2)
        frame = FrameResult(
            hands=(self._raw(obj),),
            timestamp_us=1_000,
            frame_index=0,
            inference_time_us=100,
        )
        assert processor.update(frame).hands[0].wrist_position is None
