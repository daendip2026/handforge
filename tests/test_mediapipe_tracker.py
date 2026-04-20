"""
Tests for MediaPipeTracker.

MediaPipe is not installed in CI (heavy native dependency).
All tests mock the mediapipe import surface; no actual inference runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast
from unittest.mock import MagicMock, PropertyMock, create_autospec

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

import numpy as np
import pytest

from hand_tracker.config import MediaPipeConfig, TrackerConfig
from hand_tracker.mediapipe_tracker import (
    MediaPipeConfigurationError,
    MediaPipeInferenceError,
    MediaPipeTracker,
    MPCategory,
    MPClassification,
    MPHandsSolution,
    MPLandmark,
    MPLandmarkList,
    MPResults,
)
from hand_tracker.types import (
    LANDMARK_COUNT,
    Frame,
    FrameResult,
    Handedness,
    HandLandmark,
    LandmarkPoint,
    RawHandResult,
)

# ---------------------------------------------------------------------------
# Test Constants
# ---------------------------------------------------------------------------

TEST_LANDMARK_X: Final[float] = 0.5
TEST_LANDMARK_Y: Final[float] = 0.5
TEST_LANDMARK_Z: Final[float] = 0.1
TEST_WORLD_X: Final[float] = 0.123
TEST_WORLD_Y: Final[float] = -0.456
TEST_WORLD_Z: Final[float] = 0.789
TEST_CONFIDENCE: Final[float] = 0.95
TEST_TIMESTAMP_US: Final[int] = 1_700_000_000_000_000
TEST_FRAME_INDEX: Final[int] = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_hands_instance() -> MagicMock:
    """Provides a spec-accurate mock for the MediaPipe Hands instance using Protocols."""
    # We use MPHandsSolution Protocol as the spec. This "brings the contract"
    # from the production code without requiring the actual library.
    return cast(MagicMock, create_autospec(MPHandsSolution, instance=True))


@pytest.fixture
def mock_hands_module(mock_hands_instance: MagicMock) -> MagicMock:
    """Fixture providing a mock MediaPipe hands module (for DI)."""
    from hand_tracker.mediapipe_tracker import MPHandsModule

    mock_module = cast(MagicMock, create_autospec(MPHandsModule, instance=True))
    mock_module.Hands.return_value = mock_hands_instance
    return mock_module


@pytest.fixture
def mp_cfg() -> MediaPipeConfig:
    """Fixture providing a default valid MediaPipeConfig."""
    return MediaPipeConfig(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        warmup_frame_count=5,
    )


@pytest.fixture
def tracker_cfg() -> TrackerConfig:
    """Fixture providing a default valid TrackerConfig."""
    return TrackerConfig(primary_hand=Handedness.RIGHT)


# ---------------------------------------------------------------------------
# Factories for Mock Data (Protocol-compliant)
# ---------------------------------------------------------------------------


def _make_frame(frame_index: int = 0, is_mirrored: bool = True) -> Frame:
    return Frame(
        bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp_us=TEST_TIMESTAMP_US,
        frame_index=frame_index,
        is_mirrored=is_mirrored,
    )


def _make_landmark(
    x: float = TEST_LANDMARK_X, y: float = TEST_LANDMARK_Y, z: float = TEST_LANDMARK_Z
) -> MagicMock:
    """Create a mock landmark compliant with MPLandmark protocol."""
    landmark = cast(MagicMock, create_autospec(MPLandmark, instance=True))
    landmark.x = x
    landmark.y = y
    landmark.z = z
    return landmark


def _make_landmark_list(count: int = LANDMARK_COUNT) -> MagicMock:
    """Create a mock landmark list compliant with MPLandmarkList protocol."""
    lm_list = cast(MagicMock, create_autospec(MPLandmarkList, instance=True))
    lm_list.landmark = [
        _make_landmark(
            TEST_LANDMARK_X + i * 0.01, TEST_LANDMARK_Y + i * 0.01, TEST_LANDMARK_Z
        )
        for i in range(count)
    ]
    return lm_list


def _make_handedness_result(label: str = "Right", score: float = 0.95) -> MagicMock:
    """Create a mock handedness result compliant with MPCategory protocol."""
    classification = cast(MagicMock, create_autospec(MPClassification, instance=True))
    classification.label = label
    classification.score = score

    handedness = cast(MagicMock, create_autospec(MPCategory, instance=True))
    handedness.classification = [classification]
    return handedness


def _make_mp_results(
    detections: list[tuple[str, float]] | None = None,
    missing_world: bool = False,
    missing_handedness: bool = False,
) -> MagicMock:
    """
    Build a MediaPipe results object compliant with MPResults protocol.
    """
    results = cast(MagicMock, create_autospec(MPResults, instance=True))

    if detections is None:
        type(results).multi_hand_landmarks = PropertyMock(return_value=None)
        type(results).multi_hand_world_landmarks = PropertyMock(return_value=None)
        type(results).multi_handedness = PropertyMock(return_value=None)
        return results

    type(results).multi_hand_landmarks = PropertyMock(
        return_value=[_make_landmark_list() for _ in detections]
    )
    type(results).multi_hand_world_landmarks = PropertyMock(
        return_value=None
        if missing_world
        else [_make_landmark_list() for _ in detections]
    )
    type(results).multi_handedness = PropertyMock(
        return_value=None
        if missing_handedness
        else [_make_handedness_result(label, score) for label, score in detections]
    )

    return results


# ---------------------------------------------------------------------------
# Tests: Tracker Lifecycle
# ---------------------------------------------------------------------------


class TestMediaPipeTrackerLifecycle:
    def test_context_manager_closes_hands(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        mp_cfg: MediaPipeConfig,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify that the MediaPipe Hands solution is closed on exit."""
        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module):
            pass

        mock_hands_instance.close.assert_called_once()

    def test_hands_is_none_after_exit(
        self,
        mock_hands_module: MagicMock,
        mp_cfg: MediaPipeConfig,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify that the internal hands reference is cleared after exit."""
        tracker = MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module)
        with tracker:
            pass
        assert tracker._hands is None

    def test_raises_without_context_manager(
        self, mp_cfg: MediaPipeConfig, tracker_cfg: TrackerConfig
    ) -> None:
        """Verify that process() raises an error if called outside a context manager."""
        tracker = MediaPipeTracker(mp_cfg, tracker_cfg)
        with pytest.raises(MediaPipeConfigurationError, match="context manager"):
            tracker.process(_make_frame())


class TestWarmup:
    def test_warmup_frames_return_empty_hands(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify that results are empty during the warmup period."""
        warmup_count = 3
        mp_cfg = MediaPipeConfig(warmup_frame_count=warmup_count)
        mock_hands_instance.process.return_value = _make_mp_results(
            [("Right", TEST_CONFIDENCE)]
        )

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            for i in range(warmup_count):
                result = tracker.process(_make_frame(i))
                assert isinstance(result, FrameResult)
                assert len(result.hands) == 0

    def test_post_warmup_frame_is_processed(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify that the tracker starts processing after warmup completes."""
        warmup_count = 2
        mp_cfg = MediaPipeConfig(warmup_frame_count=warmup_count)
        mock_hands_instance.process.return_value = _make_mp_results(
            [("Right", TEST_CONFIDENCE)]
        )

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            # Burn warmup frames
            for i in range(warmup_count):
                tracker.process(_make_frame(i))

            # This frame should be processed
            result = tracker.process(_make_frame(warmup_count))

        assert len(result.hands) == 1


class TestProcessDetection:
    @pytest.mark.parametrize(
        "detections, expected_count",
        [
            (None, 0),  # No detection
            ([], 0),  # Empty result from MP (shouldn't happen with landmarks but safe)
            ([("Right", 0.9)], 1),  # Single match
            ([("Left", 0.9)], 0),  # Filtered out
            ([("Right", 0.95), ("Left", 0.92)], 1),  # Filtered from multiple
        ],
    )
    def test_detection_filtering(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        detections: list[tuple[str, float]] | None,
        expected_count: int,
    ) -> None:
        """Verify filtering logic across diverse detection scenarios."""
        mock_hands_instance.process.return_value = _make_mp_results(detections)
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)
        tracker_cfg = TrackerConfig(primary_hand=Handedness.RIGHT)

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            result = tracker.process(_make_frame())

        assert len(result.hands) == expected_count

    def test_returns_both_hands_when_configured(
        self, mock_hands_module: MagicMock, mock_hands_instance: MagicMock
    ) -> None:
        """Verify that multiple hands are returned when primary_hand is BOTH."""
        mock_hands_instance.process.return_value = _make_mp_results(
            [("Right", 0.95), ("Left", 0.92)]
        )

        mp_cfg = MediaPipeConfig(max_num_hands=2, warmup_frame_count=0)
        tracker_cfg = TrackerConfig(primary_hand=Handedness.BOTH)
        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            result = tracker.process(_make_frame())

        assert len(result.hands) == 2
        sides = {h.handedness for h in result.hands}
        assert sides == {Handedness.RIGHT, Handedness.LEFT}

    def test_landmark_data_integrity(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify that landmark data is correctly mapped from MediaPipe to typed objects."""
        mock_hands_instance.process.return_value = _make_mp_results(
            [("Right", TEST_CONFIDENCE)]
        )
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)
        frame = _make_frame(frame_index=TEST_FRAME_INDEX)

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            result = tracker.process(frame)

        # Verify type hierarchies and deep data
        hand = result.hands[0]
        assert isinstance(hand, RawHandResult)
        assert hand.handedness == Handedness.RIGHT
        assert hand.confidence == pytest.approx(TEST_CONFIDENCE)

        wrist = hand.landmarks[HandLandmark.WRIST]
        assert isinstance(wrist, LandmarkPoint)
        assert wrist.x == pytest.approx(TEST_LANDMARK_X)
        assert wrist.wx == pytest.approx(TEST_LANDMARK_X)

    def test_raises_on_missing_world_landmarks(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify that MediaPipeInferenceError is raised if world landmarks are missing."""
        mock_hands_instance.process.return_value = _make_mp_results(
            [("Right", 0.9)], missing_world=True
        )
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with (
            MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker,
            pytest.raises(
                MediaPipeInferenceError,
                match=r"MediaPipe returned None for landmarks\.",
            ),
        ):
            tracker.process(_make_frame())


class TestMediaPipeTrackerEdgeCases:
    def test_process_empty_frame(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify that empty frames return an empty result and log a warning."""
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)
        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            # Test None bgr
            result_none = tracker.process(
                Frame(bgr=None, timestamp_us=0, frame_index=0, is_mirrored=True)  # type: ignore[arg-type]
            )
            assert len(result_none.hands) == 0

            # Test empty bgr array
            result_empty = tracker.process(
                Frame(bgr=np.array([]), timestamp_us=0, frame_index=0, is_mirrored=True)
            )
            assert len(result_empty.hands) == 0

    def test_process_malformed_handedness(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify that MediaPipeInferenceError is raised if handedness data is missing."""
        mock_hands_instance.process.return_value = _make_mp_results(
            [("Right", 0.9)], missing_handedness=True
        )
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with (
            MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker,
            pytest.raises(MediaPipeInferenceError, match="Handedness data missing"),
        ):
            tracker.process(_make_frame())

    def test_malformed_landmark_count(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify that MediaPipeInferenceError is raised if landmark count is not 21."""
        # Create a mock with only 5 landmarks
        mock_results = _make_mp_results([("Right", 0.9)])
        type(mock_results).multi_hand_landmarks = PropertyMock(
            return_value=[_make_landmark_list(count=5)]
        )
        type(mock_results).multi_hand_world_landmarks = PropertyMock(
            return_value=[_make_landmark_list(count=5)]
        )

        mock_hands_instance.process.return_value = mock_results
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with (
            MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker,
            pytest.raises(MediaPipeInferenceError, match="Expected 21 landmarks"),
        ):
            tracker.process(_make_frame())

    def test_inconsistent_side_lengths(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify error handling when image and world landmark lists have different lengths."""
        mock_results = _make_mp_results([("Right", 0.9)])
        # Hand 0 landmarks: 21 for image, but 20 for world
        type(mock_results).multi_hand_landmarks = PropertyMock(
            return_value=[_make_landmark_list(count=21)]
        )
        type(mock_results).multi_hand_world_landmarks = PropertyMock(
            return_value=[_make_landmark_list(count=20)]
        )

        mock_hands_instance.process.return_value = mock_results
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with (
            MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker,
            pytest.raises(
                MediaPipeInferenceError, match="Inconsistent landmark counts"
            ),
        ):
            tracker.process(_make_frame())

    def test_landmark_boundary_conditions(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Verify landmark mapping at spatial boundaries (0.0, 1.0)."""

        def _make_boundary_landmarks() -> MagicMock:
            lm_list = cast(MagicMock, create_autospec(MPLandmarkList, instance=True))
            lm_list.landmark = [
                _make_landmark(0.0, 0.0, 0.0),  # Corner 1
                _make_landmark(1.0, 1.0, 1.0),  # Corner 2
            ] + [_make_landmark(0.5, 0.5, 0.5)] * (LANDMARK_COUNT - 2)
            return lm_list

        mock_results = create_autospec(MPResults, instance=True)
        type(mock_results).multi_hand_landmarks = PropertyMock(
            return_value=[_make_boundary_landmarks()]
        )
        type(mock_results).multi_hand_world_landmarks = PropertyMock(
            return_value=[_make_boundary_landmarks()]
        )
        type(mock_results).multi_handedness = PropertyMock(
            return_value=[_make_handedness_result("Right", 0.9)]
        )

        mock_hands_instance.process.return_value = mock_results

        mp_cfg = MediaPipeConfig(warmup_frame_count=0)
        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            result = tracker.process(_make_frame())

        hand = result.hands[0]
        assert hand.landmarks[0].x == 0.0
        assert hand.landmarks[1].x == 1.0

    def test_configuration_passed_to_mediapipe(
        self, mock_hands_module: MagicMock
    ) -> None:
        """Verify that the MediaPipe solution is initialized with the correct config."""
        mp_cfg = MediaPipeConfig(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6,
        )
        tracker_cfg = TrackerConfig(primary_hand=Handedness.RIGHT)

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module):
            pass

        mock_hands_module.Hands.assert_called_once_with(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6,
        )


class TestMediaPipeTrackerThreadSafety:
    """Validation of thread-safety contracts and data immutability."""

    def test_result_immutability(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """
        Verify that produced results are immutable (thread-safe for sharing).

        Engineering Risk: In multi-threaded environments (e.g., Unity integration),
        if the results are mutable, downstream modification in a separate thread
        could cause race conditions or corrupt the tracking state.
        """
        mock_hands_instance.process.return_value = _make_mp_results([("Right", 0.9)])
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            result = tracker.process(_make_frame())

        # Ensure hands collection is a tuple (immutable)
        assert isinstance(result.hands, tuple)
        # Verify frozen dataclass
        with pytest.raises(AttributeError):
            result.hands[0].handedness = Handedness.LEFT  # type: ignore[misc]


class TestMediaPipeTrackerResourceSafety:
    """Validation of memory protection and hardware resource lifecycle."""

    def test_zero_copy_protection(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """
        Verify that the input image buffer is protected via the writeable flag.

        Engineering Risk: MediaPipe Python uses zero-copy image sharing.
        If the 'writeable' flag is not toggled to False, accidental mutation
        during inference could crash the ML model or lead to unpredictable
        memory corruption. This test also prevents performance regression
        (MediaPipe defaults to deep copy if writeable=True).
        """
        mock_hands_instance.process.return_value = _make_mp_results([("Right", 0.9)])
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        frame = _make_frame()
        # Initially writeable
        assert frame.bgr.flags.writeable is True

        # Intercept process() to check flag state during inference
        def check_flag(image: np.ndarray) -> MagicMock:
            assert image.flags.writeable is False
            return _make_mp_results([("Right", 0.9)])

        mock_hands_instance.process.side_effect = check_flag

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            tracker.process(frame)

        # Flag should be restored
        assert frame.bgr.flags.writeable is True

    def test_strict_lifecycle_enforcement(
        self,
        mock_hands_module: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """
        Ensure process() raises error if called after the tracker is closed.

        Engineering Risk: Calling processing methods on a released native resource
        typically leads to Segmentation Faults. This guard ensures a clean Python
        exception is raised instead of a hard process crash.
        """
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        tracker = MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module)
        with tracker:
            pass  # Immediately exit

        with pytest.raises(MediaPipeConfigurationError, match="context manager"):
            tracker.process(_make_frame())


class TestMediaPipeTrackerInferenceQuality:
    """Validation of system metrics, observability, and data precision."""

    def test_inference_timing_propagation(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """
        Verify that inference timing is recorded and propagated correctly.

        Engineering Risk: In real-time systems, observability of inference latency
        is critical for debugging pipeline bottlenecks. Missing or incorrect
        metrics make it impossible to diagnose performance drops in production.
        """
        mock_hands_instance.process.return_value = _make_mp_results([("Right", 0.9)])
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            result = tracker.process(_make_frame())

        assert result.inference_time_us >= 0
        assert result.hands[0].inference_time_us == result.inference_time_us

    def test_metric_world_coordinate_mapping(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """
        Verify Unity-critical metrical world coordinates (wx, wy, wz).

        Engineering Risk: Unity VRM/IK solvers require physical meter coordinates
        relative to the wrist. If the mapping from multi_hand_world_landmarks
        is incorrect, the 3D hand poses will be distorted or scaled incorrectly.
        """
        mock_lm = cast(MagicMock, create_autospec(MPLandmark, instance=True))
        mock_lm.x, mock_lm.y, mock_lm.z = TEST_WORLD_X, TEST_WORLD_Y, TEST_WORLD_Z

        mock_lm_list = cast(MagicMock, create_autospec(MPLandmarkList, instance=True))
        mock_lm_list.landmark = [mock_lm] * LANDMARK_COUNT

        mock_results = _make_mp_results([("Right", 0.9)])
        type(mock_results).multi_hand_world_landmarks = PropertyMock(
            return_value=[mock_lm_list]
        )

        mock_hands_instance.process.return_value = mock_results
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            result = tracker.process(_make_frame())

        hand = result.hands[0]
        # Spot check index 0
        assert hand.landmarks[0].wx == TEST_WORLD_X
        assert hand.landmarks[0].wy == TEST_WORLD_Y
        assert hand.landmarks[0].wz == TEST_WORLD_Z


class TestMetadataPropagation:
    """Validation of metadata persistence across the inference pipeline."""

    def test_is_mirrored_flag_propagated(
        self,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """
        Verify that the is_mirrored flag is correctly passed from Frame to FrameResult.

        Engineering Risk: Coordinate mirroring is a critical metadata bit for Unity.
        If this flag is lost or inverted in the pipeline, the 3D hands in VR/AR will
        move in the opposite direction of the user's actual hands, causing severe
        motion sickness or breaking interaction logic.
        """
        mock_hands_instance.process.return_value = _make_mp_results([("Right", 0.9)])
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:
            # 1. Test mirroring active
            res_true = tracker.process(_make_frame(is_mirrored=True))
            assert res_true.is_mirrored is True

            # 2. Test mirroring inactive
            res_false = tracker.process(_make_frame(is_mirrored=False))
            assert res_false.is_mirrored is False


class TestMediaPipeTrackerPerformance:
    """Benchmark tests for critical inference and transformation paths."""

    @pytest.mark.benchmark(group="tracker")
    def test_benchmark_coordinate_transformation(
        self,
        benchmark: BenchmarkFixture,
        mock_hands_module: MagicMock,
        mock_hands_instance: MagicMock,
        tracker_cfg: TrackerConfig,
    ) -> None:
        """Benchmark the mapping from MediaPipe results to typed FrameResult."""
        mock_hands_instance.process.return_value = _make_mp_results([("Right", 0.9)])
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)
        frame = _make_frame()

        with MediaPipeTracker(mp_cfg, tracker_cfg, mock_hands_module) as tracker:

            def _run_process() -> None:
                tracker.process(frame)

            # Benchmark the process method which includes conversion logic
            benchmark(_run_process)
