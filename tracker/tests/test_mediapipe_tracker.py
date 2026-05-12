"""
Tests for MediaPipeTracker.

MediaPipe is not installed in CI (heavy native dependency).
All tests mock the mediapipe import surface; no actual inference runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast
from unittest.mock import MagicMock, create_autospec

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

import numpy as np
import pytest

from hand_tracker.config import CameraConfig, MediaPipeConfig, TrackerConfig
from hand_tracker.mediapipe_tracker import (
    MediaPipeConfigurationError,
    MediaPipeInferenceError,
    MediaPipeTracker,
    MPCategory,
    MPHandLandmarker,
    MPHandLandmarkerResult,
    MPLandmark,
)
from hand_tracker.types import (
    LANDMARK_COUNT,
    Frame,
    FrameResult,
    Handedness,
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
# Test Helpers
# ---------------------------------------------------------------------------


def _create_mock_model_file(path: str) -> None:
    """Ensure a dummy model file exists for tests that check its existence."""
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_detector_instance() -> MagicMock:
    """Provides a spec-accurate mock for the MediaPipe HandLandmarker instance."""
    return cast(MagicMock, create_autospec(MPHandLandmarker, instance=True))


@pytest.fixture
def mock_landmarker_factory(mock_detector_instance: MagicMock) -> MagicMock:
    """
    Fixture providing a mock HandLandmarker factory (for DI).
    In tests, we immediately trigger the callback with mocked results.
    """
    mock_factory = MagicMock()

    def _create_from_options(options: Any) -> MagicMock:
        # Capture the callback
        on_result = options.result_callback

        def _detect_async(image: Any, timestamp_ms: int) -> None:
            # In tests, we immediately call the callback with mocked results
            # configured in the test case via _test_result.
            result = getattr(mock_detector_instance, "_test_result", _make_mp_results())
            on_result(result, image, timestamp_ms)

        mock_detector_instance.detect_async.side_effect = _detect_async
        return mock_detector_instance

    mock_factory.create_from_options.side_effect = _create_from_options
    return mock_factory


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


@pytest.fixture
def camera_cfg() -> CameraConfig:
    """Fixture providing a default valid CameraConfig (non-mirrored for pure logic tests)."""
    return CameraConfig(mirror_input=False)


@pytest.fixture
def create_tracker(
    mp_cfg: MediaPipeConfig,
    tracker_cfg: TrackerConfig,
    camera_cfg: CameraConfig,
    mock_landmarker_factory: MagicMock,
) -> Any:
    """Factory fixture to create MediaPipeTracker instances with less boilerplate."""

    def _create(
        custom_mp_cfg: MediaPipeConfig | None = None,
        custom_tracker_cfg: TrackerConfig | None = None,
    ) -> MediaPipeTracker:
        return MediaPipeTracker(
            mp_cfg=custom_mp_cfg or mp_cfg,
            tracker_cfg=custom_tracker_cfg or tracker_cfg,
            camera_cfg=camera_cfg,
            hand_landmarker_factory=mock_landmarker_factory,
        )

    return _create


@pytest.fixture(autouse=True)
def auto_mock_model(mp_cfg: MediaPipeConfig) -> None:
    """Ensure the mock model file exists for all tests using MediaPipeConfig."""
    _create_mock_model_file(mp_cfg.model_path)


# ---------------------------------------------------------------------------
# Factories for Mock Data (Protocol-compliant)
# ---------------------------------------------------------------------------


def _make_frame(
    frame_index: int = 0,
    timestamp_us: int = TEST_TIMESTAMP_US,
) -> Frame:
    return Frame(
        bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp_us=timestamp_us,
        frame_index=frame_index,
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


def _make_handedness_result(
    category_name: str = "Right", score: float = 0.95
) -> MagicMock:
    """Create a mock handedness category compliant with MPCategory protocol."""
    category = cast(MagicMock, create_autospec(MPCategory, instance=True))
    category.category_name = category_name
    category.score = score
    category.index = 0
    category.display_name = category_name
    return category


def _make_mp_results(
    detections: list[tuple[str, float]] | None = None,
) -> MagicMock:
    """
    Build a MediaPipe results object compliant with MPHandLandmarkerResult protocol.
    """
    results = cast(MagicMock, create_autospec(MPHandLandmarkerResult, instance=True))

    if detections is None:
        results.hand_landmarks = []
        results.hand_world_landmarks = []
        results.handedness = []
        return results

    results.hand_landmarks = [
        [_make_landmark() for _ in range(LANDMARK_COUNT)] for _ in detections
    ]
    results.hand_world_landmarks = [
        [_make_landmark() for _ in range(LANDMARK_COUNT)] for _ in detections
    ]
    results.handedness = [
        [_make_handedness_result(label, score)] for label, score in detections
    ]

    return results


# ---------------------------------------------------------------------------
# Tests: Tracker Lifecycle
# ---------------------------------------------------------------------------


class TestMediaPipeTrackerLifecycle:
    def test_context_manager_closes_hands(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """Verify that the MediaPipe HandLandmarker is closed on exit."""
        with create_tracker():
            pass

        mock_detector_instance.close.assert_called_once()

    def test_raises_without_context_manager(self, create_tracker: Any) -> None:
        """Verify that process() raises an error if called outside a context manager."""
        tracker = create_tracker()
        with pytest.raises(MediaPipeConfigurationError, match="context manager"):
            tracker.process(_make_frame())


class TestWarmup:
    def test_warmup_frames_return_empty_hands(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """Verify that results are empty during the warmup period."""
        warmup_count = 3
        mp_cfg = MediaPipeConfig(warmup_frame_count=warmup_count)
        mock_detector_instance._test_result = _make_mp_results(
            [("Right", TEST_CONFIDENCE)]
        )

        with create_tracker(custom_mp_cfg=mp_cfg) as tracker:
            for i in range(warmup_count):
                result = tracker.process(_make_frame(i))
                assert isinstance(result, FrameResult)
                assert len(result.hands) == 0

    def test_post_warmup_frame_is_processed(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """Verify that the tracker starts processing after warmup completes."""
        warmup_count = 2
        mp_cfg = MediaPipeConfig(warmup_frame_count=warmup_count)
        mock_detector_instance._test_result = _make_mp_results(
            [("Right", TEST_CONFIDENCE)]
        )

        with create_tracker(custom_mp_cfg=mp_cfg) as tracker:
            for i in range(warmup_count):
                tracker.process(_make_frame(i))

            # This frame should be processed
            result = tracker.process(_make_frame(warmup_count))

        assert len(result.hands) == 1


class TestProcessDetection:
    @pytest.mark.parametrize(
        "primary_hand, max_hands, detections, expected_count",
        [
            (Handedness.RIGHT, 1, None, 0),  # No detection
            (Handedness.RIGHT, 1, [], 0),  # Empty detection list
            (Handedness.RIGHT, 1, [("Right", 0.9)], 1),  # Exact match (Right)
            (Handedness.RIGHT, 1, [("Left", 0.9)], 0),  # Mismatch (Left filtered out)
            (
                Handedness.RIGHT,
                2,
                [("Right", 0.95), ("Left", 0.92)],
                1,
            ),  # Multi-hand but single target
            (
                Handedness.BOTH,
                2,
                [("Right", 0.95), ("Left", 0.92)],
                2,
            ),  # BOTH mode returns everything
            (Handedness.BOTH, 1, [("Right", 0.85)], 1),  # BOTH mode with single hand
        ],
    )
    def test_detection_logic(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
        primary_hand: Handedness,
        max_hands: int,
        detections: list[tuple[str, float]] | None,
        expected_count: int,
    ) -> None:
        """Verify complex detection and filtering logic in a single unified test."""
        # Setup
        mock_detector_instance._test_result = _make_mp_results(detections)
        mp_cfg = MediaPipeConfig(max_num_hands=max_hands, warmup_frame_count=0)
        tracker_cfg = TrackerConfig(primary_hand=primary_hand)

        with create_tracker(
            custom_mp_cfg=mp_cfg, custom_tracker_cfg=tracker_cfg
        ) as tracker:
            result = tracker.process(_make_frame())

        # Assert
        assert isinstance(result, FrameResult), "Must return FrameResult even if empty"
        assert len(result.hands) == expected_count


class TestMediaPipeTrackerEdgeCases:
    def test_process_malformed_handedness(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """Verify that unknown handedness strings from MediaPipe fallback to BOTH."""
        # Setup with an unexpected category name
        mock_detector_instance._test_result = _make_mp_results([("Alien_Hand", 0.9)])
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)
        tracker_cfg = TrackerConfig(primary_hand=Handedness.BOTH)

        with create_tracker(
            custom_mp_cfg=mp_cfg, custom_tracker_cfg=tracker_cfg
        ) as tracker:
            result = tracker.process(_make_frame())

        # Verify fallback logic
        assert result.hands[0].handedness == Handedness.BOTH

    def test_process_empty_frame(
        self,
        create_tracker: Any,
    ) -> None:
        """Verify that empty frames return an empty result."""
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)
        with create_tracker(custom_mp_cfg=mp_cfg) as tracker:
            result_none = tracker.process(
                Frame(bgr=cast(Any, None), timestamp_us=0, frame_index=0)
            )
            assert len(result_none.hands) == 0

    def test_malformed_landmark_count(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """Verify that MediaPipeInferenceError is raised if landmark count is not 21."""
        # Create a mock with only 5 landmarks
        mock_results = _make_mp_results([("Right", 0.9)])
        mock_results.hand_landmarks = [[_make_landmark() for _ in range(5)]]
        mock_results.hand_world_landmarks = [[_make_landmark() for _ in range(5)]]
        mock_detector_instance._test_result = mock_results

        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with (
            create_tracker(custom_mp_cfg=mp_cfg) as tracker,
            pytest.raises(MediaPipeInferenceError, match="Expected 21 landmarks"),
        ):
            tracker.process(_make_frame())

    def test_raises_on_inconsistent_result_lists(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """Verify that MediaPipeInferenceError is raised if result list lengths mismatch."""
        mock_results = _make_mp_results([("Right", 0.9)])
        # Simulate inconsistency: 1 landmark list but 0 world landmark lists
        mock_results.hand_world_landmarks = []
        mock_detector_instance._test_result = mock_results

        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with (
            create_tracker(custom_mp_cfg=mp_cfg) as tracker,
            pytest.raises(
                MediaPipeInferenceError, match="Inconsistent result list lengths"
            ),
        ):
            tracker.process(_make_frame())

    def test_landmark_boundary_conditions(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """Verify landmark mapping at spatial boundaries (0.0, 1.0)."""
        mock_results = _make_mp_results([("Right", 0.9)])
        boundary_lms = [
            _make_landmark(0.0, 0.0, 0.0),
            _make_landmark(1.0, 1.0, 1.0),
        ] + [_make_landmark(0.5, 0.5, 0.5)] * (LANDMARK_COUNT - 2)

        mock_results.hand_landmarks = [boundary_lms]
        mock_results.hand_world_landmarks = [boundary_lms]
        mock_detector_instance._test_result = mock_results

        mp_cfg = MediaPipeConfig(warmup_frame_count=0)
        with create_tracker(custom_mp_cfg=mp_cfg) as tracker:
            result = tracker.process(_make_frame())

        hand = result.hands[0]
        # NumPy indexing: [landmark_index, component_index] where 0=x
        assert hand.landmarks[0, 0] == 0.0
        assert hand.landmarks[1, 0] == 1.0


class TestMediaPipeTrackerThreadSafety:
    """Validation of thread-safety contracts and data immutability."""

    def test_result_immutability(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """
        Verify that produced results are immutable (thread-safe for sharing).

        Engineering Risk: In multi-threaded environments (e.g., Unity integration),
        if the results are mutable, downstream modification in a separate thread
        could cause race conditions or corrupt the tracking state.
        """
        mock_detector_instance._test_result = _make_mp_results([("Right", 0.9)])
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with create_tracker(custom_mp_cfg=mp_cfg) as tracker:
            result = tracker.process(_make_frame())

        # Ensure hands collection is a tuple (immutable)
        assert isinstance(result.hands, tuple)
        # Verify frozen dataclass
        with pytest.raises(AttributeError):
            result.hands[0].handedness = Handedness.LEFT  # type: ignore


class TestMediaPipeTrackerResourceSafety:
    """Validation of memory protection and hardware resource lifecycle."""

    def test_strict_lifecycle_enforcement(
        self,
        create_tracker: Any,
    ) -> None:
        """Verify that the tracker strictly enforces its lifecycle (Before -> During -> After)."""
        tracker = create_tracker()
        frame = _make_frame()

        # 1. State: PRE-INIT (Must raise error)
        with pytest.raises(MediaPipeConfigurationError, match="context manager"):
            tracker.process(frame)

        # 2. State: ACTIVE (Must work correctly)
        with tracker:
            result = tracker.process(frame)
            assert isinstance(result, FrameResult)

        # 3. State: POST-CLOSE (Must raise error again)
        with pytest.raises(MediaPipeConfigurationError, match="context manager"):
            tracker.process(frame)

    def test_double_close_is_safe(
        self,
        create_tracker: Any,
    ) -> None:
        """Verify that closing the tracker multiple times does not raise errors (Idempotency)."""
        tracker = create_tracker()
        with tracker:
            pass  # First close via __exit__

        # Second manual or implicit close should be safe
        tracker.__exit__(None, None, None)

    def test_input_buffer_memory_protection(
        self,
        create_tracker: Any,
    ) -> None:
        """
        Verify that the input image buffer is protected via the writeable flag.

        Engineering Risk: MediaPipe Python uses zero-copy image sharing.
        If the 'writeable' flag is not toggled to False, accidental mutation
        during inference could crash the ML model or lead to unpredictable
        memory corruption. This test also prevents performance regression
        (MediaPipe defaults to deep copy if writeable=True).
        """
        from unittest.mock import patch

        # We need to spy on the mp.Image constructor to check the data flags
        with (
            patch("mediapipe.Image") as mock_image_class,
            create_tracker(
                custom_mp_cfg=MediaPipeConfig(warmup_frame_count=0)
            ) as tracker,
        ):
            tracker.process(_make_frame())

            # Check the first argument (data) of the mp.Image constructor call
            assert mock_image_class.called
            args, kwargs = mock_image_class.call_args

            # In mp_image = mp.Image(image_format=..., data=dst_rgb)
            # data might be passed as a keyword or positional argument
            passed_data = kwargs.get("data")
            if passed_data is None and args:
                passed_data = args[0]  # Fallback to positional if not found in kwargs

            assert passed_data is not None
            assert passed_data.flags.writeable is False, (
                "Buffer MUST be read-only when passed to MediaPipe for zero-copy safety"
            )


class TestMediaPipeTrackerInferenceQuality:
    """Validation of system metrics, observability, and data precision."""

    def test_inference_timing_propagation(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """
        Verify that inference timing is recorded and propagated correctly.

        Engineering Risk: In real-time systems, observability of inference latency
        is critical for debugging pipeline bottlenecks. Missing or incorrect
        metrics make it impossible to diagnose performance drops in production.
        """
        mock_detector_instance._test_result = _make_mp_results([("Right", 0.9)])
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with create_tracker(custom_mp_cfg=mp_cfg) as tracker:
            result = tracker.process(_make_frame())

        assert result.hands[0].inference_time_us >= 0

    def test_landmark_data_integrity(
        self, mock_detector_instance: MagicMock, create_tracker: Any
    ) -> None:
        """Verify that landmarks and world landmarks are mapped correctly and accurately."""
        # Setup specific known landmark values
        mock_lm = cast(MagicMock, create_autospec(MPLandmark, instance=True))
        mock_lm.x, mock_lm.y, mock_lm.z = 0.1, 0.2, 0.3

        mock_wlm = cast(MagicMock, create_autospec(MPLandmark, instance=True))
        mock_wlm.x, mock_wlm.y, mock_wlm.z = 0.01, 0.02, 0.03

        mock_results = _make_mp_results([("Right", 0.95)])
        mock_results.hand_landmarks = [[mock_lm] * LANDMARK_COUNT]
        mock_results.hand_world_landmarks = [[mock_wlm] * LANDMARK_COUNT]
        mock_detector_instance._test_result = mock_results

        mp_cfg = MediaPipeConfig(warmup_frame_count=0)
        with create_tracker(custom_mp_cfg=mp_cfg) as tracker:
            result = tracker.process(_make_frame())

        hand = result.hands[0]
        assert hand.landmarks.shape == (LANDMARK_COUNT, 3)
        assert hand.landmarks[0, 0] == 0.1
        assert hand.landmarks[0, 1] == 0.2
        assert hand.landmarks[0, 2] == 0.3
        assert hand.world_landmarks[0, 0] == 0.01
        assert hand.world_landmarks[0, 1] == 0.02
        assert hand.world_landmarks[0, 2] == 0.03

        assert hand.confidence == 0.95
        assert hand.handedness == Handedness.RIGHT

    def test_metric_world_coordinate_mapping(
        self,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """
        Verify Unity-critical metrical world coordinates (wx, wy, wz).

        Engineering Risk: Unity VRM/IK solvers require physical meter coordinates
        relative to the wrist. If the mapping from multi_hand_world_landmarks
        is incorrect, the 3D hand poses will be distorted or scaled incorrectly.
        """
        mock_lm = cast(MagicMock, create_autospec(MPLandmark, instance=True))
        mock_lm.x, mock_lm.y, mock_lm.z = TEST_WORLD_X, TEST_WORLD_Y, TEST_WORLD_Z

        mock_results = _make_mp_results([("Right", 0.9)])
        mock_results.hand_world_landmarks = [[mock_lm] * LANDMARK_COUNT]
        mock_detector_instance._test_result = mock_results

        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        with create_tracker(custom_mp_cfg=mp_cfg) as tracker:
            result = tracker.process(_make_frame())

        hand = result.hands[0]
        # Spot check index 0
        assert hand.world_landmarks[0, 0] == TEST_WORLD_X
        assert hand.world_landmarks[0, 1] == TEST_WORLD_Y
        assert hand.world_landmarks[0, 2] == TEST_WORLD_Z


class TestMediaPipeTrackerPerformance:
    """Benchmark tests for critical inference and transformation paths."""

    @pytest.mark.benchmark(group="tracker-hot-path")
    def test_benchmark_processing_hot_path(
        self,
        benchmark: BenchmarkFixture,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """Benchmark the full mapping path with NEW frames (Zero-Redundancy bypass)."""
        mock_detector_instance._test_result = _make_mp_results([("Right", 0.9)])
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)

        # Prevent completely destroying the benchmark with GC pressure by pre-allocating
        # the large 1MB numpy array. In the real world, frames arrive at ~60fps,
        # but the benchmark loop runs at thousands of iterations per second.
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        def frame_iterator() -> Iterator[Frame]:
            """Yields frames with monotonically increasing timestamps."""
            ts = TEST_TIMESTAMP_US
            while True:
                yield Frame(bgr=dummy_bgr, timestamp_us=ts, frame_index=0)
                ts += 1000

        gen = frame_iterator()

        with create_tracker(custom_mp_cfg=mp_cfg) as tracker:
            # Using balanced pedantic mode (iterations=40) to filter jitter
            # while ensuring the test suite remains fast and responsive.
            benchmark.pedantic(
                lambda: tracker.process(next(gen)),
                iterations=40,
                rounds=30,
                warmup_rounds=10,
            )

    @pytest.mark.benchmark(group="tracker-polling")
    def test_benchmark_polling_efficiency(
        self,
        benchmark: BenchmarkFixture,
        mock_detector_instance: MagicMock,
        create_tracker: Any,
    ) -> None:
        """Benchmark polling speed when the same frame is received (Zero-Redundancy efficiency)."""
        mock_detector_instance._test_result = _make_mp_results([("Right", 0.9)])
        mp_cfg = MediaPipeConfig(warmup_frame_count=0)
        frame = _make_frame()

        with create_tracker(custom_mp_cfg=mp_cfg) as tracker:
            # First call warms up the buffer
            tracker.process(frame)
            # Polling is fast, iterations=200 is sufficient for stable averages.
            benchmark.pedantic(
                lambda: tracker.process(frame),
                iterations=200,
                rounds=30,
                warmup_rounds=10,
            )
