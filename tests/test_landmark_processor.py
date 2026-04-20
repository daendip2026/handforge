"""Tests for LandmarkProcessor, _FpsEstimator, and console_summary."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

import pytest

from hand_tracker.landmark_processor import (
    LandmarkProcessor,
    _FpsEstimator,
    full_landmark_dump,
)
from hand_tracker.types import (
    LANDMARK_COUNT,
    FrameResult,
    Handedness,
    LandmarkPoint,
    ProcessedFrame,
    RawHandResult,
)
from hand_tracker.utils import (
    CONSOLE_WIDTH,
    console_summary,
)

# ---------------------------------------------------------------------------
# Constants & Fixtures
# ---------------------------------------------------------------------------

_BASE_TS: int = 1_700_000_000_000_000  # Fixed base timestamp (microseconds)

# Nominal frame rate intervals in microseconds
_FPS_DELTA_MAP: dict[int, int] = {
    30: 33_333,
    60: 16_666,
    90: 11_111,
    120: 8_333,
    240: 4_166,
}
_DEFAULT_DELTA_US: int = _FPS_DELTA_MAP[30]
_DEFAULT_CONFIDENCE: float = 0.95  # Standard test confidence
_DEFAULT_INFERENCE_US: int = 8_000  # Standard simulated inference time
_DEFAULT_WINDOW_SIZE: int = 30  # Default FPS smoothing window size


@pytest.fixture
def landmark_point_factory() -> Iterator[Callable[[int], LandmarkPoint]]:
    """Provides a factory for LandmarkPoint objects with default values."""

    def _create(index: int = 0) -> LandmarkPoint:
        return LandmarkPoint(
            index=index,
            name=f"LM_{index}",
            x=index * 0.01,
            y=index * 0.02,
            z=0.0,
            wx=index * 0.001,
            wy=index * 0.002,
            wz=0.0,
        )

    yield _create


@pytest.fixture
def landmarks_factory(
    landmark_point_factory: Callable[[int], LandmarkPoint],
) -> Iterator[Callable[[], tuple[LandmarkPoint, ...]]]:
    """Provides a factory for a full set of 21 landmarks."""

    def _create() -> tuple[LandmarkPoint, ...]:
        return tuple(landmark_point_factory(i) for i in range(LANDMARK_COUNT))

    yield _create


@pytest.fixture
def raw_hand_factory(
    landmarks_factory: Callable[[], tuple[LandmarkPoint, ...]],
) -> Iterator[Callable[..., RawHandResult]]:
    """Provides a factory for RawHandResult objects."""

    def _create(
        timestamp_us: int = _BASE_TS,
        handedness: Handedness = Handedness.RIGHT,
        confidence: float = _DEFAULT_CONFIDENCE,
    ) -> RawHandResult:
        return RawHandResult(
            landmarks=landmarks_factory(),
            handedness=handedness,
            confidence=confidence,
            timestamp_us=timestamp_us,
            inference_time_us=_DEFAULT_INFERENCE_US,
        )

    yield _create


@pytest.fixture
def frame_result_factory(
    raw_hand_factory: Callable[..., RawHandResult],
) -> Iterator[Callable[..., FrameResult]]:
    """Provides a factory for FrameResult objects."""

    def _create(
        hands: tuple[RawHandResult, ...] | None = None,
        frame_index: int = 0,
        timestamp_us: int = _BASE_TS,
        is_mirrored: bool = False,
    ) -> FrameResult:
        if hands is None:
            hands = (raw_hand_factory(timestamp_us=timestamp_us),)

        return FrameResult(
            hands=hands,
            timestamp_us=timestamp_us,
            frame_index=frame_index,
            is_mirrored=is_mirrored,
            inference_time_us=_DEFAULT_INFERENCE_US,
        )

    yield _create


@pytest.fixture
def clock_gen() -> Iterator[Callable[..., Iterator[int]]]:
    """Generates a monotonic sequence of timestamps in microseconds."""

    def _gen(
        start_us: int = _BASE_TS, delta_us: int = _DEFAULT_DELTA_US
    ) -> Iterator[int]:
        current = start_us
        while True:
            yield current
            current += delta_us

    yield _gen


@pytest.fixture
def processor() -> LandmarkProcessor:
    """Provides a fresh LandmarkProcessor with nominal window size."""
    return LandmarkProcessor(window_size=_DEFAULT_WINDOW_SIZE)


# ---------------------------------------------------------------------------
# _FpsEstimator
# ---------------------------------------------------------------------------


class TestFpsEstimator:
    """Tests for the internal _FpsEstimator logic."""

    @pytest.mark.parametrize("window_size", [2, 10, 30, 100])
    def test_single_sample_returns_nan(self, window_size: int) -> None:
        """Scenario: Single data point received. Expected: No FPS estimate possible."""
        est = _FpsEstimator(window_size=window_size)
        fps = est.update(_BASE_TS)
        assert math.isnan(fps)

    @pytest.mark.parametrize("target_fps", _FPS_DELTA_MAP.keys())
    def test_fps_estimation_accuracy(
        self, clock_gen: Callable[..., Iterator[int]], target_fps: int
    ) -> None:
        """Scenario: Two points exactly at target interval. Expected: Correct FPS estimate."""
        est = _FpsEstimator(window_size=_DEFAULT_WINDOW_SIZE)
        delta = _FPS_DELTA_MAP[target_fps]
        clock = clock_gen(delta_us=delta)

        est.update(next(clock))
        fps = est.update(next(clock))
        assert fps == pytest.approx(float(target_fps), rel=0.01)

    @pytest.mark.parametrize("target_fps", [30, 60, 120])
    def test_window_stabilises_fps(
        self, clock_gen: Callable[..., Iterator[int]], target_fps: int
    ) -> None:
        """Scenario: Filling the window with steady frames. Expected: Stable FPS output."""
        window_size = 10
        est = _FpsEstimator(window_size=window_size)
        delta = _FPS_DELTA_MAP[target_fps]
        clock = clock_gen(delta_us=delta)

        for _ in range(window_size):
            est.update(next(clock))
        fps = est.update(next(clock))
        assert fps == pytest.approx(float(target_fps), rel=0.01)

    def test_window_evicts_old_samples(
        self, clock_gen: Callable[..., Iterator[int]]
    ) -> None:
        """Scenario: Framerate shift from 30 to 60 FPS. Expected: Average reflects only recent window."""
        est = _FpsEstimator(window_size=5)
        clock_30 = clock_gen(delta_us=_FPS_DELTA_MAP[30])

        # Fill window at 30fps
        ts = _BASE_TS
        for _ in range(5):
            ts = next(clock_30)
            est.update(ts)

        # Switch to 60fps (16.67ms delta)
        delta_60 = _FPS_DELTA_MAP[60]
        clock_60 = clock_gen(start_us=ts + delta_60, delta_us=delta_60)
        fps = 0.0
        for _ in range(5):
            fps = est.update(next(clock_60))

        # Window should now reflect ~60fps, not 30fps
        assert fps == pytest.approx(60.0, rel=0.01)

    def test_duplicate_timestamps_return_nan(self) -> None:
        """Scenario: Hardware sends two frames with identical TS. Expected: Guard against div-by-zero."""
        est = _FpsEstimator(window_size=30)
        est.update(_BASE_TS)
        fps = est.update(_BASE_TS)  # identical timestamp
        assert math.isnan(fps)

    def test_reset_clears_window(self, clock_gen: Callable[..., Iterator[int]]) -> None:
        """Scenario: Processor reset. Expected: Historic timestamps purged."""
        est = _FpsEstimator(window_size=_DEFAULT_WINDOW_SIZE)
        clock = clock_gen()
        est.update(next(clock))
        est.update(next(clock))
        est.reset()
        fps = est.update(next(clock))
        assert math.isnan(fps)

    @pytest.mark.parametrize("invalid_size", [0, 1, -10])
    def test_invalid_window_size_raises(self, invalid_size: int) -> None:
        """Scenario: Configuration error (window < 2). Expected: ValueError for safety."""
        with pytest.raises(ValueError, match="window_size"):
            _FpsEstimator(window_size=invalid_size)

    def test_backwards_timestamps_returns_nan(self) -> None:
        """Scenario: System clock drift (non-monotonic). Expected: Silent NaN instead of negative FPS."""
        est = _FpsEstimator(window_size=_DEFAULT_WINDOW_SIZE)
        est.update(_BASE_TS)
        # Clock drift backwards
        fps = est.update(_BASE_TS - 1000)
        assert math.isnan(fps)


# ---------------------------------------------------------------------------
# LandmarkProcessor
# ---------------------------------------------------------------------------


class TestLandmarkProcessor:
    """Tests for LandmarkProcessor orchestration and data translation."""

    def test_update_returns_processed_frame(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
    ) -> None:
        """Scenario: Nominal update. Expected: Returns a ProcessedFrame instance."""
        result = frame_result_factory()
        processed = processor.update(result)
        assert isinstance(processed, ProcessedFrame)

    def test_processed_frame_inherits_metadata(
        self,
        processor: LandmarkProcessor,
        raw_hand_factory: Callable[..., RawHandResult],
        frame_result_factory: Callable[..., FrameResult],
    ) -> None:
        """Scenario: Complex metadata in source. Expected: Full parity in downstream frame."""
        raw_frame = frame_result_factory(
            frame_index=42,
            timestamp_us=_BASE_TS + 1_000,
            is_mirrored=True,
            hands=(raw_hand_factory(handedness=Handedness.LEFT, confidence=0.88),),
        )
        processed = processor.update(raw_frame)

        assert processed.frame_index == 42
        assert processed.timestamp_us == _BASE_TS + 1_000
        assert processed.inference_time_us == _DEFAULT_INFERENCE_US
        assert processed.is_mirrored is True
        assert len(processed.hands) == 1
        assert processed.hands[0].handedness == Handedness.LEFT
        assert processed.hands[0].confidence == pytest.approx(0.88)

    def test_landmarks_preserved_exactly(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
    ) -> None:
        """Scenario: High-precision landmark data. Expected: Floating point bit-parity preserved."""
        raw_frame = frame_result_factory()
        processed = processor.update(raw_frame)

        assert len(processed.hands[0].landmarks) == LANDMARK_COUNT
        for original, result in zip(
            raw_frame.hands[0].landmarks,
            processed.hands[0].landmarks,
            strict=True,
        ):
            assert result.index == original.index
            assert result.x == pytest.approx(original.x)
            assert result.wx == pytest.approx(original.wx)

    def test_fps_nan_on_first_frame(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
    ) -> None:
        """Scenario: Pipeline cold start. Expected: No FPS until second frame."""
        processed = processor.update(frame_result_factory(timestamp_us=_BASE_TS))
        assert math.isnan(processed.fps)

    @pytest.mark.parametrize("target_fps", [30, 120])
    def test_fps_accuracy_at_nominal_speeds(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
        clock_gen: Callable[..., Iterator[int]],
        target_fps: int,
    ) -> None:
        """Scenario: Window saturated at target FPS. Expected: Average FPS stabilizes."""
        delta = _FPS_DELTA_MAP[target_fps]
        clock = clock_gen(delta_us=delta)
        last: ProcessedFrame | None = None

        # Feed enough frames to fill the window
        for i in range(_DEFAULT_WINDOW_SIZE + 1):
            last = processor.update(
                frame_result_factory(
                    frame_index=i,
                    timestamp_us=next(clock),
                )
            )
        assert last is not None
        assert last.fps == pytest.approx(float(target_fps), rel=0.01)

    def test_processed_count_increments(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
    ) -> None:
        """Scenario: Processing multiple frames. Expected: Internal counter tracks lifecycle."""
        for i in range(5):
            processor.update(frame_result_factory(frame_index=i))
        assert processor.processed_count == 5

    def test_reset_clears_state(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
        clock_gen: Callable[..., Iterator[int]],
    ) -> None:
        """Scenario: Dynamic pipeline reset. Expected: Counters and estimators purged."""
        clock = clock_gen()
        for i in range(10):
            processor.update(
                frame_result_factory(
                    frame_index=i,
                    timestamp_us=next(clock),
                )
            )
        processor.reset()
        assert processor.processed_count == 0
        processed = processor.update(frame_result_factory(timestamp_us=next(clock)))
        assert math.isnan(processed.fps)

    def test_invalid_window_size_raises_at_processor_level(self) -> None:
        """Verify that LandmarkProcessor also guards against invalid window sizes."""
        with pytest.raises(ValueError, match="window_size"):
            LandmarkProcessor(window_size=1)

    def test_processed_frame_is_frozen(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
    ) -> None:
        """Scenario: Downstream attempt to mutate. Expected: Frozen dataclass raises error."""
        processed = processor.update(frame_result_factory())
        with pytest.raises(AttributeError):
            processed.fps = 999.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        "handedness_seq",
        [
            [],  # No hands
            [Handedness.RIGHT],  # Single Right
            [Handedness.LEFT],  # Single Left
            [Handedness.RIGHT, Handedness.LEFT],  # Dual (R -> L)
            [Handedness.LEFT, Handedness.RIGHT],  # Dual (L -> R)
        ],
    )
    def test_multi_hand_processing(
        self,
        processor: LandmarkProcessor,
        raw_hand_factory: Callable[..., RawHandResult],
        frame_result_factory: Callable[..., FrameResult],
        handedness_seq: list[Handedness],
    ) -> None:
        """Scenario: Symmetric hand ordering. Expected: Correct mapping regardless of input order."""
        hands = tuple(raw_hand_factory(handedness=h) for h in handedness_seq)
        raw_frame = frame_result_factory(hands=hands)
        processed = processor.update(raw_frame)

        assert len(processed.hands) == len(handedness_seq)
        for i, expected_handedness in enumerate(handedness_seq):
            assert processed.hands[i].handedness == expected_handedness

    def test_empty_frame_advances_fps(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
        clock_gen: Callable[..., Iterator[int]],
    ) -> None:
        """Scenario: Hardware active but no hands detected. Expected: FPS still calculates correctly."""
        clock = clock_gen()
        # Initial frame
        processor.update(frame_result_factory(hands=(), timestamp_us=next(clock)))
        # Second frame
        processed = processor.update(
            frame_result_factory(hands=(), timestamp_us=next(clock))
        )

        assert len(processed.hands) == 0
        assert not math.isnan(processed.fps)
        assert processed.fps == pytest.approx(30.0, rel=0.01)


# ---------------------------------------------------------------------------
# console_summary / full_landmark_dump
# ---------------------------------------------------------------------------


class TestConsoleSummary:
    """Tests for terminal visualization and formatting."""

    @pytest.fixture
    def processed_frame(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
        clock_gen: Callable[..., Iterator[int]],
    ) -> ProcessedFrame:
        """Helper to get a nominal processed frame with stable FPS."""
        clock = clock_gen()
        # Feed enough frames to get a real FPS value
        for i in range(_DEFAULT_WINDOW_SIZE + 1):
            processor.update(
                frame_result_factory(frame_index=i, timestamp_us=next(clock))
            )
        # Return the saturated frame which has a real FPS value
        return processor.update(
            frame_result_factory(
                frame_index=_DEFAULT_WINDOW_SIZE + 1,
                timestamp_us=next(clock),
                hands=(
                    RawHandResult(
                        landmarks=tuple(
                            LandmarkPoint(i, f"LM_{i}", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
                            for i in range(LANDMARK_COUNT)
                        ),
                        handedness=Handedness.RIGHT,
                        confidence=0.99,
                        timestamp_us=_BASE_TS,
                        inference_time_us=_DEFAULT_INFERENCE_US,
                    ),
                ),
            )
        )

    def test_summary_contains_essential_metadata(
        self, processed_frame: ProcessedFrame
    ) -> None:
        """Scenario: Standard summary generation. Expected: Frame index and TS present."""
        summary = console_summary(processed_frame)
        assert str(processed_frame.frame_index) in summary
        assert str(processed_frame.timestamp_us) in summary
        assert Handedness.RIGHT.value in summary

    def test_summary_shows_wrist_and_fingertips(
        self, processed_frame: ProcessedFrame
    ) -> None:
        """Scenario: Compact view check. Expected: Key landmarks visible for IK verification."""
        summary = console_summary(processed_frame)
        expected_names = (
            "WRIST",
            "THUMB_TIP",
            "INDEX_TIP",
            "MIDDLE_TIP",
            "RING_TIP",
            "PINKY_TIP",
        )
        for name in expected_names:
            assert name in summary

    def test_summary_width_boundary(self, processed_frame: ProcessedFrame) -> None:
        """Scenario: Production design constraint. Expected: Summary never exceeds console width."""

        summary = console_summary(processed_frame)
        for line in summary.splitlines():
            # Stripping ANSI codes if they were present (none currently, but good practice)
            assert len(line) <= CONSOLE_WIDTH

    def test_nan_fps_renders_dashes(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
    ) -> None:
        """Scenario: Initial frame formatting. Expected: Placeholder dashes for unknown FPS."""
        processed = processor.update(frame_result_factory())
        summary = console_summary(processed)
        assert "---" in summary

    def test_summary_shows_no_hands_message(
        self,
        processor: LandmarkProcessor,
        frame_result_factory: Callable[..., FrameResult],
    ) -> None:
        """Scenario: No hands detected. Expected: Clear warning message in terminal."""
        processed = processor.update(frame_result_factory(hands=()))
        summary = console_summary(processed)
        assert "NO HANDS DETECTED" in summary

    def test_full_dump_integrity(self, processed_frame: ProcessedFrame) -> None:
        """Scenario: In-depth data inspection. Expected: All 21 landmarks serialized."""
        dump = full_landmark_dump(processed_frame)
        for i in range(LANDMARK_COUNT):
            assert f"LM_{i}" in dump


# ---------------------------------------------------------------------------
# Performance Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="landmark-processor")
def test_benchmark_processor_update(
    benchmark: BenchmarkFixture,
    processor: LandmarkProcessor,
    frame_result_factory: Callable[..., FrameResult],
) -> None:
    """Benchmark the core LandmarkProcessor.update() latency."""
    result = frame_result_factory()
    benchmark(processor.update, result)


@pytest.mark.benchmark(group="landmark-processor")
def test_benchmark_console_summary(
    benchmark: BenchmarkFixture,
    processor: LandmarkProcessor,
    frame_result_factory: Callable[..., FrameResult],
    clock_gen: Callable[..., Iterator[int]],
) -> None:
    """Benchmark console_summary() string formatting latency."""
    # Setup a realistic frame
    clock = clock_gen()
    processor.update(frame_result_factory(timestamp_us=next(clock)))
    processed = processor.update(frame_result_factory(timestamp_us=next(clock)))

    benchmark(console_summary, processed)
