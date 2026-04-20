"""Tests for webcam capture layer."""

from __future__ import annotations

import platform
import queue
import threading
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

import cv2
import numpy as np
import pytest

from hand_tracker.capture import (
    MAX_CONSECUTIVE_FAILURES,
    CaptureError,
    WebcamCapture,
    _TimeAnchor,
)
from hand_tracker.config import CameraConfig
from hand_tracker.types import Frame


@pytest.fixture
def mock_cfg() -> CameraConfig:
    """
    Fixture providing a default CameraConfig for testing.
    Since CameraConfig is a BaseModel (not BaseSettings), we instantiate it directly.
    """
    return CameraConfig(
        index=0,
        width=640,
        height=480,
        fps=30,
        disable_auto_exposure=True,
        disable_auto_focus=True,
        buffer_size=1,
        fourcc="MJPG",
    )


@pytest.fixture
def mock_cap() -> MagicMock:
    """Fixture providing a minimal cv2.VideoCapture mock."""
    cap = MagicMock()
    cap.isOpened.return_value = True

    # Default frame and read success
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)

    cap.get.side_effect = lambda prop: {
        3: 640.0,  # cv2.CAP_PROP_FRAME_WIDTH
        4: 480.0,  # cv2.CAP_PROP_FRAME_HEIGHT
        5: 30.0,  # cv2.CAP_PROP_FPS
        6: 1196444237.0,  # cv2.CAP_PROP_FOURCC ("MJPG")
        38: 1.0,  # cv2.CAP_PROP_BUFFERSIZE
    }.get(prop, 0.0)

    return cap


@pytest.fixture
def patch_cv2(monkeypatch: pytest.MonkeyPatch, mock_cap: MagicMock) -> MagicMock:
    """Patch cv2.VideoCapture globally for the test using pytest's monkeypatch."""
    monkeypatch.setattr(
        "hand_tracker.capture.cv2.VideoCapture", lambda *args, **kwargs: mock_cap
    )
    return mock_cap


class TestTimeAnchor:
    def test_current_us_increases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Avoid time.sleep in tests - mock clocks instead for deterministic execution
        t_perf = 100.0
        t_wall = 1600000000.0

        monkeypatch.setattr("hand_tracker.capture.time.perf_counter", lambda: t_perf)
        monkeypatch.setattr("hand_tracker.capture.time.time", lambda: t_wall)

        anchor = _TimeAnchor.now()

        # Advance mock perf_counter by 1ms (0.001 seconds)
        monkeypatch.setattr(
            "hand_tracker.capture.time.perf_counter", lambda: t_perf + 0.001
        )

        t2 = anchor.current_us()
        t1_expected = int(1600000000.0 * 1_000_000)

        assert t2 == t1_expected + 1000  # 1ms = 1000us

    def test_timestamp_is_16_digits(self) -> None:
        anchor = _TimeAnchor.now()
        ts = anchor.current_us()
        assert len(str(ts)) == 16


class TestWebcamCaptureOpen:
    def test_context_manager_opens_and_releases(
        self, mock_cfg: CameraConfig, patch_cv2: MagicMock
    ) -> None:
        with WebcamCapture(mock_cfg) as wc:
            assert wc.device_info is not None

        patch_cv2.release.assert_called_once()

    def test_device_info_populated(
        self, mock_cfg: CameraConfig, patch_cv2: MagicMock
    ) -> None:
        with WebcamCapture(mock_cfg) as wc:
            assert wc.device_info is not None
            assert wc.device_info.actual_width == 640
            assert wc.device_info.actual_height == 480

    def test_raises_capture_error_when_device_not_opened(
        self, mock_cfg: CameraConfig, patch_cv2: MagicMock
    ) -> None:
        patch_cv2.isOpened.return_value = False

        with (
            pytest.raises(CaptureError, match="Cannot open camera"),
            WebcamCapture(mock_cfg),
        ):
            pass

    def test_device_info_none_before_enter(self, mock_cfg: CameraConfig) -> None:
        wc = WebcamCapture(mock_cfg)
        assert wc.device_info is None

    def test_safe_enter_releases_on_failure_after_open(
        self,
        mock_cfg: CameraConfig,
        patch_cv2: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def mock_read_info(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Info failed")

        monkeypatch.setattr("hand_tracker.capture._read_device_info", mock_read_info)

        with pytest.raises(RuntimeError, match="Info failed"), WebcamCapture(mock_cfg):
            pass

        patch_cv2.release.assert_called_once()

    @pytest.mark.parametrize(
        "os_name, cfg_backend, expected_backend",
        [
            ("Windows", "AUTO", "CAP_DSHOW"),
            ("Linux", "AUTO", "CAP_ANY"),
            ("Darwin", "AUTO", "CAP_ANY"),
            ("Windows", "ANY", "CAP_ANY"),
        ],
    )
    def test_backend_selection(
        self,
        patch_cv2: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        os_name: str,
        cfg_backend: str,
        expected_backend: str,
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: os_name)
        cfg = CameraConfig(backend=cfg_backend)  # type: ignore[arg-type]

        with WebcamCapture(cfg) as wc:
            assert wc.device_info is not None
            assert wc.device_info.backend == expected_backend

    def test_properties_injected_successfully(
        self,
        mock_cfg: CameraConfig,
        patch_cv2: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import cv2

        patch_cv2.set.return_value = True

        # Mock VideoWriter.fourcc to avoid dependency on actual cv2 codec existence
        monkeypatch.setattr(
            "hand_tracker.capture.cv2.VideoWriter.fourcc", lambda *args: 1196444237
        )

        with WebcamCapture(mock_cfg):
            pass

        # Verify property sets with strict checks
        patch_cv2.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
        patch_cv2.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        patch_cv2.set.assert_any_call(cv2.CAP_PROP_FPS, 30)
        patch_cv2.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)
        patch_cv2.set.assert_any_call(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        patch_cv2.set.assert_any_call(cv2.CAP_PROP_AUTOFOCUS, 0)
        patch_cv2.set.assert_any_call(cv2.CAP_PROP_FOURCC, 1196444237)

    def test_property_set_failure_logs_warning_and_continues(
        self,
        mock_cfg: CameraConfig,
        patch_cv2: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import cv2

        mock_log = MagicMock()
        monkeypatch.setattr("hand_tracker.capture.log", mock_log)

        # Simulate failure for AUTOFOCUS only
        def mock_set(prop_id: int, value: float) -> bool:
            return bool(prop_id != cv2.CAP_PROP_AUTOFOCUS)

        patch_cv2.set.side_effect = mock_set

        with WebcamCapture(mock_cfg):
            pass

        # System should not crash (no exception raised), but must log exactly this warning
        mock_log.warning.assert_called_with(
            "failed to request camera property",
            extra={"property": "auto_focus", "requested": 0},
        )

    def test_mandatory_property_failure_raises(
        self,
        mock_cfg: CameraConfig,
        patch_cv2: MagicMock,
    ) -> None:
        """Verify that failure to set Width/Height results in a CaptureError."""
        # Simulate driver refusing the requested width
        patch_cv2.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 320.0,  # Negotiation failed (requested 640)
        }.get(prop, 0.0)

        with (
            pytest.raises(CaptureError, match="Camera refused mandatory width"),
            WebcamCapture(mock_cfg),
        ):
            pass


class TestWebcamCaptureIteration:
    def test_yields_frame_with_correct_shape(
        self, mock_cfg: CameraConfig, patch_cv2: MagicMock
    ) -> None:
        with WebcamCapture(mock_cfg) as wc:
            frame = next(iter(wc))

        assert isinstance(frame, Frame)
        assert frame.bgr.shape == (480, 640, 3)
        assert frame.frame_index >= 0

    def test_frame_index_increments(
        self, mock_cfg: CameraConfig, patch_cv2: MagicMock
    ) -> None:
        with WebcamCapture(mock_cfg) as wc:
            gen = iter(wc)
            f0 = next(gen)
            f1 = next(gen)
            f2 = next(gen)

        assert f0.frame_index >= 0
        assert f1.frame_index > f0.frame_index
        assert f2.frame_index > f1.frame_index

    def test_raises_after_max_consecutive_failures(
        self, mock_cfg: CameraConfig, patch_cv2: MagicMock
    ) -> None:
        patch_cv2.read.return_value = (False, None)

        with (
            pytest.raises(CaptureError, match="consecutive reads"),
            WebcamCapture(mock_cfg) as wc,
        ):
            for _ in wc:
                pass

        assert patch_cv2.read.call_count == MAX_CONSECUTIVE_FAILURES

    def test_raises_runtime_error_without_context_manager(
        self, mock_cfg: CameraConfig
    ) -> None:
        wc = WebcamCapture(mock_cfg)
        with pytest.raises(RuntimeError, match="context manager"):
            next(iter(wc))

    def test_jitter_detection_logs_warning(
        self,
        mock_cfg: CameraConfig,
        patch_cv2: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time

        mock_log = MagicMock()
        monkeypatch.setattr("hand_tracker.capture.log", mock_log)

        def jittery_read() -> tuple[bool, np.ndarray]:
            # Simulate a 60ms delay that exceeds the threshold
            time.sleep(0.06)
            return (True, np.zeros((480, 640, 3), dtype=np.uint8))

        patch_cv2.read.side_effect = jittery_read

        with WebcamCapture(mock_cfg) as wc:
            gen = iter(wc)
            next(gen)
            next(gen)

        # Check if acquisition jitter was logged
        assert any(
            "acquisition jitter detected" in call.args[0]
            for call in mock_log.warning.mock_calls
        )

    def test_concurrent_close_during_iteration(
        self, mock_cfg: CameraConfig, patch_cv2: MagicMock
    ) -> None:
        """Verify that the iterator exits cleanly when the device is closed from another thread."""
        wc = WebcamCapture(mock_cfg)
        wc.__enter__()
        try:
            first_frame_seen = threading.Event()
            exc_queue: queue.Queue[Exception] = queue.Queue()

            def run_iterator() -> None:
                try:
                    for _ in wc:
                        first_frame_seen.set()
                except Exception as e:
                    exc_queue.put(e)

            thread = threading.Thread(target=run_iterator, daemon=True)
            thread.start()

            # Wait for at least one frame to be processed.
            # We use 0.5s to ensure the producer thread has time to start and
            # push frames even under high OS CPU load/scheduling jitter.
            assert first_frame_seen.wait(timeout=0.5)

            # Concurrently close the device while the thread is iterating
            wc.__exit__(None, None, None)

            # Wait for thread to recognize stop event.
            # On Windows, thread destruction and signal propagation can take
            # up to 15-50ms due to timer resolution. 0.2s is a safe buffer
            # for deterministic CI/CD runs.
            thread.join(timeout=0.2)
            assert not thread.is_alive()

            # Surface any exceptions swallowed by the thread
            if not exc_queue.empty():
                pytest.fail(f"Iterator thread failed: {exc_queue.get()}")
        finally:
            # Safety exit if it wasn't already called
            if wc._cap is not None:
                wc.__exit__(None, None, None)

    def test_recovers_from_sporadic_read_failures(
        self, mock_cfg: CameraConfig, patch_cv2: MagicMock
    ) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        class MockRead:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self) -> tuple[bool, np.ndarray | None]:
                self.calls += 1
                if self.calls in (2, 3):
                    return (False, None)
                return (True, frame)

        patch_cv2.read.side_effect = MockRead()

        with WebcamCapture(mock_cfg) as wc:
            gen = iter(wc)
            f0 = next(gen)
            f1 = next(gen)  # Skips failures
            f2 = next(gen)

        assert f0.frame_index >= 0
        assert f1.frame_index > f0.frame_index
        assert f2.frame_index > f1.frame_index


class TestWebcamCapturePerformance:
    """Benchmark tests for the producer-consumer transfer latency."""

    @pytest.mark.benchmark(group="capture")
    def test_benchmark_frame_acquisition_latency(
        self,
        benchmark: BenchmarkFixture,
        mock_cfg: CameraConfig,
        patch_cv2: MagicMock,
    ) -> None:
        """Measure the latency of acquiring a frame from the async queue."""
        with WebcamCapture(mock_cfg) as wc:
            # Pre-create iterator once: avoids per-call generator allocation
            # that would inflate measurement with __iter__ re-entry overhead.
            gen = iter(wc)

            def _get_frame() -> None:
                next(gen)

            # Use realistic iterations/rounds for threaded I/O to avoid hangs.
            benchmark.pedantic(_get_frame, iterations=1, rounds=100, warmup_rounds=10)
