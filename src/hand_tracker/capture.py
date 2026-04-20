"""
Webcam capture layer for HandForge.

Provides a robust, thread-safe interface for high-performance video acquisition.
The implementation is platform-aware, applying specific optimizations to ensure
consistent behavior across different operating systems.

Responsibilities:
    - Manage capture device lifecycle (open -> iterate -> release).
    - Configure hardware properties (resolution, FPS, auto-controls).
    - Acquire raw BGR frames with microsecond-precision timestamps.
    - Monitor acquisition health via jitter detection and frame drop detection.

Design Decisions:
    - Context Manager Pattern: Ensures deterministic resource release, addressing
      platform-specific differences in hardware handle management.
    - Frame Drop Detection: Monitors consecutive read() failures. After reaching
      the MAX_CONSECUTIVE_FAILURES threshold, the module raises a CaptureError
      to allow for upper-layer recovery instead of silent failure.
    - Adaptive Backend Selection: Dynamically selects the most stable capture
      backend (e.g., DirectShow, V4L2) based on the host environment to minimize
      initialization latency.
    - Asynchronous Producer-Consumer: Uses a dedicated background thread (Producer)
      to handle hardware I/O independently. This completely decoupled design eliminates
      shared consumer locks, ensuring UI or modeling pipelines never block or stutter
      due to camera logic.
    - Zero-Latency Tracking: The internal transfer queue is rigidly capped (maxsize=1)
      and aggressively drops older unprocessed frames if the consumer is slow. This
      guarantees the caller always evaluates the absolute most recent reality, avoiding
      frame stacking lag in real-time AI inferences.
    - High-Resolution Timing: Utilizes monotonic performance counters anchored
      to a wall-clock origin to bypass platform-specific limitations in system
      clock resolution (e.g. 15ms increments on legacy schedulers).
"""

from __future__ import annotations

import contextlib
import platform
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, cast

import cv2
import numpy as np
import numpy.typing as npt

from hand_tracker.config import CameraConfig
from hand_tracker.logger import get_logger
from hand_tracker.types import Frame
from hand_tracker.utils import US_PER_MS, US_PER_SEC

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CONSECUTIVE_FAILURES: Final[int] = 10
MIN_ACCEPTABLE_FPS_RATIO: Final[float] = (
    0.8  # Warn if actual_fps is below 80% of cfg.fps
)

_BACKEND_NAME_MAP: Final[dict[int, str]] = {
    cv2.CAP_DSHOW: "CAP_DSHOW",
    cv2.CAP_ANY: "CAP_ANY",
    cv2.CAP_MSMF: "CAP_MSMF",
    cv2.CAP_V4L2: "CAP_V4L2",
}

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class CaptureError(RuntimeError):
    """Raised when the capture device cannot be opened or repeatedly fails."""


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """
    Actual device parameters reported by OpenCV after open().

    These may differ from the requested CameraConfig values because
    the driver negotiates the nearest supported mode.
    """

    index: int
    actual_width: int
    actual_height: int
    actual_fps: float
    backend: str
    fourcc: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_backend(cfg_backend: str) -> int:
    """Select backend based on config, with an OS-specific fallback for AUTO."""
    if cfg_backend != "AUTO":
        mapping: dict[str, int] = {
            "ANY": int(cv2.CAP_ANY),
            "DSHOW": int(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)),
            "MSMF": int(getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
            "V4L2": int(getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)),
            "AVFOUNDATION": int(getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)),
        }
        return mapping.get(cfg_backend, int(cv2.CAP_ANY))

    # Fallback smart defaults
    if platform.system() == "Windows":
        return int(cv2.CAP_DSHOW)
    return int(cv2.CAP_ANY)


def _get_backend_name(backend_id: int) -> str:
    """Convert backend ID to human-readable name using SSOT map."""
    return _BACKEND_NAME_MAP.get(backend_id, f"UNKNOWN({backend_id})")


def _open_device(cfg: CameraConfig) -> cv2.VideoCapture:
    """
    Open the capture device and apply requested parameters.

    Robustness:
    - Sets CAP_PROP_BUFFERSIZE=1 to minimize lag.
    - Locks focus/exposure if requested to prevent landmarks jitter.
    """
    backend = _select_backend(cfg.backend)
    backend_name = _get_backend_name(backend)

    log.debug(
        "opening capture device",
        extra={"index": cfg.index, "backend": backend_name},
    )

    cap = cv2.VideoCapture(cfg.index, backend)

    if not cap.isOpened():
        raise CaptureError(
            f"Cannot open camera index={cfg.index} with backend={backend_name}. "
            "Check connection and process permissions."
        )

    # Apply properties and check for silent failures
    # CRITICAL: Execution order matters in OpenCV!
    # FOURCC must be set FIRST. High resolutions/FPS will be silently rejected
    # if the default codec (e.g. YUYV) cannot support the required USB bandwidth.
    # Therefore, we use a List to strictly enforce this sequence.
    props_sequence: list[tuple[str, tuple[int, int | float]]] = [
        ("fourcc", (cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*cfg.fourcc))),
        ("width", (cv2.CAP_PROP_FRAME_WIDTH, cfg.width)),
        ("height", (cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)),
        ("fps", (cv2.CAP_PROP_FPS, cfg.fps)),
        ("buffer_size", (cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)),
    ]

    if cfg.disable_auto_exposure:
        # 0.25 (or 0) usually means manual mode in many DirectShow drivers
        props_sequence.append(("auto_exposure", (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)))

    if cfg.disable_auto_focus:
        props_sequence.append(("auto_focus", (cv2.CAP_PROP_AUTOFOCUS, 0)))

    for name, (prop_id, value) in props_sequence:
        ok = cap.set(prop_id, value)
        if not ok:
            log.warning(
                "failed to request camera property",
                extra={"property": name, "requested": value},
            )

        # Standard cap.set() can return True even if the hardware silently
        # rejected or modified the value to the nearest supported mode.
        # We MUST re-read the property to verify what the driver actually negotiated.
        actual = cap.get(prop_id)

        # CRITICAL VALIDATION: Resolution must match exactly.
        # If the driver silently falls back to a different resolution, all
        # downstream MediaPipe and Unity coordinate mappings will be scaled
        # incorrectly, leading to broken tracking.
        if name in ("width", "height") and int(actual) != int(value):
            cap.release()
            raise CaptureError(
                f"Camera refused mandatory {name}. Requested {value}, but got {actual}. "
                "Hardware limitation reached."
            )

        # FPS Verification: Many drivers vary slightly (e.g. 29.97 vs 30).
        # We allow a 10% tolerance but log a warning if it's too far off.
        if name == "fps" and actual < value * (1.0 - 0.1):
            log.warning(
                "camera driver negotiated lower FPS than requested",
                extra={"requested": value, "actual": actual},
            )

    return cap


def _decode_fourcc(v: int) -> str:
    """Decode a 32-bit FOURCC integer to a 4-character string."""
    if v <= 0:
        return ""
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


def _read_device_info(cap: cv2.VideoCapture, index: int, backend: int) -> DeviceInfo:
    return DeviceInfo(
        index=index,
        actual_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        actual_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        actual_fps=cap.get(cv2.CAP_PROP_FPS),
        backend=_get_backend_name(backend),
        fourcc=_decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC))),
    )


# ---------------------------------------------------------------------------
# Timestamp anchor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TimeAnchor:
    """
    Anchors perf_counter to wall-clock time at device open.

    perf_counter() has sub-microsecond resolution on Windows (unlike
    time.time() which has ~100ns resolution but ~15ms *update* granularity
    on older Windows versions).  We record both at the same instant and use
    perf_counter deltas for all subsequent timestamps.
    """

    wall_us: int  # time.time() at anchor point, in microseconds
    perf_origin: float  # time.perf_counter() at anchor point, in seconds

    @classmethod
    def now(cls) -> _TimeAnchor:
        """
        Create a new time anchor for high-resolution performance timing.

        Attempts to sample both clocks with minimal jitter between reads.
        """
        perf = time.perf_counter()
        wall = time.time()
        return cls(wall_us=int(wall * US_PER_SEC), perf_origin=perf)

    def current_us(self) -> int:
        """Return current wall-clock time in microseconds."""
        delta_us = int((time.perf_counter() - self.perf_origin) * US_PER_SEC)
        return self.wall_us + delta_us


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------


class WebcamCapture:
    """
    Context manager that yields Frame objects from a webcam.

    Usage
    -----
    ::

        cfg = CameraConfig()
        with WebcamCapture(cfg) as capture:
            for frame in capture:
                process(frame)

    The ``for`` loop runs until:
    - The caller ``break``s out.
    - The device fails MAX_CONSECUTIVE_FAILURES times in a row.

    Attributes
    ----------
    device_info:
        Populated after ``__enter__``.  None before.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._anchor: _TimeAnchor | None = None
        self.device_info: DeviceInfo | None = None

        # Async Producer-Consumer State
        self._frame_queue: queue.Queue[Frame | Exception] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> WebcamCapture:
        cap = None
        try:
            cap = _open_device(self._cfg)
            self._cap = cap
            self._anchor = _TimeAnchor.now()

            backend = _select_backend(self._cfg.backend)
            self.device_info = _read_device_info(cap, self._cfg.index, backend)

            if self.device_info.fourcc and self.device_info.fourcc != self._cfg.fourcc:
                log.warning(
                    "camera driver silently rejected requested format",
                    extra={
                        "requested_fourcc": self._cfg.fourcc,
                        "actual_fourcc": self.device_info.fourcc,
                        "note": "USB bandwidth may limit actual FPS or resolution.",
                    },
                )

            if (
                self.device_info.actual_fps > 0
                and self.device_info.actual_fps
                < self._cfg.fps * MIN_ACCEPTABLE_FPS_RATIO
            ):
                log.warning(
                    "camera driver cannot sustain requested FPS",
                    extra={
                        "requested_fps": self._cfg.fps,
                        "actual_fps": self.device_info.actual_fps,
                    },
                )

            log.info(
                "capture device opened",
                extra={
                    "index": self.device_info.index,
                    "actual_width": self.device_info.actual_width,
                    "actual_height": self.device_info.actual_height,
                    "actual_fps": self.device_info.actual_fps,
                    "actual_fourcc": self.device_info.fourcc,
                    "backend": self.device_info.backend,
                    "requested_width": self._cfg.width,
                    "requested_height": self._cfg.height,
                    "requested_fps": self._cfg.fps,
                },
            )

            # Spawn dedicated hardware I/O thread
            self._worker_thread = threading.Thread(
                target=self._read_loop,
                name=f"CaptureWorker-{self._cfg.index}",
                daemon=True,
            )
            self._worker_thread.start()

            return self
        except Exception:
            if cap is not None:
                cap.release()
            self._cap = None
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        # Request grace shutdown of the background thread
        self._stop_event.set()

        # CRITICAL: Releasing the hardware handle interrupts any blocking
        # cap.read() call, allowing the worker thread to exit immediately.
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._worker_thread is not None:
            # Interruption is now near-instant, so we can use a smaller timeout.
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        log.info("capture device released", extra={"index": self._cfg.index})

    # ------------------------------------------------------------------
    # Worker Thread (Producer)
    # ------------------------------------------------------------------

    def _push_to_queue(self, item: Frame | Exception) -> None:
        """Pushes to queue, aggressively dropping old frames to ensure zero latency."""
        try:
            self._frame_queue.put_nowait(item)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._frame_queue.get_nowait()
            self._frame_queue.put(item)

    def _read_loop(self) -> None:
        """Background thread executing cap.read() synchronously."""
        consecutive_failures = 0
        frame_index = 0

        if self._anchor is None or self.device_info is None or self._cap is None:
            return

        last_ts_us = self._anchor.wall_us
        expected_delta_us = (1.0 / self.device_info.actual_fps) * US_PER_SEC
        jitter_threshold_us = expected_delta_us * self._cfg.jitter_threshold_multiplier

        while not self._stop_event.is_set():
            ok, bgr = self._cap.read()
            now_us = self._anchor.current_us()

            if self._stop_event.is_set():
                break

            if not ok or bgr is None:
                consecutive_failures += 1
                log.warning(
                    "frame read failed",
                    extra={
                        "consecutive_failures": consecutive_failures,
                        "max": MAX_CONSECUTIVE_FAILURES,
                    },
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    exc = CaptureError(
                        f"Camera index={self._cfg.index} failed "
                        f"{MAX_CONSECUTIVE_FAILURES} consecutive reads. "
                        "Device may have been disconnected."
                    )
                    self._push_to_queue(exc)
                    break
                continue

            # Jitter detection
            delta_us = now_us - last_ts_us
            if frame_index > 0 and delta_us > jitter_threshold_us:
                log.warning(
                    "acquisition jitter detected",
                    extra={
                        "delta_ms": delta_us / US_PER_MS,
                        "expected_ms": expected_delta_us / US_PER_MS,
                    },
                )

            consecutive_failures = 0
            last_ts_us = now_us

            bgr_uint8 = cast(npt.NDArray[np.uint8], bgr)
            frame = Frame(
                bgr=bgr_uint8,
                timestamp_us=now_us,
                frame_index=frame_index,
                is_mirrored=self._cfg.mirror_input,
            )
            self._push_to_queue(frame)
            frame_index += 1

        log.debug("capture worker thread finished")

    # ------------------------------------------------------------------
    # Iterator protocol (Consumer)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        if self._worker_thread is None or self.device_info is None:
            raise RuntimeError(
                "WebcamCapture must be used as a context manager before iterating."
            )

        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=0.1)
                if isinstance(item, Exception):
                    raise item
                yield item
            except queue.Empty:
                if self._worker_thread is None or not self._worker_thread.is_alive():
                    # Thread died unexpectedly without emitting exception
                    break
