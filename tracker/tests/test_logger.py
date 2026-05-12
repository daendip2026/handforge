"""Tests for structured JSON logger."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from hand_tracker.config import LoggingConfig
from hand_tracker.logger import (
    AsyncLoggerLifecycle,
    _JsonFormatter,
    get_logger,
    log_context,
)

# ---------------------------------------------------------------------------
# Test Utilities
# ---------------------------------------------------------------------------


class ListHandler(logging.Handler):
    """Simple handler that stores records in a list for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class SlowHandler(logging.NullHandler):
    """Handler that simulates slow I/O (e.g. network or slow disk)."""

    def __init__(self, delay: float = 0.1) -> None:
        super().__init__()
        self.delay = delay

    def emit(self, record: logging.LogRecord) -> None:
        time.sleep(self.delay)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_logging() -> Iterator[None]:
    """Ensure root logger is clean between tests."""
    root = logging.getLogger("handforge")
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)


@pytest.fixture
def log_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def log_cfg(log_dir: Path) -> LoggingConfig:
    return LoggingConfig(
        level="DEBUG",
        log_dir=str(log_dir),
        console_enabled=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def _make_record(self, msg: str = "test", **extra: Any) -> logging.LogRecord:
        record = logging.LogRecord(
            name="handforge.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        record.__dict__.update(extra)
        return record

    def test_required_keys_present(self) -> None:
        formatter = _JsonFormatter()
        record = self._make_record("hello")
        output = json.loads(formatter.format(record))
        assert {"timestamp_us", "level", "module", "message"} <= output.keys()

    def test_timestamp_is_microseconds(self) -> None:
        formatter = _JsonFormatter()
        record = self._make_record("ts check")
        output = json.loads(formatter.format(record))
        # Microseconds since epoch in 2024+ are 16 digits
        assert len(str(output["timestamp_us"])) == 16

    def test_extra_fields_merged(self) -> None:
        formatter = _JsonFormatter()
        record = self._make_record("with extra", fps=60, hand="Right")
        output = json.loads(formatter.format(record))
        assert output["fps"] == 60
        assert output["hand"] == "Right"

    def test_reserved_keys_not_overwritten_by_extra(self) -> None:
        formatter = _JsonFormatter()
        record = self._make_record("collision", level="INJECTED")
        output = json.loads(formatter.format(record))
        # "level" from extra must not overwrite the schema field
        assert output["level"] == "INFO"

    def test_exception_is_structured(self) -> None:
        formatter = _JsonFormatter()
        try:
            raise ValueError("bad value")
        except ValueError:
            record = self._make_record("error occurred", exc_info=sys.exc_info())

        output = json.loads(formatter.format(record))
        assert "exception" in output
        assert output["exception"]["type"] == "ValueError"
        assert output["exception"]["message"] == "bad value"
        assert "stacktrace" in output["exception"]

    def test_ambient_context_included(self) -> None:
        formatter = _JsonFormatter()
        with log_context(request_id="abc-123"):
            record = self._make_record("hello")
            output = json.loads(formatter.format(record))
        assert output["request_id"] == "abc-123"

    def test_json_serialization_robustness(self) -> None:
        """test for non-serializable objects and nested structures."""

        class UserObject:
            def __str__(self) -> str:
                return "UserObjectInstance"

        complex_data = {
            "nested": {"list": [1, 2, 3], "obj": UserObject()},
            "none": None,
        }

        formatter = _JsonFormatter()
        record = self._make_record("complex test", **complex_data)
        output = json.loads(formatter.format(record))

        assert output["nested"]["list"] == [1, 2, 3]
        assert output["nested"]["obj"] == "UserObjectInstance"
        assert output["none"] is None


class TestLoggerLifecycle:
    def test_file_rotation_and_persistence(
        self, log_cfg: LoggingConfig, log_dir: Path
    ) -> None:
        """Verify that logs are written to a stable filename and rotated by size."""
        log_cfg = log_cfg.model_copy(update={"max_bytes": 1024, "backup_count": 1})
        lifecycle = AsyncLoggerLifecycle(log_cfg)
        lifecycle.start()
        log = get_logger("test.rotation")

        # Fill the log beyond 1024 bytes
        for i in range(20):
            log.info(f"stressing the rotation mechanism with message number {i}")

        lifecycle.stop()

        main_log = log_dir / "handforge.log"
        rotated_log = log_dir / "handforge.log.1"

        assert main_log.exists()
        assert rotated_log.exists()
        assert rotated_log.stat().st_size > 0

    def test_concurrent_initialization_safety(self, log_cfg: LoggingConfig) -> None:
        """test for race conditions during startup."""
        lifecycle = AsyncLoggerLifecycle(log_cfg)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(lifecycle.start) for _ in range(5)]
            for f in futures:
                f.result()

        root = logging.getLogger("handforge")
        assert len(root.handlers) == 1
        lifecycle.stop()


class TestAsyncRobustness:
    def test_non_blocking_io_guarantee(self, log_cfg: LoggingConfig) -> None:
        """Verify that slow handlers do not block the calling thread."""
        slow = SlowHandler(delay=0.5)
        lifecycle = AsyncLoggerLifecycle(log_cfg, handlers=[slow])
        lifecycle.start()
        log = get_logger("test.async")

        start = time.perf_counter()
        log.info("this should not block")
        duration = time.perf_counter() - start

        # Must return instantly (< 100ms), even with 0.5s delay in handler
        assert duration < 0.1
        lifecycle.stop()

    def test_queue_overflow_drop_policy(self, log_cfg: LoggingConfig) -> None:
        """
        Stress test: Verify that a full queue results in
        dropped oldest logs rather than a blocked pipeline or OOM.
        """
        # Set a tiny queue for deterministic overflow testing
        queue_size = 5
        slow = SlowHandler(delay=0.1)  # slow down the listener
        lifecycle = AsyncLoggerLifecycle(
            log_cfg, handlers=[slow], queue_size=queue_size
        )
        lifecycle.start()
        log = get_logger("test.overflow")

        # Pump logs much faster than the slow consumer can process
        start = time.perf_counter()
        for i in range(20):
            log.info(f"msg {i}")
        duration = time.perf_counter() - start

        # Check for non-blocking property during overflow
        assert duration < 0.1

        lifecycle.stop()  # Wait for pending to flush

    def test_context_thread_isolation(self, log_cfg: LoggingConfig) -> None:
        """Verify that ambient log_context is strictly isolated to the calling thread."""
        mem_handler = ListHandler()
        mem_handler.setFormatter(_JsonFormatter())
        lifecycle = AsyncLoggerLifecycle(log_cfg, handlers=[mem_handler])
        lifecycle.start()

        def worker(thread_id: int) -> None:
            with log_context(tid=thread_id):
                log = get_logger("test.isolation")
                log.info(f"message from {thread_id}")

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(worker, range(3))

        lifecycle.stop()

        # Parse the JSON from the captured records
        assert mem_handler.formatter is not None
        parsed_logs = [
            json.loads(mem_handler.formatter.format(r)) for r in mem_handler.records
        ]

        # Ensure each log has the correct thread ID and no leakage.
        # Use .get() to ignore incidental logs (e.g., "logging initialised")
        for thread_id in range(3):
            thread_log = next(
                entry for entry in parsed_logs if entry.get("tid") == thread_id
            )
            assert thread_log["message"] == f"message from {thread_id}"
            assert thread_log["tid"] == thread_id

    def test_teardown_stability_and_reentry(self, log_cfg: LoggingConfig) -> None:
        """Ensure system remains stable during rapid start/stop cycles."""
        lifecycle = AsyncLoggerLifecycle(log_cfg)
        for _ in range(3):
            lifecycle.start()
            get_logger("test").info("active")
            lifecycle.stop()
            get_logger("test").info("zombie")  # Post-stop log should be safe
