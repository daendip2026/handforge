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
from typing import Literal

import pytest  # type: ignore

import hand_tracker.logger as logger_module
from hand_tracker.config import LoggingConfig
from hand_tracker.logger import (
    _JsonFormatter,
    get_logger,
    log_context,
    setup_logging,
    shutdown_logging,
)


@pytest.fixture(autouse=True)  # type: ignore
def reset_logger() -> Iterator[None]:
    """Ensure logger state is clean before and after every test."""
    shutdown_logging()
    logging.getLogger("handforge").handlers.clear()
    logger_module._initialised = False
    yield
    shutdown_logging()
    logging.getLogger("handforge").handlers.clear()
    logger_module._initialised = False


def _make_cfg(
    log_dir: str, level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
) -> LoggingConfig:
    return LoggingConfig(
        level=level,
        log_dir=log_dir,
        console_enabled=False,
    )


class TestJsonFormatter:
    def _make_record(self, msg: str = "test", **extra: object) -> logging.LogRecord:
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
        # Unix microseconds in 2024+ are 16 digits
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

    def test_output_is_single_line(self) -> None:
        formatter = _JsonFormatter()
        record = self._make_record("newline\ntest")
        output = formatter.format(record)
        assert "\n" not in output

    def test_json_serialization_robustness(self) -> None:
        class UserObject:
            def __str__(self) -> str:
                return "UserObjectInstance"

        formatter = _JsonFormatter()
        record = self._make_record("obj test", data=UserObject())
        output = json.loads(formatter.format(record))
        # Should not crash and should use str() thanks to default=str
        assert output["data"] == "UserObjectInstance"


class TestSetupLogging:
    def test_log_file_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            setup_logging(cfg)
            log_files = list(Path(tmp).glob("*.log"))
            assert len(log_files) == 1
            shutdown_logging()

    def test_log_file_contains_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            setup_logging(cfg)
            log = get_logger("test.json_check")
            log.info("json validity check", extra={"key": "value"})
            shutdown_logging()

            log_file = next(Path(tmp).glob("*.log"))
            lines = log_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) >= 1
            expected_keys = {"timestamp_us", "level", "module", "message"}
            for line in lines:
                parsed = json.loads(line)
                assert expected_keys <= parsed.keys()

    def test_stable_filename_no_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            setup_logging(cfg)
            shutdown_logging()
            # Filename should be exactly handforge.log
            assert (Path(tmp) / "handforge.log").exists()
            assert not any("_20" in f.name for f in Path(tmp).glob("*.log"))

    def test_idempotent_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            setup_logging(cfg)
            setup_logging(cfg)  # second call must be a no-op
            root = logging.getLogger("handforge")
            # Only one file handler should exist
            assert len(root.handlers) == 1
            shutdown_logging()

    def test_extra_fields_written_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            setup_logging(cfg)
            log = get_logger("test.extra")
            log.info("fps measurement", extra={"fps": 58.3, "latency_ms": 12.4})
            shutdown_logging()

            log_file = next(Path(tmp).glob("*.log"))
            records = [
                json.loads(line)
                for line in log_file.read_text(encoding="utf-8").strip().splitlines()
            ]
            fps_record = next(
                r for r in records if r.get("message") == "fps measurement"
            )
            assert fps_record["fps"] == pytest.approx(58.3)
            assert fps_record["latency_ms"] == pytest.approx(12.4)

    def test_unicode_and_emoji_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            setup_logging(cfg)
            log = get_logger("test.unicode")
            msg = "안녕하세요 🚀 HandForge!"
            log.info(msg)
            shutdown_logging()

            log_file = next(Path(tmp).glob("*.log"))
            content = log_file.read_text(encoding="utf-8")
            assert msg in content

    def test_rich_handler_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = LoggingConfig(
                level="INFO",
                log_dir=tmp,
                console_enabled=True,
            )
            # Verify internal list of handlers reflects the config
            setup_logging(cfg)
            from hand_tracker.logger import _handlers

            # 1 FileHandler + 1 RichHandler (inside the queue system)
            assert len(_handlers) == 2
            assert any(type(h).__name__ == "RichHandler" for h in _handlers)
            shutdown_logging()

    def test_post_shutdown_stability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            setup_logging(cfg)
            log = get_logger("test.shutdown")
            shutdown_logging()

            # Calling after shutdown should not raise exception (idempotent/safe)
            log.info("zombie log")

    def test_thread_safe_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)

            # Attempt to initialize from 10 threads at once
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(setup_logging, cfg) for _ in range(10)]
                for f in futures:
                    f.result()  # ensure all threads finish

            root = logging.getLogger("handforge")
            # Only one QueueHandler should be attached
            assert len(root.handlers) == 1
            shutdown_logging()


class TestGetLogger:
    def test_prefix_auto_prepended(self) -> None:
        log = get_logger("my_module")
        assert log.name == "handforge.my_module"

    def test_prefix_not_doubled(self) -> None:
        log = get_logger("handforge.my_module")
        assert log.name == "handforge.my_module"


class TestLogRotation:
    def test_rotation_triggers_on_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Minimum allowed limit (1024 bytes) to comply with Pydantic validation
            cfg = LoggingConfig(
                level="INFO",
                log_dir=tmp,
                max_bytes=1024,
                backup_count=1,
                console_enabled=False,
            )
            setup_logging(cfg)
            log = get_logger("test.rotation")

            # Write enough logs to exceed 1024 bytes
            # Each record is ~150-200 bytes, so 10 logs will definitely trigger it
            for i in range(10):
                log.info(f"rotation trigger message number {i}")
            shutdown_logging()

            log_dir = Path(tmp)
            main_log = log_dir / "handforge.log"
            rotated_log = log_dir / "handforge.log.1"

            assert main_log.exists()
            assert rotated_log.exists()
            # Verify rotated contents are still valid JSON
            content = rotated_log.read_text(encoding="utf-8")
            json.loads(content.splitlines()[0])


class SlowHandler(logging.NullHandler):
    """Handler that simulates slow I/O (e.g. network or slow disk)."""

    def emit(self, record: logging.LogRecord) -> None:
        time.sleep(0.5)


class TestAsyncLogging:
    def test_logging_is_non_blocking_on_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            slow_handler = SlowHandler()

            # Dependency Injection: Inject the slow handler directly.
            # If the logger is truly async, log.info() should return instantly
            # because the QueueHandler only puts the record into a memory queue.
            setup_logging(cfg, handlers=[slow_handler])
            log = get_logger("test.async")

            start_time = time.perf_counter()
            # If it's blocking, this will take 0.5s.
            # If it's non-blocking (async), it will take < 0.01s.
            log.info("non-blocking check")
            duration = time.perf_counter() - start_time

            # Non-blocking should be nearly instantaneous (< 10ms typically)
            assert duration < 0.1  # Allowing margin for CI overhead
            shutdown_logging()
