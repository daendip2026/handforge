"""
Structured JSON logger for HandForge.

Design decisions:
- Asynchronous Non-blocking Dispatch (Zero-Latency Queue):
  Logging is offloaded to a background thread via a bounded queue (maxsize=10000).
  If disk/console I/O stalls and the queue fills up, the custom ZeroLatencyQueueHandler
  aggressively drops the oldest logs. For a real-time computer vision pipeline,
  this ensures that the main inference loop never OOMs or stutters due to logging overhead.
- Machine-Parseable JSON:
  Every log record is a single-line JSON. This allows log aggregators (Datadog,
  Loki, GCP Logging) to index the data without post-processing (e.g. regex),
  while remaining human-readable in dev-mode via Rich.
- Stable JSON Schema:
  Includes {timestamp_us, level, module, message} by default. Exceptions are
  structured as sub-objects (type, message, stacktrace) for better observability.
- Ambient Context (ContextVar):
  Supports injecting 'ambient' metadata (e.g. track_id, session_id) into all
  logs within a scope via `log_context`, avoiding manual passing of variables.
- Size-based Rotation:
  Uses RotatingFileHandler instead of TimedRotatingFileHandler. For desktop
  processes with variable uptime, size-based rotation is more predictable
  and prevents log accumulation. Stable filenames ensure persistence across restarts.
- Encapsulated Lifecycle:
  All state logic and thread management is cleanly encapsulated inside the
  `AsyncLoggerLifecycle` class, replacing fragile global states and module-level locks,
  ensuring safe dependency injection and modular teardown.

Usage:
    from hand_tracker.logger import AsyncLoggerLifecycle, get_logger, log_context

    # 1. At application startup
    lifecycle = AsyncLoggerLifecycle(cfg)
    lifecycle.start()

    # 2. Anywhere in your application
    log = get_logger(__name__)
    with log_context(track_id=42):
        log.info("tracker started", extra={"fps": 30})

    # 3. At application exit
    lifecycle.stop()
"""

from __future__ import annotations

import contextlib
import json
import logging
import logging.handlers
import queue
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from hand_tracker.config import LoggingConfig

# ---------------------------------------------------------------------------
# Constants & Global State
# ---------------------------------------------------------------------------

ROOT_LOGGER_NAME = "handforge"

_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

# Ambient context for all logs in the current execution flow
_LOG_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "log_context", default=None
)

# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """
    Emit each log record as a single-line JSON object.

    Schema (always present):
        timestamp_us  int     Unix timestamp in microseconds
        level         str     DEBUG | INFO | WARNING | ERROR
        module        str     Logger name (e.g. "handforge.mediapipe_tracker")
        message       str     Formatted log message

    Optional (present when passed via extra={}):
        Any key-value pair passed to extra={} is merged at the top level.
        Keys that collide with the schema above are silently dropped to
        preserve schema stability.
    """

    _RESERVED: frozenset[str] = frozenset(
        {"timestamp_us", "level", "module", "message"}
    )

    # Standard LogRecord attributes that must not leak into the JSON output.
    # Full list: https://docs.python.org/3/library/logging.html#logrecord-attributes
    _RECORD_ATTRS: frozenset[str] = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp_us": int(record.created * 1_000_000),
            "level": record.levelname,
            "module": record.name,
            "message": record.message,
        }

        # 1. Merge all attributes from the record, skipping reserved/internal keys.
        # This includes fields captured from log_context by the filter in the producer thread.
        for key, value in record.__dict__.items():
            if key in self._RECORD_ATTRS or key in self._RESERVED:
                continue
            if key.startswith("_"):
                continue
            payload[key] = value

        # 2. Merge ambient context (ContextVar) as a fallback.
        # This ensures direct usage and same-thread testing still see the context.
        context = _LOG_CONTEXT.get()
        if context:
            for key, value in context.items():
                if key not in self._RESERVED and key not in payload:
                    payload[key] = value

        # 3. Handle exceptions with structure
        if record.exc_info:
            ex_type, ex_val, _ = record.exc_info
            payload["exception"] = {
                "type": ex_type.__name__ if ex_type else "Unknown",
                "message": str(ex_val),
                "stacktrace": self.formatException(record.exc_info),
            }

        return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Rich console formatter (development only)
# ---------------------------------------------------------------------------

_RICH_THEME = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "green",
        "logging.level.warning": "yellow",
        "logging.level.error": "bold red",
    }
)

_RICH_CONSOLE = Console(theme=_RICH_THEME, stderr=False)


def _make_rich_handler() -> RichHandler:
    handler = RichHandler(
        console=_RICH_CONSOLE,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        log_time_format="[%H:%M:%S.%f]",
    )
    # Rich handles its own formatting; suppress Formatter interference
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


# ---------------------------------------------------------------------------
# Custom Queue Handler
# ---------------------------------------------------------------------------


class ZeroLatencyQueueHandler(logging.handlers.QueueHandler):
    """
    Custom QueueHandler that drops the oldest logs when the queue is full,
    ensuring the real-time AI pipeline never blocks and unbounded queues
    don't cause Out Of Memory (OOM) crashes.
    """

    def __init__(self, q: queue.Queue[Any]) -> None:
        super().__init__(q)
        self._queue: queue.Queue[Any] = q

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """
        Capture the caller's ContextVar state and inject it into the record
        BEFORE it leaves the thread. This is critical for async context stability.
        """
        context = _LOG_CONTEXT.get()
        if context:
            for key, value in context.items():
                if not hasattr(record, key):
                    setattr(record, key, value)
        return cast(logging.LogRecord, super().prepare(record))

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Active drop policy: drop the oldest to make space for reality
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()

            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(record)


# ---------------------------------------------------------------------------
# Lifecycle Management
# ---------------------------------------------------------------------------


class AsyncLoggerLifecycle:
    """
    Encapsulates the lifecycle of the asynchronous logging background thread.
    Replaces global state variables to ensure safe, modular teardown.
    """

    def __init__(
        self,
        cfg: LoggingConfig,
        handlers: list[logging.Handler] | None = None,
        queue_size: int | None = None,
    ) -> None:
        self.cfg = cfg
        self._injected_handlers = handlers
        self._queue_size = queue_size or cfg.max_queue_size
        self._listener: logging.handlers.QueueListener | None = None
        self._queue_handler: ZeroLatencyQueueHandler | None = None
        self._handlers: list[logging.Handler] = []
        self._active = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Initialize the background thread and attach handlers."""
        with self._lock:
            if self._active:
                return

            level = _LEVEL_MAP.get(self.cfg.level.upper(), logging.INFO)
            root = logging.getLogger(ROOT_LOGGER_NAME)
            root.setLevel(level)
            root.propagate = False

            if self._injected_handlers is not None:
                self._handlers = list(self._injected_handlers)
                log_file_val = None
                injected_list = [type(h).__name__ for h in self._handlers]
            else:
                log_dir = Path(self.cfg.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{self.cfg.filename_prefix}.log"
                log_file_val = str(log_path)
                injected_list = []

                file_handler = logging.handlers.RotatingFileHandler(
                    filename=log_path,
                    maxBytes=self.cfg.max_bytes,
                    backupCount=self.cfg.backup_count,
                    encoding="utf-8",
                )
                file_handler.setFormatter(_JsonFormatter())
                self._handlers.append(file_handler)

                if self.cfg.console_enabled:
                    self._handlers.append(_make_rich_handler())

            # Bounded queue is critical for OOM defense!
            log_queue: queue.Queue[Any] = queue.Queue(maxsize=self._queue_size)
            self._queue_handler = ZeroLatencyQueueHandler(log_queue)

            self._listener = logging.handlers.QueueListener(
                log_queue, *self._handlers, respect_handler_level=True
            )
            self._listener.start()
            root.addHandler(self._queue_handler)
            self._active = True

            _startup_log = logging.getLogger(f"{ROOT_LOGGER_NAME}.logger")
            _startup_log.info(
                "logging initialised",
                extra={
                    "log_file": log_file_val,
                    "injected": injected_list,
                    "level": self.cfg.level,
                    "console": self.cfg.console_enabled,
                    "mode": "injected"
                    if self._injected_handlers is not None
                    else "standard",
                },
            )

    def stop(self) -> None:
        """Stop the background formatting thread and close handlers."""
        with self._lock:
            if not self._active:
                return

            root = logging.getLogger(ROOT_LOGGER_NAME)

            if self._listener:
                # If the queue is full, QueueListener.stop() will crash while
                # trying to enqueue the sentinel. We force a space if needed.
                # Accessing internal queue from listener: self._listener.queue
                q = getattr(self._listener, "queue", None)
                if q and hasattr(q, "full") and q.full():
                    with contextlib.suppress(Exception):
                        q.get_nowait()

                self._listener.stop()
                self._listener = None

            for handler in self._handlers:
                handler.flush()
                handler.close()
            self._handlers.clear()

            if self._queue_handler:
                self._queue_handler.flush()
                self._queue_handler.close()
                root.removeHandler(self._queue_handler)
                self._queue_handler = None

            self._active = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the "handforge" root.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
        The "handforge." prefix is prepended automatically if absent.

    Returns
    -------
    logging.Logger
        Child logger. Do not attach handlers to it directly.
    """
    if not name.startswith(f"{ROOT_LOGGER_NAME}."):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


@contextlib.contextmanager
def log_context(**kwargs: Any) -> Iterator[None]:
    """
    Context manager to inject ambient fields into all logs within this scope.
    """
    context = _LOG_CONTEXT.get() or {}
    token = _LOG_CONTEXT.set({**context, **kwargs})
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)
