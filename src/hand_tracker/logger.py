"""
Structured JSON logger for HandForge.

Design decisions:
- Asynchronous Non-blocking Dispatch:
  Logging is offloaded to a background thread via QueueHandler/QueueListener.
  For a real-time computer vision pipeline (high FPS), this ensures disk/console
  I/O never stalls the main inference loop.
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
- Root-level Handler Attachment:
  All handlers are attached to the root "handforge" logger only. This
  prevents duplicate records often caused by child-logger propagation.

Usage:
    from hand_tracker.logger import get_logger, log_context
    log = get_logger(__name__)

    with log_context(track_id=42):
        log.info("tracker started", extra={"fps": 30})
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
from typing import Any

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

# Background listener for non-blocking logging
_listener: logging.handlers.QueueListener | None = None
_log_queue: queue.Queue[Any] | None = None
_handlers: list[logging.Handler] = []

# Internal state for idempotency and thread-safety
_initialised = False
_init_lock = threading.Lock()

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

        # 1. Merge caller-supplied extra fields, skipping reserved/internal keys
        for key, value in record.__dict__.items():
            if key in self._RECORD_ATTRS or key in self._RESERVED:
                continue
            if key.startswith("_"):
                continue
            payload[key] = value

        # 2. Merge ambient context (ContextVar)
        context = _LOG_CONTEXT.get()
        if context:
            for key, value in context.items():
                if key not in self._RESERVED:
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
# Setup
# ---------------------------------------------------------------------------


def setup_logging(
    cfg: LoggingConfig, handlers: list[logging.Handler] | None = None
) -> None:
    """
    Configure the root "handforge" logger.

    Must be called exactly once at process startup, before any get_logger()
    calls. Subsequent calls are no-ops (idempotent guard via _initialised).

    Parameters
    ----------
    cfg:
        LoggingConfig section from AppConfig.
    handlers:
        Optional list of handlers to inject. If provided, the internal
        file and console handlers are not created. This is primarily
        for dependency injection during testing.
    """
    global _initialised
    with _init_lock:
        if _initialised:
            return

        level = _LEVEL_MAP.get(cfg.level.upper(), logging.INFO)

        root = logging.getLogger(ROOT_LOGGER_NAME)
        root.setLevel(level)
        root.propagate = False  # do not bleed into Python root logger

        # ------------------------------------------------------------------
        # Handlers Configuration
        # ------------------------------------------------------------------
        global _log_queue, _listener, _handlers

        if handlers is not None:
            # Dependency Injection path (usually for testing)
            _handlers = handlers
            log_file_val = None
            injected_list = [type(h).__name__ for h in handlers]
        else:
            # Standard Production path
            log_dir = Path(cfg.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Stable filename: handforge.log (required for reliable rotation)
            log_path = log_dir / f"{cfg.filename_prefix}.log"
            log_file_val = str(log_path)
            injected_list = []

            internal_handlers: list[logging.Handler] = []

            # File handler — rotating JSON
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=cfg.max_bytes,
                backupCount=cfg.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(_JsonFormatter())
            internal_handlers.append(file_handler)

            # Console handler — Rich (human-readable)
            if cfg.console_enabled:
                rich_handler = _make_rich_handler()
                internal_handlers.append(rich_handler)

            _handlers = internal_handlers

        # ------------------------------------------------------------------
        # Asynchronous Dispatch (QueueHandler -> QueueListener)
        # ------------------------------------------------------------------
        _log_queue = queue.Queue(-1)  # unbounded
        queue_handler = logging.handlers.QueueHandler(_log_queue)

        # Listener runs in a separate thread and pulls from the queue
        _listener = logging.handlers.QueueListener(
            _log_queue, *_handlers, respect_handler_level=True
        )
        _listener.start()

        root.addHandler(queue_handler)
        _initialised = True

        # Emit the first record through the now-configured logger
        _startup_log = logging.getLogger(f"{ROOT_LOGGER_NAME}.logger")
        _startup_log.info(
            "logging initialised",
            extra={
                "log_file": log_file_val,
                "injected": injected_list,
                "level": cfg.level,
                "console": cfg.console_enabled,
                "mode": "injected" if handlers is not None else "standard",
            },
        )


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
        The "handforge." prefix is prepended automatically if absent,
        so loggers created from outside the package still route correctly.

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

    Example:
        with log_context(track_id=42):
            log.info("processing frame")  # JSON will include "track_id": 42
    """
    context = _LOG_CONTEXT.get() or {}
    token = _LOG_CONTEXT.set({**context, **kwargs})
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def shutdown_logging() -> None:
    """
    Shut down the background listener and close all handlers.

    Must be called at process exit to ensure final logs are flushed to disk.
    """
    global _initialised, _listener, _log_queue, _handlers
    root = logging.getLogger(ROOT_LOGGER_NAME)

    # 1. Stop the listener thread
    if _listener:
        _listener.stop()
        _listener = None

    # 2. Close the internal handlers (this is critical for Windows file locks)
    for handler in _handlers:
        handler.flush()
        handler.close()
    _handlers = []

    # 3. Remove the QueueHandler from the root logger
    for handler in root.handlers[:]:
        handler.flush()
        handler.close()
        root.removeHandler(handler)

    _log_queue = None
    _initialised = False
