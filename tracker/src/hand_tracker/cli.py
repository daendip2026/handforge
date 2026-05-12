"""
HandForge — real-time hand tracker command line interface.
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from rich import box
from rich.console import Console
from rich.table import Table

from hand_tracker.capture import CaptureError, WebcamCapture
from hand_tracker.config import load_config
from hand_tracker.landmark_processor import (
    LandmarkProcessor,
    full_landmark_dump,
)
from hand_tracker.logger import AsyncLoggerLifecycle, get_logger
from hand_tracker.mediapipe_tracker import MediaPipeTracker
from hand_tracker.utils import console_summary
from hand_tracker.viewer import DebugViewer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Print console summary every N *processed* frames (hand detected).
SUMMARY_EVERY_N_FRAMES: Final[int] = 30  # Increased from 6 to prevent terminal flood

# Latency threshold for log warning emission.
LATENCY_WARN_MS: Final[float] = 50.0

_rich_console = Console()

# ---------------------------------------------------------------------------
# Shutdown coordination
# ---------------------------------------------------------------------------

_shutdown_event = threading.Event()


def _install_signal_handlers() -> None:
    """
    Register OS signal handlers for graceful shutdown.

    SIGINT  (Ctrl+C)    — supported on all platforms.
    SIGTERM             — registered on non-Windows only.
    SIGBREAK (Ctrl+Brk) — Windows-specific; registered when available.
    """

    def _request_shutdown(signum: int, _frame: Any) -> None:
        _shutdown_event.set()

    signal.signal(signal.SIGINT, _request_shutdown)

    if platform.system() != "Windows":
        signal.signal(signal.SIGTERM, _request_shutdown)
    else:
        # SIGBREAK is Ctrl+Break on Windows cmd / PowerShell
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _request_shutdown)


# ---------------------------------------------------------------------------
# Pipeline statistics
# ---------------------------------------------------------------------------


@dataclass
class _PipelineStats:
    """
    Accumulator for end-of-run performance metrics.
    """

    total_captured: int = 0
    total_processed: int = 0  # frames where a hand was detected
    latency_ms_min: float = float("inf")
    latency_ms_max: float = 0.0
    _latency_sum: float = field(default=0.0, repr=False)
    _latency_count: int = field(default=0, repr=False)
    _start_perf: float = field(default_factory=time.perf_counter, repr=False)

    def record_latency(self, latency_ms: float) -> None:
        if latency_ms < self.latency_ms_min:
            self.latency_ms_min = latency_ms
        if latency_ms > self.latency_ms_max:
            self.latency_ms_max = latency_ms
        self._latency_sum += latency_ms
        self._latency_count += 1

    @property
    def latency_ms_mean(self) -> float:
        if self._latency_count == 0:
            return float("nan")
        return self._latency_sum / self._latency_count

    @property
    def detection_rate_pct(self) -> float:
        if self.total_captured == 0:
            return 0.0
        return (self.total_processed / self.total_captured) * 100.0

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start_perf

    @property
    def effective_fps(self) -> float:
        elapsed = self.elapsed_s
        if elapsed <= 0.0:
            return 0.0
        return self.total_captured / elapsed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="handforge-tracker",
        description="HandForge  — real-time hand tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config.yaml (relative to cwd or absolute)",
    )
    parser.add_argument(
        "--full-dump",
        action="store_true",
        help="Print all 21 landmarks per summary frame (verbose)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override logging.level from config.yaml",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Enable real-time OpenCV debug visualization",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Exit summary
# ---------------------------------------------------------------------------


def _print_exit_summary(stats: _PipelineStats) -> None:
    """Render a Rich table summarising the pipeline run to stdout."""
    _rich_console.print()

    table = Table(
        title="[bold cyan]HandForge — Run Summary[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right")

    def _fmt_float(v: float, unit: str = "", decimals: int = 1) -> str:
        if v != v:  # NaN check
            return "—"
        return f"{v:.{decimals}f}{unit}"

    elapsed = stats.elapsed_s
    latency_ok = (
        stats.latency_ms_mean <= LATENCY_WARN_MS
        if stats.latency_ms_mean == stats.latency_ms_mean
        else True
    )
    latency_style = "" if latency_ok else "[yellow]"
    latency_end = "" if latency_ok else "[/yellow]"

    table.add_row("Elapsed time", f"{elapsed:.1f} s")
    table.add_row("Frames captured", str(stats.total_captured))
    table.add_row("Frames processed", str(stats.total_processed))
    table.add_row("Detection rate", _fmt_float(stats.detection_rate_pct, "%"))
    table.add_row("Effective FPS", _fmt_float(stats.effective_fps, " fps"))
    table.add_row(
        "Latency — mean",
        f"{latency_style}{_fmt_float(stats.latency_ms_mean, ' ms')}{latency_end}",
    )
    table.add_row("Latency — min", _fmt_float(stats.latency_ms_min, " ms"))
    table.add_row("Latency — max", _fmt_float(stats.latency_ms_max, " ms"))
    table.add_row(
        "Latency target (< 50 ms)",
        "[green]PASS[/green]" if latency_ok else "[yellow]MISS[/yellow]",
    )

    _rich_console.print(table)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Entry point.

    Returns
    -------
    int
        0 on clean exit, 1 on unrecoverable error.
    """
    args = _parse_args()

    # Apply CLI overrides before config is loaded and cached
    if args.log_level is not None:
        os.environ["HANDFORGE__LOGGING__LEVEL"] = args.log_level

    # Validate config path early for a clear error message
    config_path = Path(args.config)
    if not config_path.exists():
        _rich_console.print(
            f"[bold red]Error:[/bold red] config file not found: {config_path.resolve()}"
        )
        return 1

    try:
        cfg = load_config(str(config_path))
    except Exception as exc:  # pydantic.ValidationError or YAML parse error
        _rich_console.print(f"[bold red]Config error:[/bold red] {exc}")
        return 1

    # Initialize async logging lifecycle
    log_lifecycle = AsyncLoggerLifecycle(cfg.logging)
    log_lifecycle.start()
    log = get_logger(__name__)

    viewer = DebugViewer() if args.view else None

    _install_signal_handlers()

    # Log to file/logger
    log.info("HandForge tracker starting", extra={"config": str(config_path.resolve())})

    # Print a beautiful config table for the user
    from rich.table import Table

    config_table = Table(title="HandForge — Active Configuration", box=None)
    config_table.add_column("Section", style="cyan")
    config_table.add_column("Key", style="magenta")
    config_table.add_column("Value", style="green")

    config_table.add_row(
        "Camera", "Resolution", f"{cfg.camera.width}x{cfg.camera.height}"
    )
    config_table.add_row("Camera", "Target FPS", str(cfg.camera.fps))
    config_table.add_row(
        "MediaPipe", "Detection Conf", f"{cfg.mediapipe.min_detection_confidence:.2f}"
    )
    config_table.add_row(
        "MediaPipe", "Tracking Conf", f"{cfg.mediapipe.min_tracking_confidence:.2f}"
    )
    config_table.add_row(
        "MediaPipe", "Presence Conf", f"{cfg.mediapipe.min_presence_confidence:.2f}"
    )
    config_table.add_row("MediaPipe", "Model Path", cfg.mediapipe.model_path)

    _rich_console.print(config_table)
    _rich_console.print(f"[dim]Log file: {cfg.logging.log_dir}[/dim]\n")

    stats = _PipelineStats(_start_perf=time.perf_counter())
    exit_code = 0

    try:
        with WebcamCapture(cfg.camera) as capture:
            # Log actual device capabilities after open
            if capture.device_info is not None:
                di = capture.device_info
                log.info(
                    "camera ready",
                    extra={
                        "actual_resolution": f"{di.actual_width}x{di.actual_height}",
                        "actual_fps": di.actual_fps,
                        "backend": di.backend,
                    },
                )
                if di.actual_fps < cfg.tracker.target_fps:
                    log.warning(
                        "device FPS below target",
                        extra={
                            "actual_fps": di.actual_fps,
                            "target_fps": cfg.tracker.target_fps,
                        },
                    )

            with MediaPipeTracker(cfg.mediapipe, cfg.tracker, cfg.camera) as tracker:
                processor = LandmarkProcessor(window_size=cfg.tracker.fps_window_size)
                _rich_console.print(
                    "\n[bold green]Tracker running.[/bold green] "
                    "Press [bold]Ctrl+C[/bold] to stop.\n"
                )

                for frame in capture:
                    if _shutdown_event.is_set():
                        log.info("shutdown signal received")
                        break

                    t_loop_start = time.perf_counter()
                    stats.total_captured += 1

                    result = tracker.process(frame)

                    if result is None:
                        continue

                    processed = processor.update(result)
                    if processed.hands:
                        stats.total_processed += 1

                    latency_ms = (time.perf_counter() - t_loop_start) * 1_000.0
                    stats.record_latency(latency_ms)

                    if latency_ms > LATENCY_WARN_MS:
                        log.warning(
                            "pipeline latency exceeds target",
                            extra={
                                "latency_ms": round(latency_ms, 2),
                                "target_ms": LATENCY_WARN_MS,
                                "frame_index": processed.frame_index,
                            },
                        )

                    if stats.total_processed % SUMMARY_EVERY_N_FRAMES == 1:
                        if args.full_dump:
                            print(full_landmark_dump(processed), flush=True)
                        else:
                            print(console_summary(processed), flush=True)

                    if viewer and not viewer.render(frame, result):
                        log.info("viewer window closed or exit requested")
                        break

    except CaptureError as exc:
        log.error("capture device error", extra={"error": str(exc)})
        _rich_console.print(f"\n[bold red]Capture error:[/bold red] {exc}")
        exit_code = 1

    except KeyboardInterrupt:
        pass

    finally:
        _print_exit_summary(stats)
        log.info(
            "tracker stopped",
            extra={
                "total_captured": stats.total_captured,
                "total_processed": stats.total_processed,
                "elapsed_s": round(stats.elapsed_s, 2),
                "effective_fps": round(stats.effective_fps, 2),
                "latency_ms_mean": round(stats.latency_ms_mean, 2)
                if stats.latency_ms_mean == stats.latency_ms_mean
                else None,
            },
        )
        if viewer:
            viewer.close()
        log_lifecycle.stop()

    return exit_code


def _get_version() -> str:
    try:
        from hand_tracker import __version__

        return __version__
    except ImportError:
        return "0.0.0+unknown"
