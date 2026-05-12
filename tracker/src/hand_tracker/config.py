"""
Runtime configuration loader for HandForge.

Loading priority (highest → lowest):
  1. Environment variables (HANDFORGE__<SECTION>__<KEY>)
  2. .env file (python-dotenv)
  3. config.yaml (custom settings source)
  4. Pydantic model field defaults

The top-level AppConfig is a frozen, singleton-like object.
Call load_config() once at process startup and pass it through dependency injection.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from hand_tracker.types import Handedness

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_FILE = "config.yaml"


# ---------------------------------------------------------------------------
# Base Models
# ---------------------------------------------------------------------------


class HandForgeConfigModel(BaseModel):
    """
    Base model for all configuration sections.
    Enforces immutability and strict validation (no extra fields) across all levels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


class CameraConfig(HandForgeConfigModel):
    """Webcam capture parameters."""

    index: int = Field(default=0, ge=0, description="Webcam device index")
    width: int = Field(default=640, ge=160, le=3840)
    height: int = Field(default=480, ge=120, le=2160)
    fps: int = Field(default=30, ge=1, le=120)
    disable_auto_exposure: bool = Field(default=True)
    disable_auto_focus: bool = Field(default=True)
    buffer_size: int = Field(default=1, ge=1, le=10)
    jitter_threshold_multiplier: float = Field(default=1.5, ge=1.0, le=5.0)
    backend: Literal["AUTO", "ANY", "DSHOW", "MSMF", "V4L2", "AVFOUNDATION"] = Field(
        default="AUTO",
        description="OpenCV capture backend. Keep AUTO for OS-specific defaults.",
    )
    fourcc: str = Field(
        default="MJPG",
        min_length=4,
        max_length=4,
        description="Video codec FOURCC code (e.g., MJPG, YUYV, H264)",
    )
    mirror_input: bool = Field(
        default=True,
        description="Mirror the input frame horizontally (standard for webcams)",
    )


class MediaPipeConfig(HandForgeConfigModel):
    """MediaPipe Hands solution parameters."""

    model_path: str = Field(default="models/hand_landmarker.task")
    max_num_hands: int = Field(default=2, ge=1, le=2)
    # 0 = lite (faster), 1 = full (world_landmarks available - REQUIRED)
    # model_complexity 0=lite, 1=full. Completion requires world_landmarks.
    model_complexity: Literal[1] = Field(default=1)
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_presence_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    warmup_frame_count: int = Field(default=5, ge=0)

    @field_validator("model_complexity", mode="before")
    @classmethod
    def coerce_model_complexity(cls, v: Any) -> int:
        """Accept int-like values from various sources."""
        try:
            coerced = int(v)
            if coerced not in (0, 1):
                raise ValueError
            return coerced
        except (ValueError, TypeError):
            raise ValueError("model_complexity must be 0 or 1") from None


class TrackerConfig(HandForgeConfigModel):
    """Tracker runtime behaviour."""

    target_fps: int = Field(default=30, ge=1, le=120)
    primary_hand: Handedness = Field(default=Handedness.BOTH)
    fps_window_size: int = Field(default=30, ge=2, le=300)


class WebSocketConfig(HandForgeConfigModel):
    """WebSocket server/client settings."""

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8080, ge=1024, le=65535)


class OSCConfig(HandForgeConfigModel):
    """Open Sound Control (OSC) settings."""

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=9000, ge=1024, le=65535)
    address: str = Field(default="/handforge/landmarks")


class OutputConfig(HandForgeConfigModel):
    """Networking & Integration settings."""

    mode: Literal["websocket", "osc", "console", "none"] = Field(default="websocket")
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    osc: OSCConfig = Field(default_factory=OSCConfig)


class LoggingConfig(HandForgeConfigModel):
    """Structured logging parameters."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_dir: str = Field(default="logs")
    filename_prefix: str = Field(default="handforge")
    max_bytes: int = Field(default=10_485_760, ge=1024)
    backup_count: int = Field(default=5, ge=0)
    max_queue_size: int = Field(default=10000, ge=1, le=100000)
    console_enabled: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class AppConfig(BaseSettings):
    """
    Top-level application configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="HANDFORGE__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="forbid",
    )

    version: str = Field(default="1.0.0")
    camera: CameraConfig = Field(default_factory=CameraConfig)
    mediapipe: MediaPipeConfig = Field(default_factory=MediaPipeConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def validate_world_landmarks_requirement(self) -> AppConfig:
        """
        world_landmarks are only available when model_complexity == 1.
        Completion criterion requires world_position extraction,
        so model_complexity=0 is a hard error.
        """
        if self.mediapipe.model_complexity != 1:
            raise ValueError(
                "model_complexity must be 1 to enable world_landmarks. "
                "world_position extraction is required."
            )

        if (
            self.tracker.primary_hand == Handedness.BOTH
            and self.mediapipe.max_num_hands < 2
        ):
            raise ValueError(
                "max_num_hands must be at least 2 when primary_hand is 'Both'."
            )

        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Define the priority of configuration sources.
        Env Var > Dotenv (.env) > YAML > Default
        """
        return (
            env_settings,
            dotenv_settings,
            init_settings,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_config(config_path: str = DEFAULT_CONFIG_FILE) -> AppConfig:
    """
    Load and return the singleton AppConfig.

    Parameters
    ----------
    config_path:
        Path to the YAML config file. Defaults to ``config.yaml``.
        Can be overridden by passing a custom path.

    Returns
    -------
    AppConfig
        Frozen, validated configuration object.

    Raises
    ------
    pydantic.ValidationError
        If any value fails schema validation (range, type, cross-field rule).
    yaml.YAMLError
        If the YAML file contains syntactical errors (fail-fast rule).
    """
    path = Path(config_path)
    yaml_data: dict[str, Any] = {}

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
            if isinstance(parsed, dict):
                yaml_data = parsed
                log.info(f"configuration loaded from {path.absolute()}")
    else:
        log.warning(
            f"configuration file NOT found at {path.absolute()}. Using defaults."
        )

    # Instantiating AppConfig with yaml_data directly as kwargs injects them as
    # init_settings, neatly bypassing any model schema pollution.
    return AppConfig(**yaml_data)
