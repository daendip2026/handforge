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


class MediaPipeConfig(HandForgeConfigModel):
    """MediaPipe Hands solution parameters."""

    max_num_hands: int = Field(default=1, ge=1, le=2)
    # 0 = lite (faster, less accurate), 1 = full (world_landmarks available)
    model_complexity: Literal[0, 1] = Field(default=1)
    min_detection_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    static_image_mode: bool = Field(default=False)

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
    primary_hand: Literal["Right", "Left", "Both"] = Field(default="Right")


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
    console_enabled: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Custom Settings Source (YAML)
# ---------------------------------------------------------------------------


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A settings source that loads data from a YAML file.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_path: str | Path | None = None,
    ):
        """
        Initialize the YAML settings source.

        Args:
            settings_cls: The settings class this source is used for.
            yaml_path: Explicit path provided at runtime. Stored separately
                      to bypass AppConfig's extra="forbid" constraint.
        """
        super().__init__(settings_cls)
        self.yaml_path = yaml_path

    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:  # pragma: no cover
        # This method is required by the interface but not used for full dict loads
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        # Priority: explicit passed path > filename default
        # Cast to str to satisfy Path() type requirements in mypy
        config_path = str(self.yaml_path or DEFAULT_CONFIG_FILE)
        path = Path(config_path)
        if not path.exists():
            return {}

        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            # you might want to log this or raise a specific error
            return {}


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

    # Orchestration field: allowed in constructor but excluded from model output
    config_path: str | None = Field(default=None, exclude=True)

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
        # Extract the config_path from init_settings (constructor arguments)
        init_data = init_settings()
        extracted_path = init_data.get("config_path")

        # If no explicit path is passed, fetch from field default
        if extracted_path is None:
            field_info = settings_cls.model_fields.get("config_path")
            if field_info:
                extracted_path = field_info.get_default()

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_path=extracted_path),
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
    """
    # Simply instantiating AppConfig triggers the Pydantic loading pipeline.
    return AppConfig(config_path=config_path)
