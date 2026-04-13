"""Tests for config loading and validation."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hand_tracker.config import (
    AppConfig,
    CameraConfig,
    LoggingConfig,
    MediaPipeConfig,
    TrackerConfig,
    load_config,
)


@pytest.fixture(autouse=True)
def clean_config_cache() -> Iterator[None]:
    """Ensure load_config cache is cleared before and after each test."""
    load_config.cache_clear()
    yield
    load_config.cache_clear()


@pytest.fixture(autouse=True)
def clean_env() -> Iterator[None]:
    """Provide a clean environment for testing env var overrides."""
    old_env = os.environ.copy()
    # Remove any existing HANDFORGE__ variables
    for key in list(os.environ.keys()):
        if key.startswith("HANDFORGE__"):
            del os.environ[key]
    yield
    os.environ.clear()
    os.environ.update(old_env)


class TestCameraConfig:
    def test_defaults(self) -> None:
        cfg = CameraConfig()
        assert cfg.index == 0
        assert cfg.width == 640
        assert cfg.fps == 30
        assert cfg.disable_auto_exposure is True
        assert cfg.disable_auto_focus is True
        assert cfg.buffer_size == 1
        assert cfg.jitter_threshold_multiplier == 1.5
        assert cfg.backend == "AUTO"
        assert cfg.fourcc == "MJPG"

    def test_invalid_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CameraConfig(index=-1)

    def test_buffer_size_boundaries(self) -> None:
        # Valid range [1, 10]
        assert CameraConfig(buffer_size=1).buffer_size == 1
        assert CameraConfig(buffer_size=10).buffer_size == 10
        with pytest.raises(ValidationError):
            CameraConfig(buffer_size=0)
        with pytest.raises(ValidationError):
            CameraConfig(buffer_size=11)

    def test_jitter_multiplier_boundaries(self) -> None:
        # Valid range [1.0, 5.0]
        assert (
            CameraConfig(jitter_threshold_multiplier=1.0).jitter_threshold_multiplier
            == 1.0
        )
        assert (
            CameraConfig(jitter_threshold_multiplier=5.0).jitter_threshold_multiplier
            == 5.0
        )
        with pytest.raises(ValidationError, match="greater_than_equal"):
            CameraConfig(jitter_threshold_multiplier=0.9)
        with pytest.raises(ValidationError, match="less_than_equal"):
            CameraConfig(jitter_threshold_multiplier=5.1)

    def test_backend_validation(self) -> None:
        # Valid
        assert CameraConfig(backend="MSMF").backend == "MSMF"
        assert CameraConfig(backend="AUTO").backend == "AUTO"

        # Invalid
        with pytest.raises(ValidationError):
            CameraConfig(backend="INVALID_BACKEND")  # type: ignore

    def test_fourcc_validation(self) -> None:
        # Valid
        assert CameraConfig(fourcc="MJPG").fourcc == "MJPG"
        assert CameraConfig(fourcc="YUY2").fourcc == "YUY2"

        # Invalid length
        with pytest.raises(ValidationError):
            CameraConfig(fourcc="JPG")
        with pytest.raises(ValidationError):
            CameraConfig(fourcc="MJPEG")


class TestMediaPipeConfig:
    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            MediaPipeConfig(min_detection_confidence=1.5)


class TestTrackerConfig:
    def test_primary_hand_options(self) -> None:
        # Should accept Right, Left, Both option
        assert TrackerConfig(primary_hand="Right").primary_hand == "Right"
        assert TrackerConfig(primary_hand="Left").primary_hand == "Left"
        assert TrackerConfig(primary_hand="Both").primary_hand == "Both"

        with pytest.raises(ValidationError):
            TrackerConfig(primary_hand="None")  # type: ignore

    def test_invalid_fps(self) -> None:
        with pytest.raises(ValidationError):
            TrackerConfig(target_fps=0)
        with pytest.raises(ValidationError):
            TrackerConfig(target_fps=200)


class TestNetworkConfig:
    """Critical tests for network availability (Port ranges)."""

    def test_websocket_invalid_port(self) -> None:
        from hand_tracker.config import WebSocketConfig

        with pytest.raises(ValidationError):
            WebSocketConfig(port=80)  # Privileged port
        with pytest.raises(ValidationError):
            WebSocketConfig(port=70000)  # Invalid port

    def test_osc_invalid_port(self) -> None:
        from hand_tracker.config import OSCConfig

        with pytest.raises(ValidationError):
            OSCConfig(port=1023)  # Just outside user range


class TestLoggingConfig:
    def test_invalid_level(self) -> None:
        with pytest.raises(ValidationError):
            LoggingConfig(level="VERBOSE")  # type: ignore

    def test_too_small_max_bytes(self) -> None:
        with pytest.raises(ValidationError):
            LoggingConfig(max_bytes=512)  # Minimum is 1024


class TestAppConfig:
    """Tests for the root AppConfig which handles hierarchy and env vars."""

    def test_env_override(self) -> None:
        os.environ["HANDFORGE__CAMERA__INDEX"] = "2"
        cfg = AppConfig()
        assert cfg.camera.index == 2

    def test_model_complexity_0_rejected(self) -> None:
        """model_complexity=0 must fail AppConfig cross-field validation."""
        os.environ["HANDFORGE__MEDIAPIPE__MODEL_COMPLEXITY"] = "0"
        with pytest.raises(ValidationError, match="world_landmarks"):
            AppConfig()

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(unknown_field="value")  # type: ignore


class TestYamlLoading:
    def test_load_config_from_yaml(self, tmp_path: Path) -> None:
        config_data = {
            "camera": {"index": 1, "width": 1280},
            "tracker": {"primary_hand": "Left"},
            "logging": {"level": "DEBUG"},
        }
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        cfg = load_config(str(config_file))
        assert cfg.camera.index == 1
        assert cfg.camera.width == 1280
        assert cfg.tracker.primary_hand == "Left"
        assert cfg.logging.level == "DEBUG"

    def test_load_config_default_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that load_config() defaults to 'config.yaml' in CWD if no path is given."""
        # Setup: create a config.yaml in a temporary directory
        config_data = {"camera": {"index": 5}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Action: Change working directory and call load_config without args
        monkeypatch.chdir(tmp_path)
        cfg = load_config()

        # Assert: It should have picked up the file from the current directory
        assert cfg.camera.index == 5
        assert cfg.config_path == "config.yaml"


class TestLoadConfig:
    def test_load_config_returns_frozen(self) -> None:
        # Pydantic v2 frozen models raise ValidationError on assignment
        cfg = load_config("non_existent.yaml")
        with pytest.raises(ValidationError):
            cfg.camera = CameraConfig()

    def test_deep_immutability(self) -> None:
        cfg = load_config("non_existent.yaml")
        with pytest.raises(ValidationError):
            cfg.camera.index = 99

    def test_load_config_is_cached(self) -> None:
        a = load_config("config.yaml")
        b = load_config("config.yaml")
        assert a is b
