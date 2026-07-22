"""
Unit tests for libs/config/settings.py (issue #32).

Verifies:
  - Default values match the documented contract.
  - Environment variables override defaults correctly.
  - The singleton ``settings`` object is importable and valid.
"""
from __future__ import annotations

import os
import pytest

from libs.config.settings import Settings


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch):
    """Ensure that the local .env file and existing environment variables
    do not pollute the Settings unit tests.
    """
    # Prevent loading of .env file
    original_init = Settings.__init__
    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("_env_file", None)
        original_init(self, *args, **kwargs)
    monkeypatch.setattr(Settings, "__init__", patched_init)

    # Clear environment variables corresponding to Settings fields
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)





# ── Default values ────────────────────────────────────────────────────────────

class TestDefaults:
    """Verify all defaults match the values specified in the issue."""

    def test_redis_url(self):
        s = Settings()
        assert s.redis_url == "redis://localhost:6379"

    def test_max_events_per_track(self):
        s = Settings()
        assert s.max_events_per_track == 50

    def test_track_ttl_seconds(self):
        s = Settings()
        assert s.track_ttl_seconds == 86_400

    def test_lingering_threshold_sec(self):
        s = Settings()
        assert s.lingering_threshold_sec == 5.0

    def test_movement_threshold_px(self):
        s = Settings()
        assert s.movement_threshold_px == 8.0

    def test_near_keypad_dist_px(self):
        s = Settings()
        assert s.near_keypad_dist_px == 80.0

    def test_keypad_center_x(self):
        s = Settings()
        assert s.keypad_center_x == 600.0

    def test_keypad_center_y(self):
        s = Settings()
        assert s.keypad_center_y == 280.0

    def test_yolo_model(self):
        s = Settings()
        assert s.yolo_model == "yolov8n.pt"

    def test_confidence_threshold(self):
        s = Settings()
        assert s.confidence_threshold == 0.45

    def test_tracker_max_age(self):
        s = Settings()
        assert s.tracker_max_age == 30

    def test_tracker_n_init(self):
        s = Settings()
        assert s.tracker_n_init == 3

    def test_vlm_provider_default_is_mock(self):
        s = Settings()
        assert s.vlm_provider == "mock"


# ── Environment overrides ────────────────────────────────────────────────────

class TestEnvOverrides:
    """Verify that environment variables correctly override defaults."""

    def test_redis_url_override(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://prod:6380/1")
        s = Settings()
        assert s.redis_url == "redis://prod:6380/1"

    def test_max_events_per_track_override(self, monkeypatch):
        monkeypatch.setenv("MAX_EVENTS_PER_TRACK", "100")
        s = Settings()
        assert s.max_events_per_track == 100

    def test_track_ttl_seconds_override(self, monkeypatch):
        monkeypatch.setenv("TRACK_TTL_SECONDS", "3600")
        s = Settings()
        assert s.track_ttl_seconds == 3600

    def test_lingering_threshold_override(self, monkeypatch):
        monkeypatch.setenv("LINGERING_THRESHOLD_SEC", "10.0")
        s = Settings()
        assert s.lingering_threshold_sec == 10.0

    def test_movement_threshold_override(self, monkeypatch):
        monkeypatch.setenv("MOVEMENT_THRESHOLD_PX", "20.0")
        s = Settings()
        assert s.movement_threshold_px == 20.0

    def test_keypad_center_override(self, monkeypatch):
        monkeypatch.setenv("KEYPAD_CENTER_X", "400.0")
        monkeypatch.setenv("KEYPAD_CENTER_Y", "300.0")
        s = Settings()
        assert s.keypad_center_x == 400.0
        assert s.keypad_center_y == 300.0

    def test_yolo_model_override(self, monkeypatch):
        monkeypatch.setenv("YOLO_MODEL", "yolov8s.pt")
        s = Settings()
        assert s.yolo_model == "yolov8s.pt"

    def test_confidence_threshold_override(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.60")
        s = Settings()
        assert s.confidence_threshold == 0.60

    def test_tracker_max_age_override(self, monkeypatch):
        monkeypatch.setenv("TRACKER_MAX_AGE", "60")
        s = Settings()
        assert s.tracker_max_age == 60

    def test_vlm_provider_override(self, monkeypatch):
        monkeypatch.setenv("VLM_PROVIDER", "ollama")
        s = Settings()
        assert s.vlm_provider == "ollama"

    def test_api_port_override(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9000")
        s = Settings()
        assert s.api_port == 9000


# ── Singleton import ─────────────────────────────────────────────────────────

class TestSingleton:
    """Verify the module-level ``settings`` instance is usable."""

    def test_settings_singleton_importable(self):
        from libs.config.settings import settings
        assert isinstance(settings, Settings)

    def test_settings_singleton_has_expected_type(self):
        from libs.config.settings import settings
        assert hasattr(settings, "redis_url")
        assert hasattr(settings, "max_events_per_track")
        assert hasattr(settings, "track_ttl_seconds")
        assert hasattr(settings, "lingering_threshold_sec")
