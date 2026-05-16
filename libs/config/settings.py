from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Redis / Storage ─────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── Policy / System ─────────────────────────
    policy_path: str = "policies/default.yaml"

    # ── Detection ──────────────────────────────
    yolo_model: str = "yolov8n.pt"
    detector_model: str = "yolov8n.pt"
    detection_confidence_threshold: float = 0.45
    detector_device: str = "cpu"

    # ── Tracking ────────────────────────────────
    tracker_fps: float = 30
    tracker_max_age: int = 30
    tracker_n_init: int = 3
    tracker_max_cosine_distance: float = 0.4

    max_events_per_track: int = 50
    track_ttl_seconds: int = 300

    # ── Behavioral thresholds ───────────────────
    lingering_threshold_sec: float = 5.0
    movement_threshold_px: float = 8.0
    near_keypad_dist_px: float = 80.0

    keypad_center_x: float = 600.0
    keypad_center_y: float = 280.0

    # ── Reasoning ───────────────────────────────
    reasoning_dwell_threshold_seconds: float = 5.0
    reasoning_cooldown_seconds: float = 5.0

    # ── Device / Camera ─────────────────────────
    camera_id: str = "cam_01"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()