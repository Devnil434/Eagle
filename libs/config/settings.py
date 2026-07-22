from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration for the Eagle surveillance system.

    Every field can be overridden via an environment variable of the same
    (uppercased) name or via a ``.env`` file in the project root.
    """

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── Memory / ring-buffer ──────────────────────────────────────────────
    max_events_per_track: int = 50
    track_ttl_seconds: int = 86_400  # 24 h

    # ── Action classifier thresholds ──────────────────────────────────────
    lingering_threshold_sec: float = 5.0
    movement_threshold_px: float = 8.0
    near_keypad_dist_px: float = 80.0
    keypad_center_x: float = 600.0
    keypad_center_y: float = 280.0

    # ── Detection ─────────────────────────────────────────────────────────
    yolo_model: str = "yolov8n.pt"
    detector_model: str = "yolov8n.pt"
    detection_confidence: float = 0.4
    detection_confidence_threshold: float = 0.45
    detector_device: str = "cpu"
    confidence_threshold: float = 0.45

    # ── Tracker ───────────────────────────────────────────────────────────
    tracker_fps: float = 30
    tracker_max_age: int = 30
    tracker_n_init: int = 3
    tracker_max_cosine_distance: float = 0.4

    # ── VLM / LLM providers ──────────────────────────────────────────────
    vlm_provider: str = "mock"
    llm_provider: str = "mock"
    ollama_host: str = "http://localhost:11434"

    # ── Reasoning / alerts ────────────────────────────────────────────────
    reasoning_dwell_threshold_seconds: float = 5.0
    reasoning_cooldown_seconds: float = 5.0
    reasoning_trigger_sec: float = 5.0
    ring_buffer_max: int = 50
    alert_dedup_window: int = 300

    # ── Backend / API ─────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev
    max_alerts_page: int = 50
    snapshot_dir: str = "/tmp/eagle_snapshots"

    # ── Policy ────────────────────────────────────────────────────────────
    policy_path: str = "policies/default.yaml"
    camera_id: str = "cam_01"

    # ── Kafka ─────────────────────────────────────────────────────────────
    use_kafka: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "track-events"

    # Lifecycle logging
    lifecycle_log_path: str = "data/logs/tracks.jsonl"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
