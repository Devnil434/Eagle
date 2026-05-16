from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    max_events_per_track: int = 50
    track_ttl_seconds: int = 300

    lingering_threshold_sec: float = 5.0
    movement_threshold_px: float = 8.0
    near_keypad_dist_px: float = 80.0

    keypad_center_x: float = 600.0
    keypad_center_y: float = 280.0

    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.45

    tracker_max_age: int = 30
    tracker_n_init: int = 3

    class Config:
        env_file = ".env"

settings = Settings()