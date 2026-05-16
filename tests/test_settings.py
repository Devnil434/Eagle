from libs.config.settings import Settings


def test_settings_defaults():
    settings = Settings()

    assert settings.redis_url == "redis://localhost:6379"
    assert settings.max_events_per_track == 50
    assert settings.track_ttl_seconds == 300
    assert settings.lingering_threshold_sec == 5.0
    assert settings.movement_threshold_px == 8.0
    assert settings.near_keypad_dist_px == 80.0
    assert settings.confidence_threshold == 0.45