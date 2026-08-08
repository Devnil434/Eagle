
import json
import pytest

from utils.synthetic_event_generator import (
    EVENT_TYPES,
    export_events_to_json,
    generate_event,
    generate_events,
)


def test_generate_event_returns_valid_structure():
    event = generate_event(event_type="loitering", person_id=1)

    assert event["person_id"] == 1
    assert event["event_type"] == "loitering"
    assert "timestamp" in event
    assert "location" in event
    assert "confidence" in event
    assert "metadata" in event


def test_generate_event_rejects_invalid_event_type():
    with pytest.raises(ValueError):
        generate_event(event_type="invalid_event")


def test_generate_events_returns_requested_count():
    events = generate_events(count=5)

    assert len(events) == 5
    assert all(event["event_type"] in EVENT_TYPES for event in events)


def test_generate_events_rejects_invalid_count():
    with pytest.raises(ValueError):
        generate_events(count=0)


def test_export_events_to_json(tmp_path):
    events = generate_events(count=2)
    output_file = tmp_path / "events.json"

    export_events_to_json(events, str(output_file))

    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert len(data) == 2


def test_generate_event_is_deterministic_with_seed():
    from datetime import datetime, timezone

    fixed_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    event_one = generate_event(seed=42, timestamp=fixed_time)
    event_two = generate_event(seed=42, timestamp=fixed_time)

    assert event_one == event_two


def test_generate_events_rejects_invalid_interval_seconds():
    with pytest.raises(ValueError):
        generate_events(count=3, interval_seconds=0)

    with pytest.raises(ValueError):
        generate_events(count=3, interval_seconds=-10)
