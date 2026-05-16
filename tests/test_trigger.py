import pytest
from libs.config.settings import settings

from services.memory.trigger import (
    should_trigger_reasoning,
    reset_cooldown,
)

from libs.schemas.tracking import (
    TrackLifecycleEvent,
    TrackState,
)

@pytest.fixture(autouse=True)
def fixed_reasoning_gate_config():
    old_dwell = settings.reasoning_dwell_threshold_seconds
    old_cooldown = settings.reasoning_cooldown_seconds

    settings.reasoning_dwell_threshold_seconds = 5.0
    settings.reasoning_cooldown_seconds = 5.0

    yield

    settings.reasoning_dwell_threshold_seconds = old_dwell
    settings.reasoning_cooldown_seconds = old_cooldown


def make_event(
    dwell: float = 10.0,
    zones: list[str] | None = None,
    track_id: int = 1,
):
    return TrackLifecycleEvent(
        event=TrackState.LOST,
        track_id=track_id,
        frame_id=1,
        zones_present=zones if zones is not None else ["restricted_zone"],
        dwell_time_seconds=dwell,
    )


def test_returns_false_without_zone():
    event = make_event(zones=[])

    result = should_trigger_reasoning(
        event,
        {"LINGERING"},
    )

    assert result is False


def test_returns_false_below_dwell_threshold():
    event = make_event(dwell=1.0)

    result = should_trigger_reasoning(
        event,
        {"LINGERING"},
    )

    assert result is False


def test_returns_false_without_suspicious_actions():
    event = make_event()

    result = should_trigger_reasoning(
        event,
        {"NORMAL_WALKING"},
    )

    assert result is False


def test_returns_true_for_valid_suspicious_sequence():
    event = make_event(track_id=100)

    reset_cooldown(100)

    result = should_trigger_reasoning(
        event,
        {"LINGERING"},
    )

    assert result is True


def test_returns_false_during_cooldown():
    event = make_event(track_id=200)

    reset_cooldown(200)

    first = should_trigger_reasoning(
        event,
        {"LINGERING"},
    )

    second = should_trigger_reasoning(
        event,
        {"LINGERING"},
    )

    assert first is True
    assert second is False


def test_reset_cooldown_allows_retrigger():
    event = make_event(track_id=300)

    reset_cooldown(300)

    first = should_trigger_reasoning(
        event,
        {"LINGERING"},
    )

    second = should_trigger_reasoning(
        event,
        {"LINGERING"},
    )

    reset_cooldown(300)

    third = should_trigger_reasoning(
        event,
        {"LINGERING"},
    )

    assert first is True
    assert second is False
    assert third is True