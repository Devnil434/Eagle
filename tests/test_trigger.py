import pytest

from libs.config.settings import settings
from libs.schemas.memory import (
    TrackSequence,
    TrackEvent,
    ActionHint,
)

from services.memory.trigger import (
    should_trigger_reasoning,
    reset_cooldown,
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


def make_sequence(
    dwell: float = 10.0,
    zones: list[str] | None = None,
    track_id: int = 1,
    action: ActionHint = ActionHint.LINGERING,
):
    event = TrackEvent(
        track_id=track_id,
        frame_id=1,
        timestamp_ms=1000,
        action_hint=action,
        dwell_time_seconds=dwell,
    )

    return TrackSequence(
        track_id=track_id,
        events=[event],
        total_dwell=dwell,
        zones_visited=(zones if zones is not None else ["restricted_zone"]),
    )


def test_returns_false_without_zone():
    seq = make_sequence(zones=[])

    result = should_trigger_reasoning(seq)

    assert result is False


def test_returns_false_below_dwell_threshold():
    seq = make_sequence(dwell=1.0)

    result = should_trigger_reasoning(seq)

    assert result is False


def test_returns_false_without_suspicious_actions():
    seq = make_sequence(action=ActionHint.WALKING)

    result = should_trigger_reasoning(seq)

    assert result is False


def test_returns_true_for_valid_suspicious_sequence():
    seq = make_sequence(track_id=100)

    reset_cooldown(100)

    result = should_trigger_reasoning(seq)

    assert result is True


def test_returns_false_during_cooldown():
    seq = make_sequence(track_id=200)

    reset_cooldown(200)

    first = should_trigger_reasoning(seq)
    second = should_trigger_reasoning(seq)

    assert first is True
    assert second is False


def test_reset_cooldown_allows_retrigger():
    seq = make_sequence(track_id=300)

    reset_cooldown(300)

    first = should_trigger_reasoning(seq)
    second = should_trigger_reasoning(seq)

    reset_cooldown(300)

    third = should_trigger_reasoning(seq)

    assert first is True
    assert second is False
    assert third is True
