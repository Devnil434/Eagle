import pytest

from libs.config.settings import settings
from libs.schemas.memory import (
    TrackSequence,
    TrackEvent,
    ActionHint,
)

from libs.schemas.rules import AlertRule, RuleTimeWindow

from services.memory.trigger import (
    should_trigger_reasoning,
    reset_cooldown,
)
from services.rules.engine import RuleEngine


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


# ── Configurable alert rules ──────────────────────────────────────────────────

def make_rule_sequence(
    label: str | None = "person",
    zone: str = "restricted_door",
    dwell: float = 10.0,
    track_id: int = 1,
    action: ActionHint = ActionHint.LINGERING,
    confidence: float = 0.9,
    timestamp_ms: float = 1_755_000_000_000,
):
    """A sequence carrying the fields rules match on."""
    event = TrackEvent(
        track_id=track_id,
        frame_id=1,
        timestamp_ms=timestamp_ms,
        label=label,
        zone=zone,
        action_hint=action,
        dwell_time_seconds=dwell,
        confidence=confidence,
    )
    return TrackSequence(
        track_id=track_id,
        events=[event],
        total_dwell=dwell,
        zones_visited=[zone],
    )


def test_no_rules_configured_keeps_default_behaviour():
    """The engine abstains with an empty rule set, so the gate is unchanged."""
    reset_cooldown(400)
    seq = make_rule_sequence(track_id=400)

    assert should_trigger_reasoning(seq, engine=RuleEngine([])) is True


def test_matching_rule_allows_the_trigger():
    reset_cooldown(401)
    engine = RuleEngine([AlertRule(id="people", object_types=["person"])])
    seq = make_rule_sequence(track_id=401, label="person")

    assert should_trigger_reasoning(seq, engine=engine) is True


def test_non_matching_rule_suppresses_the_trigger():
    reset_cooldown(402)
    engine = RuleEngine([AlertRule(id="vehicles", object_types=["vehicle"])])
    seq = make_rule_sequence(track_id=402, label="person")

    assert should_trigger_reasoning(seq, engine=engine) is False


def test_rules_cannot_widen_the_gate():
    """A permissive rule must not bypass the dwell threshold."""
    reset_cooldown(403)
    engine = RuleEngine([AlertRule(id="everything")])
    seq = make_rule_sequence(track_id=403, dwell=0.5)

    assert should_trigger_reasoning(seq, engine=engine) is False


def test_zone_scoped_rule_filters_by_zone():
    reset_cooldown(404)
    engine = RuleEngine([AlertRule(id="door_only", zones=["restricted_door"])])

    assert should_trigger_reasoning(
        make_rule_sequence(track_id=404, zone="restricted_door"), engine=engine
    ) is True
    reset_cooldown(405)
    assert should_trigger_reasoning(
        make_rule_sequence(track_id=405, zone="safe_corridor"), engine=engine
    ) is False


def test_confidence_floor_suppresses_weak_detections():
    reset_cooldown(406)
    engine = RuleEngine([AlertRule(id="confident", min_confidence=0.8)])
    seq = make_rule_sequence(track_id=406, confidence=0.4)

    assert should_trigger_reasoning(seq, engine=engine) is False


def test_disabled_rule_is_ignored_and_engine_abstains():
    """Disabling the only rule restores default behaviour rather than muting all."""
    reset_cooldown(407)
    engine = RuleEngine([
        AlertRule(id="off", enabled=False, object_types=["vehicle"])
    ])
    seq = make_rule_sequence(track_id=407, label="person")

    assert should_trigger_reasoning(seq, engine=engine) is True


def test_per_rule_cooldown_overrides_the_global_default():
    """A rule with a longer cooldown throttles harder than the global setting."""
    reset_cooldown(408)
    engine = RuleEngine([AlertRule(id="throttled", cooldown_seconds=3600.0)])
    seq = make_rule_sequence(track_id=408)

    assert should_trigger_reasoning(seq, engine=engine) is True
    assert should_trigger_reasoning(seq, engine=engine) is False


def test_time_scoped_rule_uses_the_event_timestamp():
    """Rules judge activity when it happened, not when it is evaluated."""
    engine = RuleEngine([
        AlertRule(id="after_hours", time_windows=[RuleTimeWindow(start="19:00", end="07:00")])
    ])
    # 2026-08-14 23:30 UTC — inside the window
    reset_cooldown(409)
    inside = make_rule_sequence(track_id=409, timestamp_ms=1_755_214_200_000)
    # 2026-08-14 12:00 UTC — outside it
    reset_cooldown(410)
    outside = make_rule_sequence(track_id=410, timestamp_ms=1_755_172_800_000)

    assert should_trigger_reasoning(inside, engine=engine) is True
    assert should_trigger_reasoning(outside, engine=engine) is False


def test_legacy_events_without_a_label_do_not_match_typed_rules():
    reset_cooldown(411)
    engine = RuleEngine([AlertRule(id="people", object_types=["person"])])
    seq = make_rule_sequence(track_id=411, label=None)

    assert should_trigger_reasoning(seq, engine=engine) is False
