"""
Unit tests for configurable alert rules: schema, object-group resolution, time
windows, and the matching engine.

The engine is pure, so these tests need no config file and no frozen clock.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, time, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from libs.schemas.memory import ActionHint, TrackEvent, TrackSequence
from libs.schemas.rules import (
    AlertRule,
    RuleSet,
    RuleTimeWindow,
    resolve_object_types,
)
from services.rules.engine import RuleContext, RuleEngine

# Friday 2026-08-14 and Saturday 2026-08-15
FRIDAY_NOON = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
FRIDAY_NIGHT = datetime(2026, 8, 14, 23, 0, tzinfo=timezone.utc)
SATURDAY_NOON = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)


def make_event(
    label: str | None = "person",
    zone: str | None = "restricted_door",
    hint: ActionHint = ActionHint.LINGERING,
    confidence: float = 0.9,
    timestamp_ms: float = 0.0,
) -> TrackEvent:
    return TrackEvent(
        track_id     = 1,
        frame_id     = 0,
        timestamp_ms = timestamp_ms,
        label        = label,
        zone         = zone,
        action_hint  = hint,
        confidence   = confidence,
    )


def make_context(moment: datetime | None = FRIDAY_NOON, **event_kwargs) -> RuleContext:
    event = make_event(**event_kwargs)
    seq = TrackSequence(
        track_id      = 1,
        events        = [event],
        zones_visited = [event.zone] if event.zone else [],
        total_dwell   = 10.0,
    )
    return RuleContext.from_sequence(seq, moment=moment)


# ── Object group resolution ───────────────────────────────────────────────────

def test_vehicle_group_expands_to_coco_classes():
    resolved = resolve_object_types(["vehicle"])

    assert {"car", "truck", "bus", "motorcycle", "bicycle"} <= resolved
    assert "person" not in resolved


def test_raw_coco_class_passes_through():
    assert resolve_object_types(["car"]) == {"car"}


def test_unknown_object_type_is_kept_verbatim():
    """Any COCO class must be nameable without the taxonomy knowing it."""
    assert resolve_object_types(["fire hydrant"]) == {"fire hydrant"}


def test_group_names_are_case_insensitive():
    assert resolve_object_types(["Vehicle"]) == resolve_object_types(["vehicle"])


def test_groups_and_raw_classes_combine():
    resolved = resolve_object_types(["person", "car"])

    assert resolved == {"person", "car"}


# ── Time windows ──────────────────────────────────────────────────────────────

def test_window_matches_inside_range():
    window = RuleTimeWindow(start=time(9, 0), end=time(17, 0))

    assert window.contains(FRIDAY_NOON) is True


def test_window_rejects_outside_range():
    window = RuleTimeWindow(start=time(9, 0), end=time(17, 0))

    assert window.contains(FRIDAY_NIGHT) is False


def test_window_crossing_midnight_matches_late_evening():
    window = RuleTimeWindow(start=time(19, 0), end=time(7, 0))

    assert window.contains(FRIDAY_NIGHT) is True


def test_window_crossing_midnight_matches_early_morning():
    window = RuleTimeWindow(start=time(19, 0), end=time(7, 0))
    early = datetime(2026, 8, 15, 3, 0, tzinfo=timezone.utc)

    assert window.contains(early) is True


def test_window_crossing_midnight_rejects_midday():
    window = RuleTimeWindow(start=time(19, 0), end=time(7, 0))

    assert window.contains(FRIDAY_NOON) is False


def test_window_limited_to_weekdays_rejects_other_days():
    window = RuleTimeWindow(start=time(0, 0), end=time(23, 59), days=["sat", "sun"])

    assert window.contains(SATURDAY_NOON) is True
    assert window.contains(FRIDAY_NOON) is False


def test_window_parses_hh_mm_strings_from_yaml():
    window = RuleTimeWindow(start="22:00", end="06:00")

    assert window.start == time(22, 0)
    assert window.end == time(6, 0)


# ── Rule schema ───────────────────────────────────────────────────────────────

def test_rule_defaults_to_enabled_and_unrestricted():
    rule = AlertRule(id="any")

    assert rule.enabled is True
    assert rule.object_types == []
    assert rule.min_confidence == 0.0


def test_blank_entries_are_stripped():
    rule = AlertRule(id="r", object_types=[" person ", ""], zones=["  dock  "])

    assert rule.object_types == ["person"]
    assert rule.zones == ["dock"]


def test_confidence_outside_unit_range_is_rejected():
    with pytest.raises(ValueError):
        AlertRule(id="r", min_confidence=1.5)


def test_empty_rule_id_is_rejected():
    with pytest.raises(ValueError):
        AlertRule(id="")


def test_duplicate_rule_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate rule id"):
        RuleSet(rules=[AlertRule(id="same"), AlertRule(id="same")])


def test_rule_set_reports_enabled_rules():
    rule_set = RuleSet(
        rules=[AlertRule(id="on"), AlertRule(id="off", enabled=False)]
    )

    assert [r.id for r in rule_set.enabled_rules] == ["on"]


# ── Engine: abstention ────────────────────────────────────────────────────────

def test_engine_abstains_with_no_rules():
    """No rules configured must mean no opinion, so callers keep their default."""
    decision = RuleEngine([]).evaluate(make_context())

    assert decision.abstained is True
    assert decision.matched is False
    assert decision.suppressed is False


def test_engine_abstains_when_every_rule_is_disabled():
    engine = RuleEngine([AlertRule(id="off", enabled=False, object_types=["person"])])

    decision = engine.evaluate(make_context())

    assert decision.abstained is True
    assert decision.suppressed is False


def test_disabled_rule_cannot_match():
    engine = RuleEngine([
        AlertRule(id="off", enabled=False, object_types=["person"]),
        AlertRule(id="on", object_types=["car"]),
    ])

    decision = engine.evaluate(make_context(label="person"))

    assert decision.matched is False
    assert decision.suppressed is True


# ── Engine: matching dimensions ───────────────────────────────────────────────

def test_unrestricted_rule_matches_anything():
    decision = RuleEngine([AlertRule(id="any")]).evaluate(make_context())

    assert decision.matched is True
    assert decision.rule_id == "any"


def test_object_type_must_match():
    engine = RuleEngine([AlertRule(id="vehicles", object_types=["vehicle"])])

    assert engine.evaluate(make_context(label="car")).matched is True
    assert engine.evaluate(make_context(label="person")).matched is False


def test_zone_must_match():
    engine = RuleEngine([AlertRule(id="door", zones=["restricted_door"])])

    assert engine.evaluate(make_context(zone="restricted_door")).matched is True
    assert engine.evaluate(make_context(zone="safe_corridor")).matched is False


def test_action_hint_must_match():
    engine = RuleEngine([
        AlertRule(id="lingering", action_hints=[ActionHint.LINGERING])
    ])

    assert engine.evaluate(make_context(hint=ActionHint.LINGERING)).matched is True
    assert engine.evaluate(make_context(hint=ActionHint.WALKING)).matched is False


def test_confidence_floor_is_enforced():
    engine = RuleEngine([AlertRule(id="confident", min_confidence=0.8)])

    assert engine.evaluate(make_context(confidence=0.9)).matched is True
    assert engine.evaluate(make_context(confidence=0.5)).matched is False


def test_confidence_floor_is_inclusive():
    engine = RuleEngine([AlertRule(id="confident", min_confidence=0.8)])

    assert engine.evaluate(make_context(confidence=0.8)).matched is True


def test_dimensions_are_anded_together():
    engine = RuleEngine([
        AlertRule(id="strict", object_types=["person"], zones=["restricted_door"])
    ])

    assert engine.evaluate(make_context(label="person", zone="restricted_door")).matched
    assert not engine.evaluate(make_context(label="person", zone="safe_corridor")).matched
    assert not engine.evaluate(make_context(label="car", zone="restricted_door")).matched


def test_time_scoped_rule_respects_the_window():
    engine = RuleEngine([
        AlertRule(id="after_hours", time_windows=[RuleTimeWindow(start="19:00", end="07:00")])
    ])

    assert engine.evaluate(make_context(moment=FRIDAY_NIGHT)).matched is True
    assert engine.evaluate(make_context(moment=FRIDAY_NOON)).matched is False


def test_time_scoped_rule_does_not_match_without_a_timestamp():
    """Absent a timestamp a time-scoped rule must not fire unconditionally."""
    engine = RuleEngine([
        AlertRule(id="after_hours", time_windows=[RuleTimeWindow(start="19:00", end="07:00")])
    ])

    assert engine.evaluate(make_context(moment=None)).matched is False


def test_events_without_an_object_class_cannot_match_a_typed_rule():
    """Events predating object-class capture must not match person/vehicle rules."""
    engine = RuleEngine([AlertRule(id="people", object_types=["person"])])

    assert engine.evaluate(make_context(label=None)).matched is False


def test_events_without_an_object_class_still_match_untyped_rules():
    engine = RuleEngine([AlertRule(id="any", zones=["restricted_door"])])

    assert engine.evaluate(make_context(label=None)).matched is True


# ── Engine: ordering and metadata ─────────────────────────────────────────────

def test_first_matching_rule_wins():
    engine = RuleEngine([
        AlertRule(id="specific", object_types=["person"], zones=["restricted_door"]),
        AlertRule(id="general", object_types=["person"]),
    ])

    assert engine.evaluate(make_context()).rule_id == "specific"


def test_rejected_rules_are_reported_for_diagnosis():
    engine = RuleEngine([
        AlertRule(id="vehicles", object_types=["vehicle"]),
        AlertRule(id="people", object_types=["person"]),
    ])

    decision = engine.evaluate(make_context(label="person"))

    assert decision.rule_id == "people"
    assert decision.rejected == ("vehicles",)


def test_matched_rule_exposes_its_cooldown_override():
    engine = RuleEngine([AlertRule(id="throttled", cooldown_seconds=120.0)])

    assert engine.evaluate(make_context()).cooldown_seconds == 120.0


def test_engine_accepts_a_rule_set():
    engine = RuleEngine(RuleSet(rules=[AlertRule(id="any")]))

    assert [r.id for r in engine.rules] == ["any"]


# ── Context construction ──────────────────────────────────────────────────────

def test_context_uses_highest_confidence_in_the_sequence():
    """A confidence floor should pass if the detector was ever that certain."""
    seq = TrackSequence(
        track_id = 1,
        events   = [
            make_event(confidence=0.4),
            make_event(confidence=0.95),
            make_event(confidence=0.5),
        ],
        zones_visited = ["restricted_door"],
    )

    assert RuleContext.from_sequence(seq).confidence == pytest.approx(0.95)


def test_context_collects_every_label_zone_and_hint():
    seq = TrackSequence(
        track_id = 1,
        events   = [
            make_event(label="person", zone="safe_corridor", hint=ActionHint.WALKING),
            make_event(label="backpack", zone="restricted_door", hint=ActionHint.LINGERING),
        ],
    )

    context = RuleContext.from_sequence(seq)

    assert context.object_labels == {"person", "backpack"}
    assert context.zones == {"safe_corridor", "restricted_door"}
    assert context.action_hints == {ActionHint.WALKING, ActionHint.LINGERING}


def test_context_from_empty_sequence_is_harmless():
    context = RuleContext.from_sequence(TrackSequence(track_id=1))

    assert context.confidence == 0.0
    assert context.object_labels == frozenset()
