"""
Unit tests for Phase 3: event schemas, Redis ring buffer, action classification.
All Redis tests use fakeredis — no real Redis server needed.
"""
from __future__ import annotations

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import fakeredis
from libs.schemas.memory   import TrackEvent, TrackSequence, ActionHint
from services.memory.memory import MemoryStore, MAX_EVENTS_PER_TRACK


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    """MemoryStore backed by an in-memory fakeredis instance."""
    fake_r = fakeredis.FakeRedis(decode_responses=True)
    return MemoryStore(redis_client=fake_r)


def make_event(track_id: int, frame_id: int, zone: str | None = None,
               hint: ActionHint = ActionHint.WALKING, dwell: float = 0.0) -> TrackEvent:
    return TrackEvent(
        track_id           = track_id,
        camera_id          = "cam_01",
        frame_id           = frame_id,
        timestamp_ms       = time.time() * 1000 + frame_id * 33,
        zone               = zone,
        action_hint        = hint,
        bbox               = [100.0, 80.0, 150.0, 200.0],
        center             = (125.0, 140.0),
        dwell_time_seconds = dwell,
        confidence         = 0.91,
    )


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_track_event_serialises_cleanly():
    evt = make_event(1, 0)
    assert evt.track_id == 1
    assert ActionHint.WALKING.value == "walking"


def test_track_sequence_action_summary():
    seq = TrackSequence(
        track_id = 1,
        events   = [
            make_event(1, 0, hint=ActionHint.WALKING),
            make_event(1, 1, hint=ActionHint.WALKING),
            make_event(1, 2, hint=ActionHint.ZONE_ENTRY),
            make_event(1, 3, hint=ActionHint.LINGERING),
        ]
    )
    assert seq.action_summary == "walking → zone_entry → lingering"

def test_action_summary_empty():
    seq = TrackSequence(track_id=99, events=[])
    assert seq.action_summary == "unknown"

def test_action_summary_single_event():
    seq = TrackSequence(
        track_id=100,
        events=[make_event(100, 0, hint=ActionHint.WALKING)]
    )
    assert seq.action_summary == "walking"

def test_action_summary_all_same():
    seq = TrackSequence(
        track_id=101,
        events=[
            make_event(101, 0, hint=ActionHint.WALKING),
            make_event(101, 1, hint=ActionHint.WALKING),
            make_event(101, 2, hint=ActionHint.WALKING),
        ]
    )
    assert seq.action_summary == "walking"

def test_action_summary_alternating():
    seq = TrackSequence(
        track_id=102,
        events=[
            make_event(102, 0, hint=ActionHint.WALKING),
            make_event(102, 1, hint=ActionHint.ZONE_ENTRY),
            make_event(102, 2, hint=ActionHint.WALKING),
            make_event(102, 3, hint=ActionHint.LINGERING),
        ]
    )
    assert seq.action_summary == "walking → zone_entry → walking → lingering"


def test_track_sequence_duration():
    seq = TrackSequence(
        track_id = 2,
        events   = [
            make_event(2, 0),
            make_event(2, 30),
        ]
    )
    # timestamps differ by 30 * 33ms = ~990ms
    assert seq.duration_seconds == pytest.approx(0.99, abs=0.1)


# ── MemoryStore (fakeredis) tests ─────────────────────────────────────────────

def test_store_and_retrieve_event(store):
    evt = make_event(5, 0, zone="restricted_door", hint=ActionHint.ZONE_ENTRY)
    store.store_event(evt)
    seq = store.get_sequence(track_id=5)
    assert len(seq.events) == 1
    assert seq.events[0].zone == "restricted_door"
    assert seq.events[0].action_hint == ActionHint.ZONE_ENTRY


def test_ring_buffer_caps_at_max(store):
    """Storing MAX + 10 events should result in exactly MAX stored."""
    for i in range(MAX_EVENTS_PER_TRACK + 10):
        store.store_event(make_event(3, i))
    seq = store.get_sequence(3)
    assert len(seq.events) == MAX_EVENTS_PER_TRACK


def test_oldest_events_dropped(store):
    """After ring-buffer overflow, only the most recent MAX events remain."""
    for i in range(MAX_EVENTS_PER_TRACK + 5):
        store.store_event(make_event(6, i))
    seq = store.get_sequence(6)
    first_kept = seq.events[0].frame_id
    assert first_kept == 5   # first 5 frames dropped


def test_sequence_chronological_order(store):
    """Events must be returned in the order they were inserted (oldest first)."""
    for i in [0, 1, 2, 3, 4]:
        store.store_event(make_event(7, i))
    seq = store.get_sequence(7)
    frame_ids = [e.frame_id for e in seq.events]
    assert frame_ids == sorted(frame_ids)


def test_empty_sequence_for_unknown_track(store):
    seq = store.get_sequence(track_id=9999 )
    assert len(seq.events) == 0


def test_zones_visited_populated(store):
    store.store_event(make_event(10, 0, zone="safe_corridor", hint=ActionHint.ZONE_ENTRY))
    store.store_event(make_event(10, 5, zone="restricted_door", hint=ActionHint.ZONE_ENTRY))
    seq = store.get_sequence(10)
    assert set(seq.zones_visited) == {"safe_corridor", "restricted_door"}


def test_zone_entry_count(store):
    store.store_event(make_event(11, 0, zone="restricted_door", hint=ActionHint.ZONE_ENTRY))
    store.store_event(make_event(11, 30, zone="restricted_door", hint=ActionHint.ZONE_ENTRY))
    count = store.get_zone_entry_count(track_id=11, zone="restricted_door")
    assert count == 2


def test_active_tracks_set(store):
    store.store_event(make_event(20, 0))
    store.store_event(make_event(21, 0))
    active = store.get_active_track_ids("cam_01")
    assert {20, 21}.issubset(active)


def test_expire_track_removes_keys(store):
    store.store_event(make_event(30, 0))
    store.expire_track(30)
    seq = store.get_sequence(30)
    assert len(seq.events) == 0


def test_get_sequence_last_n(store):
    for i in range(30):
        store.store_event(make_event(40, i))
    seq = store.get_sequence(40, last_n=10)
    assert len(seq.events) == 10
    assert seq.events[0].frame_id == 20   # last 10 of frames 0–29


# ── Action classifier tests ───────────────────────────────────────────────────

def test_zone_entry_hint():
    from services.memory.action_classifier import classify_action
    from libs.schemas.tracking import TrackedObject, TrackState

    obj = TrackedObject(
        track_id=1, label="person", bbox=[100,80,200,300],
        confidence=0.9, center=(150,190), dwell_time_frames=1,
        dwell_time_seconds=0.0, state=TrackState.ACTIVE,
        zones_present=["restricted_door"],
    )
    registry = {}
    hint = classify_action(obj, None, registry)
    assert hint == ActionHint.ZONE_ENTRY
    # Second call same zone → no longer ZONE_ENTRY
    hint2 = classify_action(obj, obj, registry)
    assert hint2 != ActionHint.ZONE_ENTRY


def test_lingering_hint():
    from services.memory.action_classifier import classify_action, LINGERING_THRESHOLD_SEC
    from libs.schemas.tracking import TrackedObject, TrackState

    obj = TrackedObject(
        track_id=2, label="person", bbox=[100,80,200,300],
        confidence=0.9, center=(150,190),
        dwell_time_frames=300, dwell_time_seconds=LINGERING_THRESHOLD_SEC + 1,
        state=TrackState.ACTIVE, zones_present=["restricted_door"],
    )
    registry = {2: {"restricted_door"}}   # already entered
    hint = classify_action(obj, obj, registry)
    assert hint == ActionHint.LINGERING


def test_repeated_approach_second_entry():
    from services.memory.action_classifier import classify_action
    from libs.schemas.tracking import TrackedObject, TrackState

    obj = TrackedObject(
        track_id=3, label="person", bbox=[100,80,200,300],
        confidence=0.9, center=(150,190), dwell_time_frames=1,
        dwell_time_seconds=0.0, state=TrackState.ACTIVE,
        zones_present=["restricted_door"],
    )
    registry = {}
    counts = {}
    cooldown = {}
    
    # First entry
    hint1 = classify_action(obj, None, registry, counts, cooldown, 1000.0)
    assert hint1 == ActionHint.ZONE_ENTRY
    
    # Second entry
    hint2 = classify_action(obj, None, registry, counts, cooldown, 2000.0)
    assert hint2 == ActionHint.REPEATED_APPROACH


def test_repeated_approach_no_spam():
    from services.memory.action_classifier import classify_action
    from libs.schemas.tracking import TrackedObject, TrackState

    obj = TrackedObject(
        track_id=4, label="person", bbox=[100,80,200,300],
        confidence=0.9, center=(150,190), dwell_time_frames=1,
        dwell_time_seconds=0.0, state=TrackState.ACTIVE,
        zones_present=["restricted_door"],
    )
    registry = {}
    counts = {}
    cooldown = {}
    
    # First entry
    hint1 = classify_action(obj, None, registry, counts, cooldown, 1000.0)
    assert hint1 == ActionHint.ZONE_ENTRY
    
    # Still inside the zone, not a new entry
    hint_stay = classify_action(obj, obj, registry, counts, cooldown, 2000.0)
    assert hint_stay != ActionHint.REPEATED_APPROACH
    assert hint_stay != ActionHint.ZONE_ENTRY
    
    # Leave and enter again
    hint2 = classify_action(obj, None, registry, counts, cooldown, 3000.0)
    assert hint2 == ActionHint.REPEATED_APPROACH


def test_repeated_approach_cooldown():
    from services.memory.action_classifier import classify_action
    from libs.schemas.tracking import TrackedObject, TrackState

    obj = TrackedObject(
        track_id=5, label="person", bbox=[100,80,200,300],
        confidence=0.9, center=(150,190), dwell_time_frames=1,
        dwell_time_seconds=0.0, state=TrackState.ACTIVE,
        zones_present=["restricted_door"],
    )
    registry = {}
    counts = {}
    cooldown = {}
    
    # First entry
    classify_action(obj, None, registry, counts, cooldown, 1000.0)
    
    # Second entry (triggers REPEATED_APPROACH, sets cooldown)
    hint2 = classify_action(obj, None, registry, counts, cooldown, 2000.0)
    assert hint2 == ActionHint.REPEATED_APPROACH
    
    # Third entry right after (within 10s cooldown)
    hint3 = classify_action(obj, None, registry, counts, cooldown, 5000.0)
    assert hint3 != ActionHint.REPEATED_APPROACH
    
    # Fourth entry after cooldown expires (15000 is > 10000 ms since 2000)
    hint4 = classify_action(obj, None, registry, counts, cooldown, 15000.0)
    assert hint4 == ActionHint.REPEATED_APPROACH


def test_walking_hint():
    from services.memory.action_classifier import classify_action
    from libs.schemas.tracking import TrackedObject, TrackState

    obj = TrackedObject(
        track_id=10,
        label="person",
        bbox=[100, 80, 200, 300],
        confidence=0.9,
        center=(150, 190),
        dwell_time_frames=1,
        dwell_time_seconds=0.0,
        state=TrackState.ACTIVE,
        zones_present=[],
    )

    registry = {}

    # simulate previous frame far away → movement detected
    prev = TrackedObject(
        track_id=10,
        label="person",
        bbox=[100, 80, 200, 300],
        confidence=0.9,
        center=(300, 400),  # big movement difference
        dwell_time_frames=1,
        dwell_time_seconds=0.0,
        state=TrackState.ACTIVE,
        zones_present=[],
    )

    hint = classify_action(obj, prev, registry)
    assert hint == ActionHint.WALKING


def test_standing_hint():
    from services.memory.action_classifier import classify_action
    from libs.schemas.tracking import TrackedObject, TrackState

    obj = TrackedObject(
        track_id=11,
        label="person",
        bbox=[100, 80, 200, 300],
        confidence=0.9,
        center=(150, 190),
        dwell_time_frames=1,
        dwell_time_seconds=0.0,
        state=TrackState.ACTIVE,
        zones_present=[],
    )

    registry = {}

    prev = TrackedObject(
        track_id=11,
        label="person",
        bbox=[100, 80, 200, 300],
        confidence=0.9,
        center=(151, 191),  # tiny movement
        dwell_time_frames=1,
        dwell_time_seconds=0.0,
        state=TrackState.ACTIVE,
        zones_present=[],
    )

    hint = classify_action(obj, prev, registry)
    assert hint == ActionHint.STANDING


def test_near_keypad_hint():
    from services.memory.action_classifier import classify_action, KEYPAD_CENTER
    from libs.schemas.tracking import TrackedObject, TrackState

    obj = TrackedObject(
        track_id=12,
        label="person",
        bbox=[100, 80, 200, 300],
        confidence=0.9,
        center=KEYPAD_CENTER,  # directly near keypad
        dwell_time_frames=1,
        dwell_time_seconds=0.0,
        state=TrackState.ACTIVE,
        zones_present=[],
    )

    registry = {}

    hint = classify_action(obj, None, registry)
    assert hint == ActionHint.NEAR_KEYPAD


# ── reasoning_result_id tests ──────────────────────────────────────────────────

def test_reasoning_result_id_absent_by_default():
    """reasoning_result_id should default to None when no reasoning has run."""
    evt = make_event(50, 0)
    assert evt.reasoning_result_id is None


def test_reasoning_result_id_present_after_set(store):
    """reasoning_result_id should be stored and retrieved correctly."""
    evt = make_event(51, 0, zone="restricted_door", hint=ActionHint.ZONE_ENTRY)
    evt.reasoning_result_id = "test-alert-id-123"
    store.store_event(evt)
    seq = store.get_sequence(track_id=51)
    assert seq.events[0].reasoning_result_id == "test-alert-id-123"

