"""
Unit tests for alert querying on the Redis-backed store.

Covers the time-window and bulk-feedback reads that incident summary reports
depend on, plus the guarantee that alerts written before provenance fields
existed still deserialize.  All Redis access is fakeredis.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fakeredis
import pytest

from libs.schemas.reasoning import ReasoningResult
from services.memory.ring_buffer import MemoryStore

BASE_MS = 1_718_000_000_000.0


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(redis_client=fakeredis.FakeRedis(decode_responses=True))


def seed_alert(
    store: MemoryStore,
    *,
    alert_id: str,
    offset_ms: float,
    camera_id: str = "cam_01",
    zone: str | None = "restricted_door",
) -> ReasoningResult:
    alert = ReasoningResult(
        track_id       = 1,
        camera_id      = camera_id,
        label          = "Suspicious",
        confidence     = 0.8,
        reason         = "Lingering near the keypad.",
        timestamp_ms   = BASE_MS + offset_ms,
        severity_score = 0.6,
        alert_id       = alert_id,
        zone           = zone,
        object_labels  = ["person"],
    )
    store.store_alert(
        alert_json   = alert.model_dump_json(),
        timestamp_ms = alert.timestamp_ms,
        camera_id    = camera_id,
    )
    return alert


def alert_ids(raws: list[str]) -> list[str]:
    return [json.loads(raw)["alert_id"] for raw in raws]


# ── Time-window queries ───────────────────────────────────────────────────────

def test_range_query_returns_alerts_oldest_first(store):
    seed_alert(store, alert_id="third", offset_ms=3_000)
    seed_alert(store, alert_id="first", offset_ms=1_000)
    seed_alert(store, alert_id="second", offset_ms=2_000)

    raws = store.get_alerts_in_range(BASE_MS, BASE_MS + 10_000, camera_id="cam_01")

    assert alert_ids(raws) == ["first", "second", "third"]


def test_range_bounds_are_inclusive(store):
    seed_alert(store, alert_id="at_start", offset_ms=0)
    seed_alert(store, alert_id="at_end", offset_ms=5_000)

    raws = store.get_alerts_in_range(BASE_MS, BASE_MS + 5_000, camera_id="cam_01")

    assert set(alert_ids(raws)) == {"at_start", "at_end"}


def test_alerts_outside_the_window_are_excluded(store):
    seed_alert(store, alert_id="before", offset_ms=-60_000)
    seed_alert(store, alert_id="inside", offset_ms=1_000)
    seed_alert(store, alert_id="after", offset_ms=60_000)

    raws = store.get_alerts_in_range(BASE_MS, BASE_MS + 10_000, camera_id="cam_01")

    assert alert_ids(raws) == ["inside"]


def test_empty_window_returns_nothing(store):
    seed_alert(store, alert_id="a1", offset_ms=1_000)

    assert store.get_alerts_in_range(BASE_MS + 500_000, BASE_MS + 600_000) == []


def test_inverted_window_returns_nothing_rather_than_raising(store):
    seed_alert(store, alert_id="a1", offset_ms=1_000)

    assert store.get_alerts_in_range(BASE_MS + 10_000, BASE_MS) == []


def test_unknown_camera_returns_nothing(store):
    seed_alert(store, alert_id="a1", offset_ms=1_000, camera_id="cam_01")

    assert store.get_alerts_in_range(BASE_MS, BASE_MS + 10_000, camera_id="cam_99") == []


# ── Multi-camera aggregation ──────────────────────────────────────────────────

def test_omitting_camera_id_aggregates_every_camera(store):
    seed_alert(store, alert_id="a1", offset_ms=1_000, camera_id="cam_01")
    seed_alert(store, alert_id="a2", offset_ms=2_000, camera_id="cam_02")

    raws = store.get_alerts_in_range(BASE_MS, BASE_MS + 10_000)

    assert set(alert_ids(raws)) == {"a1", "a2"}


def test_cross_camera_results_stay_chronological(store):
    """Each camera's slice arrives sorted; the merge must re-sort globally."""
    seed_alert(store, alert_id="cam1_late", offset_ms=4_000, camera_id="cam_01")
    seed_alert(store, alert_id="cam2_early", offset_ms=1_000, camera_id="cam_02")
    seed_alert(store, alert_id="cam1_mid", offset_ms=2_000, camera_id="cam_01")

    raws = store.get_alerts_in_range(BASE_MS, BASE_MS + 10_000)

    assert alert_ids(raws) == ["cam2_early", "cam1_mid", "cam1_late"]


def test_camera_filter_restricts_results(store):
    seed_alert(store, alert_id="a1", offset_ms=1_000, camera_id="cam_01")
    seed_alert(store, alert_id="a2", offset_ms=2_000, camera_id="cam_02")

    raws = store.get_alerts_in_range(BASE_MS, BASE_MS + 10_000, camera_id="cam_02")

    assert alert_ids(raws) == ["a2"]


# ── Limit ─────────────────────────────────────────────────────────────────────

def test_limit_caps_returned_alerts(store):
    for i in range(10):
        seed_alert(store, alert_id=f"a{i}", offset_ms=i * 1_000)

    raws = store.get_alerts_in_range(BASE_MS, BASE_MS + 100_000, limit=4)

    assert len(raws) == 4


def test_limit_keeps_the_earliest_alerts_in_the_window(store):
    for i in range(5):
        seed_alert(store, alert_id=f"a{i}", offset_ms=i * 1_000)

    raws = store.get_alerts_in_range(BASE_MS, BASE_MS + 100_000, limit=2)

    assert alert_ids(raws) == ["a0", "a1"]


def test_non_positive_limit_returns_nothing(store):
    seed_alert(store, alert_id="a1", offset_ms=1_000)

    assert store.get_alerts_in_range(BASE_MS, BASE_MS + 10_000, limit=0) == []


# ── Bulk feedback ─────────────────────────────────────────────────────────────

def test_bulk_feedback_resolves_only_alerts_with_verdicts(store):
    store.store_feedback("a1", "confirmed", "op_1", "", BASE_MS)
    store.store_feedback("a2", "dismissed", "op_1", "", BASE_MS)

    resolved = store.get_feedback_bulk(["a1", "a2", "a3"])

    assert resolved == {"a1": "confirmed", "a2": "dismissed"}


def test_bulk_feedback_handles_empty_input(store):
    assert store.get_feedback_bulk([]) == {}


def test_bulk_feedback_matches_single_lookup(store):
    store.store_feedback("a1", "confirmed", "op_1", "", BASE_MS)

    assert store.get_feedback_bulk(["a1"])["a1"] == store.get_feedback("a1")


# ── Backward compatibility ────────────────────────────────────────────────────

def test_alerts_stored_before_provenance_fields_still_parse():
    """Alerts already in Redis lack zone/object_labels and must remain readable."""
    legacy = {
        "track_id": 4,
        "camera_id": "cam_01",
        "label": "Suspicious",
        "confidence": 0.77,
        "reason": "Loitering by the door.",
        "key_signal": "lingering",
        "timestamp_ms": BASE_MS,
        "vlm_captions": [],
        "severity_score": 0.5,
        "alert_id": "legacy-1",
    }

    alert = ReasoningResult(**legacy)

    assert alert.zone is None
    assert alert.zones_visited == []
    assert alert.object_labels == []


def test_provenance_fields_round_trip_through_storage(store):
    seeded = seed_alert(store, alert_id="a1", offset_ms=1_000, zone="keypad_area")

    raws = store.get_alerts_in_range(BASE_MS, BASE_MS + 10_000)
    restored = ReasoningResult(**json.loads(raws[0]))

    assert restored.zone == "keypad_area"
    assert restored.object_labels == ["person"]
    assert restored.alert_id == seeded.alert_id
