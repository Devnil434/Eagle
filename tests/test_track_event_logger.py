"""
Unit tests for TrackEventLogger and TrackLifecycleEvent.to_jsonl_dict().

Validates:
- JSONL file creation and content
- Issue #14 schema compliance (BIRTH/LOST/DEAD field names)
- Batch logging
- Edge cases (empty zones, zero dwell time)
"""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path


from libs.schemas.tracking import TrackLifecycleEvent, TrackState
from libs.logging.track_event_logger import TrackEventLogger


# ── to_jsonl_dict() tests ────────────────────────────────────────────────────

def test_born_maps_to_birth():
    """Internal BORN must serialize as 'BIRTH' per issue #14 spec."""
    evt = TrackLifecycleEvent(
        event=TrackState.BORN,
        track_id=5,
        frame_id=0,
        zones_present=["restricted_door"],
        timestamp_ms=1700000000000.0,
    )
    d = evt.to_jsonl_dict()
    assert d["event"] == "BIRTH"
    assert d["track_id"] == 5
    assert "zone" in d
    assert d["zone"] == "restricted_door"
    assert "timestamp" in d
    # BIRTH should NOT have dwell_time_sec
    assert "dwell_time_sec" not in d


def test_lost_includes_dwell_time():
    """LOST events must include dwell_time_sec."""
    evt = TrackLifecycleEvent(
        event=TrackState.LOST,
        track_id=5,
        frame_id=10,
        dwell_time_seconds=18.33,
        timestamp_ms=1700000001000.0,
    )
    d = evt.to_jsonl_dict()
    assert d["event"] == "LOST"
    assert d["dwell_time_sec"] == 18.33
    # LOST should NOT have zone
    assert "zone" not in d


def test_dead_is_minimal():
    """DEAD events have only event, track_id, and timestamp."""
    evt = TrackLifecycleEvent(
        event=TrackState.DEAD,
        track_id=5,
        frame_id=40,
        timestamp_ms=1700000002000.0,
    )
    d = evt.to_jsonl_dict()
    assert d["event"] == "DEAD"
    assert d["track_id"] == 5
    assert "timestamp" in d
    assert "zone" not in d
    assert "dwell_time_sec" not in d


def test_born_empty_zones_fallback():
    """When zones_present is empty, zone should be 'unknown'."""
    evt = TrackLifecycleEvent(
        event=TrackState.BORN,
        track_id=1,
        frame_id=0,
        zones_present=[],
        timestamp_ms=1700000000000.0,
    )
    d = evt.to_jsonl_dict()
    assert d["zone"] == "unknown"


def test_timestamp_is_iso8601():
    """Timestamp must be ISO 8601 format."""
    evt = TrackLifecycleEvent(
        event=TrackState.BORN,
        track_id=1,
        frame_id=0,
        zones_present=["lobby"],
        timestamp_ms=1700000000000.0,
    )
    d = evt.to_jsonl_dict()
    # Must contain 'T' separator and timezone info
    assert "T" in d["timestamp"]
    assert "+" in d["timestamp"] or "Z" in d["timestamp"]


# ── TrackEventLogger tests ───────────────────────────────────────────────────

def test_jsonl_file_created_and_written(tmp_path: Path):
    """Logger must create the JSONL file and write valid JSON lines."""
    log_file = tmp_path / "tracks.jsonl"
    tl = TrackEventLogger(log_path=log_file)

    evt = TrackLifecycleEvent(
        event=TrackState.BORN,
        track_id=7,
        frame_id=0,
        zones_present=["restricted_door"],
        timestamp_ms=1700000000000.0,
    )
    tl.log_event(evt)

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["event"] == "BIRTH"
    assert record["track_id"] == 7
    assert record["zone"] == "restricted_door"


def test_log_batch(tmp_path: Path):
    """log_batch() must write multiple lines."""
    log_file = tmp_path / "tracks.jsonl"
    tl = TrackEventLogger(log_path=log_file)

    events = [
        TrackLifecycleEvent(
            event=TrackState.BORN, track_id=1, frame_id=0,
            zones_present=["door"], timestamp_ms=1700000000000.0,
        ),
        TrackLifecycleEvent(
            event=TrackState.LOST, track_id=1, frame_id=5,
            dwell_time_seconds=5.0, timestamp_ms=1700000001000.0,
        ),
        TrackLifecycleEvent(
            event=TrackState.DEAD, track_id=1, frame_id=10,
            timestamp_ms=1700000002000.0,
        ),
    ]
    tl.log_batch(events)

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    parsed = [json.loads(line) for line in lines]
    assert [r["event"] for r in parsed] == ["BIRTH", "LOST", "DEAD"]


def test_jsonl_append_mode(tmp_path: Path):
    """Multiple log_event() calls must append, not overwrite."""
    log_file = tmp_path / "tracks.jsonl"
    tl = TrackEventLogger(log_path=log_file)

    for i in range(3):
        evt = TrackLifecycleEvent(
            event=TrackState.BORN, track_id=i, frame_id=i,
            zones_present=["zone_a"], timestamp_ms=1700000000000.0 + i * 1000,
        )
        tl.log_event(evt)

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_parent_dirs_created(tmp_path: Path):
    """Logger must create parent directories if they don't exist."""
    log_file = tmp_path / "deep" / "nested" / "tracks.jsonl"
    tl = TrackEventLogger(log_path=log_file)

    evt = TrackLifecycleEvent(
        event=TrackState.DEAD, track_id=99, frame_id=0,
        timestamp_ms=1700000000000.0,
    )
    tl.log_event(evt)

    assert log_file.exists()
