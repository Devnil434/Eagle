from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from libs.config.settings import settings

import fakeredis
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inspect_tracks import (
    _positive_int,
    build_json_payload,
    inspect_tracks,
    render_text,
)


@pytest.fixture()
def fake_redis():
    return fakeredis.FakeRedis()


def _seed_track(fake_redis, track_id: int = 3) -> None:
    fake_redis.set(
        f"track:cam_01:{track_id}",
        json.dumps(
            {
                "camera_id": "cam_01",
                "track_id": track_id,
                "global_id": f"gid-{track_id}",
                "state": "ACTIVE",
                "dwell_time_seconds": 4.1,
                "zones_present": ["safe_corridor"],
            }
        ),
    )


def _seed_events(fake_redis, track_id: int = 3) -> None:
    fake_redis.set(
        "event:cam_01:1",
        json.dumps(
            [
                {
                    "event": "BORN",
                    "track_id": track_id,
                    "frame_id": 1,
                    "timestamp_ms": 1000.0,
                    "dwell_time_seconds": 0.0,
                    "zones_present": ["safe_corridor"],
                }
            ]
        ),
    )
    fake_redis.set(
        "event:cam_01:2",
        json.dumps(
            [
                {
                    "event": "ACTIVE",
                    "track_id": track_id,
                    "frame_id": 2,
                    "timestamp_ms": 2000.0,
                    "dwell_time_seconds": 4.1,
                    "zones_present": ["safe_corridor"],
                },
                {
                    "event": "ACTIVE",
                    "track_id": 99,
                    "frame_id": 2,
                    "timestamp_ms": 2000.0,
                    "dwell_time_seconds": 1.0,
                    "zones_present": ["restricted_door"],
                },
            ]
        ),
    )


def test_inspect_tracks_lists_active_tracks(fake_redis):
    _seed_track(fake_redis)
    _seed_events(fake_redis)

    summaries = inspect_tracks(fake_redis, camera_id="cam_01")

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.track_id == 3
    assert summary.event_count == 2
    assert summary.zone == "safe_corridor"
    assert summary.action_summary == "born -> active"


def test_inspect_tracks_filters_single_track(fake_redis):
    _seed_track(fake_redis, track_id=3)
    _seed_track(fake_redis, track_id=4)
    _seed_events(fake_redis, track_id=3)

    summaries = inspect_tracks(fake_redis, camera_id="cam_01", track_id=3)

    assert [summary.track_id for summary in summaries] == [3]


def test_inspect_tracks_limits_recent_events(fake_redis):
    _seed_track(fake_redis)
    _seed_events(fake_redis)

    summaries = inspect_tracks(fake_redis, camera_id="cam_01", track_id=3, last=1)

    assert summaries[0].event_count == 1
    assert summaries[0].events[0]["frame_id"] == 2
    assert summaries[0].action_summary == "active"


def test_inspect_tracks_skips_malformed_events(fake_redis):
    _seed_track(fake_redis)
    _seed_events(fake_redis)
    fake_redis.set(
        "event:cam_01:3",
        json.dumps(
            [
                {
                    "event": "BROKEN",
                    "track_id": "not-an-int",
                    "frame_id": 3,
                    "timestamp_ms": 3000.0,
                    "zones_present": ["safe_corridor"],
                },
                {
                    "event": "STILL_ACTIVE",
                    "track_id": 3,
                    "frame_id": "4",
                    "timestamp_ms": 4000.0,
                    "zones_present": ["safe_corridor"],
                },
            ]
        ),
    )

    summaries = inspect_tracks(fake_redis, camera_id="cam_01", track_id=3)

    assert summaries[0].event_count == 3
    assert summaries[0].events[-1]["event"] == "STILL_ACTIVE"
    assert summaries[0].events[-1]["frame_id"] == 4


def test_positive_int_rejects_zero_and_negative_values():
    assert _positive_int("2") == 2

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("-1")


def test_json_payload_is_machine_readable(fake_redis):
    _seed_track(fake_redis)
    _seed_events(fake_redis)
    summaries = inspect_tracks(fake_redis, camera_id="cam_01", track_id=3)

    payload = build_json_payload(
        summaries,
        camera_id="cam_01",
        redis_url="redis://user:secret@localhost:6379/0",
    )

    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["redis_url"] == settings.REDIS_URL
    assert "secret" not in encoded
    assert decoded["tracks"][0]["track_id"] == 3
    assert decoded["tracks"][0]["events"][0]["event"] == "BORN"


def test_text_render_includes_detail_rows_for_single_track(fake_redis):
    _seed_track(fake_redis)
    _seed_events(fake_redis)
    summaries = inspect_tracks(fake_redis, camera_id="cam_01", track_id=3)

    output = render_text(
        summaries,
        camera_id="cam_01",
        redis_url=settings.REDIS_URL,
        show_event_rows=True,
    )

    assert "Track #3" in output
    assert "Redis @ localhost:6379" in output
    assert "summary: born -> active" in output
    assert "event rows:" in output
    assert "frame 2: ACTIVE" in output


def test_text_render_redacts_redis_credentials(fake_redis):
    _seed_track(fake_redis)
    summaries = inspect_tracks(fake_redis, camera_id="cam_01", track_id=3)

    output = render_text(
        summaries,
        camera_id="cam_01",
        redis_url="redis://user:secret@redis.example.com:6380/0",
        show_event_rows=False,
    )

    assert "redis.example.com:6380" in output
    assert "user" not in output
    assert "secret" not in output
