from pathlib import Path

import pytest

from scripts.visualize_timeline import (
    TrackSequenceEvent,
    create_synthetic_track_sequence,
    render_timeline,
    _zone_color,
    ZONE_COLORS,
)


# ── existing test ────────────────────────────────────────────────────────────

def test_render_timeline_creates_png(tmp_path: Path):
    events = create_synthetic_track_sequence(track_count=5)
    output_file = tmp_path / "timeline.png"
    result_path = render_timeline(events, output_file)
    assert result_path.exists()
    assert result_path.suffix == ".png"


# ── synthetic data tests ─────────────────────────────────────────────────────

def test_synthetic_data_returns_five_tracks():
    events = create_synthetic_track_sequence(track_count=5)
    track_ids = {e.track_id for e in events}
    assert len(track_ids) == 5


def test_synthetic_data_track_ids_are_correct():
    events = create_synthetic_track_sequence(track_count=5)
    track_ids = sorted({e.track_id for e in events})
    assert track_ids == [1, 2, 3, 4, 5]


def test_synthetic_data_events_per_track():
    events = create_synthetic_track_sequence(track_count=5)
    for track_id in range(1, 6):
        track_events = [e for e in events if e.track_id == track_id]
        assert len(track_events) == 3


def test_synthetic_data_zones_are_valid():
    events = create_synthetic_track_sequence(track_count=5)
    valid_zones = {"safe", "restricted"}
    for event in events:
        assert event.zone in valid_zones


def test_synthetic_data_timestamps_are_ordered():
    events = create_synthetic_track_sequence(track_count=5)
    for event in events:
        assert event.start_seconds < event.end_seconds


# ── zone color tests ─────────────────────────────────────────────────────────

def test_zone_color_restricted():
    assert _zone_color("restricted") == ZONE_COLORS["restricted"]


def test_zone_color_safe():
    assert _zone_color("safe") == ZONE_COLORS["safe"]


def test_zone_color_unknown_zone_returns_default():
    assert _zone_color("random_zone") == ZONE_COLORS["unknown"]


def test_zone_color_case_insensitive():
    assert _zone_color("RESTRICTED") == ZONE_COLORS["restricted"]
    assert _zone_color("SAFE") == ZONE_COLORS["safe"]


# ── render_timeline edge case tests ──────────────────────────────────────────

def test_render_timeline_empty_events_raises():
    with pytest.raises(ValueError, match="No timeline events provided"):
        render_timeline([], Path("output.png"))


def test_render_timeline_single_track(tmp_path: Path):
    events = create_synthetic_track_sequence(track_count=1)
    output_file = tmp_path / "single.png"
    result = render_timeline(events, output_file)
    assert result.exists()


def test_render_timeline_creates_parent_dirs(tmp_path: Path):
    events = create_synthetic_track_sequence(track_count=2)
    output_file = tmp_path / "nested" / "dir" / "timeline.png"
    result = render_timeline(events, output_file)
    assert result.exists()


def test_render_timeline_output_is_png_bytes(tmp_path: Path):
    events = create_synthetic_track_sequence(track_count=2)
    output_file = tmp_path / "timeline.png"
    render_timeline(events, output_file)
    # PNG files start with the PNG magic bytes
    with open(output_file, "rb") as f:
        header = f.read(8)
    assert header[:4] == b"\x89PNG"


def test_render_timeline_works_with_no_action_hints(tmp_path: Path):
    events = [
        TrackSequenceEvent(
            track_id=1,
            start_seconds=0.0,
            end_seconds=5.0,
            zone="safe",
            action_hint=None,
        )
    ]
    output_file = tmp_path / "no_hints.png"
    result = render_timeline(events, output_file)
    assert result.exists()


def test_render_timeline_with_ten_tracks(tmp_path: Path):
    events = create_synthetic_track_sequence(track_count=10)
    output_file = tmp_path / "ten_tracks.png"
    result = render_timeline(events, output_file)
    assert result.exists()