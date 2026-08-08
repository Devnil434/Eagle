"""
Unit tests for Phase 2: tracking schema validation, tracker state machine,
dwell time accumulation, and trajectory building.

All tests use mock frames and mock detections — no real video or YOLO model needed.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from libs.schemas.detection import DetectionFrameSchema, DetectionSchema, BoundingBox
from libs.schemas.tracking  import TrackedFrame, TrackedObject, TrackState, TrajectoryPoint
from services.tracking.tracker import _interpolate_trajectory


# ── Schema unit tests (no tracker needed) ────────────────────────────────────

def test_tracked_object_schema():
    obj = TrackedObject(
        track_id           = 7,
        label              = "person",
        bbox               = [100.0, 150.0, 200.0, 350.0],
        confidence         = 0.91,
        center             = (150.0, 250.0),
        dwell_time_frames  = 45,
        dwell_time_seconds = 1.5,
        state              = TrackState.ACTIVE,
        zones_present      = ["restricted_door"],
    )
    assert obj.track_id == 7
    assert obj.dwell_time_seconds == 1.5
    assert "restricted_door" in obj.zones_present


def test_tracked_frame_schema():
    frame = TrackedFrame(
        frame_id     = 100,
        camera_id    = "cam_02",
        tracks       = [],
        timestamp_ms = 999.0,
    )
    assert frame.frame_id == 100
    assert frame.tracks  == []


def test_trajectory_point_schema():
    pt = TrajectoryPoint(x=320.5, y=240.1, frame_id=55)
    assert pt.frame_id == 55


def test_track_state_enum():
    assert TrackState.BORN  == "BORN"
    assert TrackState.DEAD  == "DEAD"
    assert TrackState.LOST  == "LOST"
    assert TrackState.ACTIVE == "ACTIVE"


# ── Tracker integration tests (mock DeepSort) ─────────────────────────────────

def _make_det_frame(frame_id: int, boxes: list[list[float]]) -> DetectionFrameSchema:
    """Helper: build a DetectionFrameSchema with given bounding boxes."""
    dets = [
        DetectionSchema(
            label      = "person",
            bbox       = BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
            confidence = 0.9,
        )
        for b in boxes
    ]
    return DetectionFrameSchema(
        frame_id     = frame_id,
        camera_id    = "cam_01",
        detections   = dets,
        timestamp_ms = float(frame_id * 33),
    )


def _make_mock_track(
    tid: int,
    ltwh: list[float],
    conf: float = 0.9,
    embedding=None,
):
    t = MagicMock()
    t.track_id   = tid
    t.is_confirmed.return_value = True
    t.to_ltwh.return_value      = np.array(ltwh)
    t.det_conf   = conf
    if embedding is not None:
        t.features = [embedding]
    else:
        t.features = []
    return t


@patch("services.tracking.tracker.DeepSort")
def test_tracker_returns_tracked_frame(MockDeepSort):
    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 30
    mock_ds.update_tracks.return_value = [
        _make_mock_track(1, [100, 80, 50, 120])
    ]

    tracker   = Tracker(fps=30)
    det_frame = _make_det_frame(0, [[100, 80, 150, 200]])
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result    = tracker.update(det_frame, raw_frame)

    assert isinstance(result, TrackedFrame)
    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 1


@patch("services.tracking.tracker.DeepSort")
def test_dwell_time_accumulates(MockDeepSort):
    """Same track seen across N frames → dwell_time_frames == N."""
    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 30

    tracker   = Tracker(fps=30)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(10):
        mock_ds.update_tracks.return_value = [_make_mock_track(1, [100, 80, 50, 120])]
        det    = _make_det_frame(i, [[100, 80, 150, 200]])
        result = tracker.update(det, raw_frame)

    assert result.tracks[0].dwell_time_frames  == 10
    assert result.tracks[0].dwell_time_seconds == pytest.approx(10 / 30, abs=0.01)


@patch("services.tracking.tracker.DeepSort")
def test_trajectory_grows_and_caps(MockDeepSort):
    """Trajectory should grow each frame but cap at MAX_TRAJECTORY_LEN."""
    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 30

    tracker   = Tracker(fps=30)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(100):    # more than MAX_TRAJECTORY_LEN (80)
        mock_ds.update_tracks.return_value = [_make_mock_track(1, [100+i, 80, 50, 120])]
        result = tracker.update(_make_det_frame(i, [[100+i, 80, 150+i, 200]]), raw_frame)

    assert len(result.tracks[0].trajectory) == tracker.MAX_TRAJECTORY_LEN


@patch("services.tracking.tracker.DeepSort")
def test_born_lifecycle_event_emitted(MockDeepSort):
    """First appearance of a track_id must emit a BORN lifecycle event."""
    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 30
    mock_ds.update_tracks.return_value = [_make_mock_track(42, [100, 80, 50, 120])]

    tracker   = Tracker(fps=30)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.update(_make_det_frame(0, [[100, 80, 150, 200]]), raw_frame)

    events = tracker.drain_lifecycle_events()
    born   = [e for e in events if e.event == TrackState.BORN]
    assert len(born) == 1
    assert born[0].track_id == 42


@patch("services.tracking.tracker.DeepSort")
def test_born_only_fires_once_per_id(MockDeepSort):
    """BORN must fire exactly once per track_id, not on every frame."""
    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 30

    tracker   = Tracker(fps=30)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    all_events = []

    for i in range(5):
        mock_ds.update_tracks.return_value = [_make_mock_track(5, [100, 80, 50, 120])]
        tracker.update(_make_det_frame(i, [[100, 80, 150, 200]]), raw_frame)
        all_events += tracker.drain_lifecycle_events()

    born_events = [e for e in all_events if e.event == TrackState.BORN and e.track_id == 5]
    assert len(born_events) == 1


@patch("services.tracking.tracker.DeepSort")
def test_multiple_tracks_get_unique_ids(MockDeepSort):
    """Two people in the same frame → two different track_ids."""
    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 30
    mock_ds.update_tracks.return_value = [
        _make_mock_track(1, [50,  80, 50, 120]),
        _make_mock_track(2, [400, 80, 50, 120]),
    ]

    tracker   = Tracker(fps=30)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result    = tracker.update(
        _make_det_frame(0, [[50, 80, 100, 200], [400, 80, 450, 200]]),
        raw_frame,
    )
    ids = {t.track_id for t in result.tracks}
    assert ids == {1, 2}


@patch("services.tracking.tracker.DeepSort")
def test_lifecycle_sequence_10_frame_mock(MockDeepSort):
    """
    Acceptance criteria (Issue #14): verify BORN → LOST → DEAD event sequence
    for a 10-frame mock video.

    Scenario:
        - Frames 0–4: track #1 visible (BORN at frame 0)
        - Frames 5–9: track #1 disappears (LOST at frame 5, DEAD after max_age)
        - max_age=3 so DEAD fires at frame 5 + 3 + 1 = frame 9

    Expected lifecycle events in order: [BORN, LOST, DEAD]
    """
    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 3  # short max_age so DEAD fires within 10 frames

    tracker   = Tracker(fps=30)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    all_events = []

    for frame_id in range(10):
        if frame_id < 5:
            # Track #1 visible in frames 0–4
            mock_ds.update_tracks.return_value = [
                _make_mock_track(1, [100, 80, 50, 120])
            ]
        else:
            # Track #1 vanished in frames 5–9
            mock_ds.update_tracks.return_value = []

        det = _make_det_frame(frame_id, [[100, 80, 150, 200]])
        tracker.update(det, raw_frame)
        all_events.extend(tracker.drain_lifecycle_events())

    # Extract event types in order
    event_types = [e.event for e in all_events]
    assert TrackState.BORN in event_types, "BORN event missing"
    assert TrackState.LOST in event_types, "LOST event missing"
    assert TrackState.DEAD in event_types, "DEAD event missing"

    # Verify ordering: BORN before LOST before DEAD
    born_idx = event_types.index(TrackState.BORN)
    lost_idx = event_types.index(TrackState.LOST)
    dead_idx = event_types.index(TrackState.DEAD)
    assert born_idx < lost_idx < dead_idx, (
        f"Wrong order: BORN@{born_idx}, LOST@{lost_idx}, DEAD@{dead_idx}"
    )

    # Verify all events belong to track #1
    for e in all_events:
        assert e.track_id == 1


@patch("services.tracking.tracker.DeepSort")
def test_lifecycle_logging_integration(MockDeepSort, tmp_path):
    """
    End-to-end: Tracker with event_logger writes JSONL file with correct schema.
    """
    import json
    from services.tracking.tracker import Tracker
    from libs.logging.track_event_logger import TrackEventLogger

    log_file = tmp_path / "tracks.jsonl"
    event_logger = TrackEventLogger(log_path=log_file)

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 2

    tracker   = Tracker(fps=30, event_logger=event_logger)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 0: track appears → BORN
    mock_ds.update_tracks.return_value = [_make_mock_track(10, [100, 80, 50, 120])]
    tracker.update(_make_det_frame(0, [[100, 80, 150, 200]]), raw_frame)

    # Frames 1–4: track vanishes → LOST at frame 1, DEAD at frame 1 + 2 + 1 = 4
    for fid in range(1, 5):
        mock_ds.update_tracks.return_value = []
        tracker.update(_make_det_frame(fid, []), raw_frame)

    # Verify JSONL file
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

    event_names = [r["event"] for r in records]
    assert "BIRTH" in event_names
    assert "LOST" in event_names
    assert "DEAD" in event_names

    # Verify BIRTH record has zone field
    birth_rec = next(r for r in records if r["event"] == "BIRTH")
    assert "zone" in birth_rec
    assert "track_id" in birth_rec
    assert birth_rec["track_id"] == 10

@patch("services.tracking.tracker.DeepSort")
def test_reid_restores_original_id(MockDeepSort):

    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 30

    tracker = Tracker(fps=30)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    embedding = np.array([0.1, 0.2, 0.3])

    # Frame 0: original track
    mock_ds.update_tracks.return_value = [
        _make_mock_track(
            1,
            [100, 80, 50, 120],
            embedding=embedding,
        )
    ]

    result1 = tracker.update(
        _make_det_frame(0, [[100, 80, 150, 200]]),
        raw_frame,
    )

    original_id = result1.tracks[0].track_id

    # Frame 1: track disappears
    mock_ds.update_tracks.return_value = []

    tracker.update(
        _make_det_frame(1, []),
        raw_frame,
    )

    # Frame 2: same person reappears with NEW tracker ID
    mock_ds.update_tracks.return_value = [
        _make_mock_track(
            99,
            [102, 82, 50, 120],
            embedding=embedding,
        )
    ]

    result2 = tracker.update(
        _make_det_frame(2, [[102, 82, 152, 202]]),
        raw_frame,
    )

    restored_id = result2.tracks[0].track_id

    assert restored_id == original_id

@patch("services.tracking.tracker.DeepSort")
def test_reid_rejects_low_similarity(MockDeepSort):

    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 30

    tracker = Tracker(fps=30)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])

    # Original track
    mock_ds.update_tracks.return_value = [
        _make_mock_track(
            1,
            [100, 80, 50, 120],
            embedding=emb1,
        )
    ]

    tracker.update(
        _make_det_frame(0, [[100, 80, 150, 200]]),
        raw_frame,
    )

    # Disappear
    mock_ds.update_tracks.return_value = []
    tracker.update(
        _make_det_frame(1, []),
        raw_frame,
    )

    # Reappear with different embedding
    mock_ds.update_tracks.return_value = [
        _make_mock_track(
            99,
            [100, 80, 50, 120],
            embedding=emb2,
        )
    ]

    result = tracker.update(
        _make_det_frame(2, [[100, 80, 150, 200]]),
        raw_frame,
    )

    assert result.tracks[0].track_id != 1

@patch("services.tracking.tracker.DeepSort")
def test_reid_expires_after_max_age(MockDeepSort):

    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 2

    tracker = Tracker(fps=30, max_age=2)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    embedding = np.array([0.1, 0.2, 0.3])

    # Original track
    mock_ds.update_tracks.return_value = [
        _make_mock_track(
            1,
            [100, 80, 50, 120],
            embedding=embedding,
        )
    ]

    tracker.update(
        _make_det_frame(0, [[100, 80, 150, 200]]),
        raw_frame,
    )

    # Track disappears for longer than max_age
    for fid in range(1, 5):

        mock_ds.update_tracks.return_value = []

        tracker.update(
            _make_det_frame(fid, []),
            raw_frame,
        )

    # Reappears with same embedding
    mock_ds.update_tracks.return_value = [
        _make_mock_track(
            99,
            [100, 80, 50, 120],
            embedding=embedding,
        )
    ]

    result = tracker.update(
        _make_det_frame(5, [[100, 80, 150, 200]]),
        raw_frame,
    )

    # Should NOT restore old ID
    assert result.tracks[0].track_id == 99


def test_interpolate_trajectory_success():
    """Test standard linear interpolation for a 3-frame gap including width and height scaling."""
    last_pos = {"x": 10.0, "y": 20.0, "w": 50.0, "h": 50.0}
    new_pos = {"x": 50.0, "y": 60.0, "w": 90.0, "h": 90.0}
    gap_frames = 3
    start_frame = 101

    result = _interpolate_trajectory(last_pos, new_pos, gap_frames, start_frame)

    assert len(result) == 3
    
    # Assert step progression and metadata across all items
    expected_values = [
        {"frame_id": 101, "x": 20.0, "y": 30.0, "w": 60.0, "h": 60.0},
        {"frame_id": 102, "x": 30.0, "y": 40.0, "w": 70.0, "h": 70.0},
        {"frame_id": 103, "x": 40.0, "y": 50.0, "w": 80.0, "h": 80.0},
    ]

    for idx, expected in enumerate(expected_values):
        assert result[idx]["frame_id"] == expected["frame_id"]
        assert result[idx]["interpolated"] is True
        assert result[idx]["x"] == expected["x"]
        assert result[idx]["y"] == expected["y"]
        assert result[idx]["w"] == expected["w"]
        assert result[idx]["h"] == expected["h"]

def test_interpolate_trajectory_no_gap():
    last_pos = {"x": 10, "y": 20}
    new_pos = {"x": 20, "y": 30}
    assert _interpolate_trajectory(last_pos, new_pos, 0, 100) == []

def test_interpolate_trajectory_no_movement():
    last_pos = {"x": 100.0, "y": 100.0}
    new_pos = {"x": 100.0, "y": 100.0}
    gap_frames = 2
    start_frame = 50
    result = _interpolate_trajectory(last_pos, new_pos, gap_frames, start_frame)
    assert len(result) == 2
    assert result[0]["x"] == 100.0