"""
Unit tests for Phase 2: tracking schema validation, tracker state machine,
dwell time accumulation, and trajectory building.

All tests use mock frames and mock detections — no real video or YOLO model needed.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from libs.config.settings import settings

from libs.schemas.detection import DetectionFrameSchema, DetectionSchema, BoundingBox
from libs.schemas.tracking  import TrackedFrame, TrackedObject, TrackState, TrajectoryPoint


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


def _make_mock_track(tid: int, ltwh: list[float], conf: float = 0.9):
    t = MagicMock()
    t.track_id   = tid
    t.is_confirmed.return_value = True
    t.to_ltwh.return_value      = np.array(ltwh)
    t.det_conf   = conf
    return t


@patch("services.tracking.tracker.DeepSort")
def test_tracker_returns_tracked_frame(MockDeepSort):
    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = settings.tracker_max_age
    mock_ds.update_tracks.return_value = [
        _make_mock_track(1, [100, 80, 50, 120])
    ]

    tracker   = Tracker(fps=settings.tracker_max_age)
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
    mock_ds.max_age = settings.tracker_max_age

    tracker   = Tracker(fps=settings.tracker_max_age)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(10):
        mock_ds.update_tracks.return_value = [_make_mock_track(1, [100, 80, settings.max_events_per_track, 120])]
        det    = _make_det_frame(i, [[100, 80, 150, 200]])
        result = tracker.update(det, raw_frame)

    assert result.tracks[0].dwell_time_frames  == 10
    assert result.tracks[0].dwell_time_seconds == pytest.approx(10 / settings.tracker_max_age, abs=0.01)


@patch("services.tracking.tracker.DeepSort")
def test_trajectory_grows_and_caps(MockDeepSort):
    """Trajectory should grow each frame but cap at MAX_TRAJECTORY_LEN."""
    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = settings.tracker_max_age

    tracker   = Tracker(fps=settings.tracker_max_age)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(100):    # more than MAX_TRAJECTORY_LEN (80)
        mock_ds.update_tracks.return_value = [_make_mock_track(1, [100+i, 80, settings.max_events_per_track, 120])]
        result = tracker.update(_make_det_frame(i, [[100+i, 80, 150+i, 200]]), raw_frame)

    assert len(result.tracks[0].trajectory) == tracker.MAX_TRAJECTORY_LEN


@patch("services.tracking.tracker.DeepSort")
def test_born_lifecycle_event_emitted(MockDeepSort):
    """First appearance of a track_id must emit a BORN lifecycle event."""
    from services.tracking.tracker import Tracker

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = settings.tracker_max_age
    mock_ds.update_tracks.return_value = [_make_mock_track(42, [100, 80, settings.max_events_per_track, 120])]

    tracker   = Tracker(fps=settings.tracker_max_age)
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
    mock_ds.max_age = settings.tracker_max_age

    tracker   = Tracker(fps=settings.tracker_max_age)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    all_events = []

    for i in range(5):
        mock_ds.update_tracks.return_value = [_make_mock_track(5, [100, 80, settings.max_events_per_track, 120])]
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
    mock_ds.max_age = settings.tracker_max_age
    mock_ds.update_tracks.return_value = [
        _make_mock_track(1, [settings.max_events_per_track,  80, settings.max_events_per_track, 120]),
        _make_mock_track(2, [400, 80, settings.max_events_per_track, 120]),
    ]

    tracker   = Tracker(fps=settings.tracker_max_age)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result    = tracker.update(
        _make_det_frame(0, [[settings.max_events_per_track, 80, 100, 200], [400, 80, 450, 200]]),
        raw_frame,
    )
    ids = {t.track_id for t in result.tracks}
    assert ids == {1, 2}