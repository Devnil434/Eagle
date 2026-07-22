"""Unit tests for temporal action recognition."""
from __future__ import annotations

import numpy as np

from libs.schemas.action_recognition import ActionLabel
from libs.schemas.tracking import TrackedFrame, TrackedObject, TrajectoryPoint
from services.action_recognition.inference import ActionRecognizer, HeuristicClassifier
from services.action_recognition.temporal_buffer import TemporalBuffer


def _make_track(track_id: int, step_x: float = 5.0, step_y: float = 0.0) -> TrackedObject:
    pts = [
        TrajectoryPoint(x=100 + i * step_x, y=100 + i * step_y, frame_id=i)
        for i in range(5)
    ]
    return TrackedObject(
        track_id=track_id,
        label="person",
        bbox=[90.0, 80.0, 160.0, 200.0],
        center=(125.0, 140.0),
        confidence=0.9,
        trajectory=pts,
        dwell_time_seconds=12.0,
        zones_present=[],
    )


def test_temporal_buffer_requires_full_sequence():
    buf = TemporalBuffer(seq_len=4)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox = [10.0, 10.0, 80.0, 120.0]
    for _ in range(3):
        buf.add_frame(1, frame, bbox)
    assert buf.get_sequence(1) is None
    buf.add_frame(1, frame, bbox)
    seq = buf.get_sequence(1)
    assert seq is not None
    assert len(seq) == 4


def test_heuristic_running():
    obj = _make_track(1, step_x=20.0, step_y=0.0)
    pred = HeuristicClassifier().classify(obj)
    assert pred.action == ActionLabel.RUNNING
    assert pred.source == "heuristic"


def test_action_recognizer_heuristic_mode():
    recognizer = ActionRecognizer(model_path="/nonexistent/model.onnx")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracked = TrackedFrame(
        frame_id=1,
        camera_id="cam_01",
        tracks=[_make_track(7)],
        timestamp_ms=0.0,
    )
    result = recognizer.update(tracked, frame)
    assert len(result.predictions) == 1
    assert result.predictions[0].track_id == 7
