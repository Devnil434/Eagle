from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libs.schemas.detection import BoundingBox, DetectionFrameSchema, DetectionSchema
from scripts.run_pipeline import process_frames
from services.memory.memory import MemoryService
from services.tracking.cross_camera_reid import CrossCameraReID
from services.tracking.tracker import Tracker


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> DetectionFrameSchema:
        self.calls += 1
        return DetectionFrameSchema(
            frame_id=frame_id,
            camera_id="cam_01",
            detections=[
                DetectionSchema(
                    label="person",
                    bbox=BoundingBox(x1=100, y1=80, x2=150, y2=200),
                    confidence=0.9,
                )
            ],
        )


def _make_mock_track(track_id: int):
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = True
    track.to_ltwh.return_value = np.array([100, 80, 50, 120])
    track.det_conf = 0.9
    track.features = []
    return track


@pytest.fixture()
def fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeRedis()


@patch("services.tracking.tracker.DeepSort")
def test_full_p1_p2_p3_pipeline_with_synthetic_frame(MockDeepSort, fake_redis):
    mock_deepsort = MagicMock()
    MockDeepSort.return_value = mock_deepsort
    mock_deepsort.max_age = 30
    mock_deepsort.update_tracks.return_value = [_make_mock_track(7)]

    detector = FakeDetector()
    tracker = Tracker(fps=30, camera_id="cam_01")
    reid = CrossCameraReID(fake_redis)
    memory = MemoryService(fake_redis, reid)
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]

    result = process_frames(
        frames,
        detector=detector,
        tracker=tracker,
        memory_service=memory,
    )

    track_record = memory.get_track_record("cam_01", 7)
    event_keys = fake_redis.keys("event:cam_01:*")

    assert detector.calls == 1
    assert mock_deepsort.update_tracks.called
    assert result.processed_frames == 1
    assert result.events
    assert result.events[0].track_id == 7
    assert result.action_summary
    assert track_record is not None
    assert track_record["track_id"] == 7
    assert fake_redis.exists("track:cam_01:7")
    assert event_keys
    stored_events = json.loads(fake_redis.get(event_keys[0]))
    assert stored_events
    assert stored_events[0]["track_id"] == 7
    assert stored_events[0]["event"] == "BORN"
