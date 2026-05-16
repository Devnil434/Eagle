"""
demo_lifecycle_mock_video.py — Manual verification script for Issue #14.

Simulates a 10-frame "mock video" WITHOUT any real video file, YOLO model,
or DeepSort dependency. Uses the same mock strategy as the unit tests but
prints everything to console so you can visually inspect the lifecycle events
and the resulting JSONL file.

Scenario:
    Frame 0-4:  One person (track #1) is visible -> BORN at frame 0
    Frame 5-9:  Person disappears -> LOST at frame 5, DEAD after max_age=3

Run:
    python tests/demo_lifecycle_mock_video.py
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# -- Setup import path --------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.schemas.detection import DetectionFrameSchema, DetectionSchema, BoundingBox
from libs.schemas.tracking import TrackState
from libs.logging.track_event_logger import TrackEventLogger


def make_det_frame(frame_id: int, boxes: list[list[float]]) -> DetectionFrameSchema:
    """Build a DetectionFrameSchema with given bounding boxes."""
    dets = [
        DetectionSchema(
            label="person",
            bbox=BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
            confidence=0.9,
        )
        for b in boxes
    ]
    return DetectionFrameSchema(
        frame_id=frame_id,
        camera_id="cam_01",
        detections=dets,
        timestamp_ms=float(frame_id * 33),
    )


def make_mock_track(tid: int, ltwh: list[float], conf: float = 0.9):
    """Create a fake DeepSort track object."""
    t = MagicMock()
    t.track_id = tid
    t.is_confirmed.return_value = True
    t.to_ltwh.return_value = np.array(ltwh)
    t.det_conf = conf
    return t


@patch("services.tracking.tracker.DeepSort")
def run_mock_video(MockDeepSort):
    from services.tracking.tracker import Tracker

    # -- Setup -----------------------------------------------------------------
    log_path = Path("data/logs/tracks.jsonl")
    if log_path.exists():
        log_path.unlink()

    event_logger = TrackEventLogger(log_path=log_path)

    mock_ds = MagicMock()
    MockDeepSort.return_value = mock_ds
    mock_ds.max_age = 3  # short max_age so DEAD fires within 10 frames

    tracker = Tracker(fps=30, event_logger=event_logger)
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # -- Simulate 10 frames ----------------------------------------------------
    print("=" * 70)
    print("  MOCK VIDEO SIMULATION -- 10 Frames")
    print("  Scenario: person appears frames 0-4, disappears frames 5-9")
    print("=" * 70)

    all_events = []

    for frame_id in range(10):
        if frame_id < 5:
            mock_ds.update_tracks.return_value = [
                make_mock_track(1, [100, 80, 50, 120])
            ]
            status = "[VISIBLE] Person in frame"
        else:
            mock_ds.update_tracks.return_value = []
            status = "[  GONE ] Person left   "

        det = make_det_frame(frame_id, [[100, 80, 150, 200]])
        result = tracker.update(det, raw_frame)
        events = tracker.drain_lifecycle_events()
        all_events.extend(events)

        event_strs = [e.event.value for e in events]
        event_display = f"  >> Events: {', '.join(event_strs)}" if events else ""
        print(f"  Frame {frame_id:2d}: {status}  |  Active tracks: {len(result.tracks)}{event_display}")

    # -- Summary ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("  LIFECYCLE EVENT SUMMARY")
    print("=" * 70)

    for e in all_events:
        d = e.to_jsonl_dict()
        print(f"  {json.dumps(d, default=str)}")

    # -- Verify JSONL file -----------------------------------------------------
    print()
    print("=" * 70)
    print(f"  JSONL FILE: {log_path.resolve()}")
    print("=" * 70)

    if log_path.exists():
        content = log_path.read_text(encoding="utf-8").strip()
        lines = content.splitlines()
        print(f"  Lines written: {len(lines)}")
        print()
        for line in lines:
            print(f"  {line}")
    else:
        print("  [!] File not found!")

    # -- Acceptance check ------------------------------------------------------
    print()
    print("=" * 70)
    print("  ACCEPTANCE CRITERIA CHECK (Issue #14)")
    print("=" * 70)

    event_types = [e.event for e in all_events]
    checks = [
        ("BORN event logged",         TrackState.BORN in event_types),
        ("LOST event logged",         TrackState.LOST in event_types),
        ("DEAD event logged",         TrackState.DEAD in event_types),
        ("JSONL file exists",         log_path.exists()),
        ("One event per line",        log_path.exists() and all(
            json.loads(l) for l in log_path.read_text().strip().splitlines()
        )),
        ("Correct order (BORN<LOST<DEAD)", (
            event_types.index(TrackState.BORN)
            < event_types.index(TrackState.LOST)
            < event_types.index(TrackState.DEAD)
        )),
    ]

    all_pass = True
    for label, passed in checks:
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} {label}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  >>> ALL ACCEPTANCE CRITERIA PASSED! <<<")
    else:
        print("  >>> SOME CHECKS FAILED <<<")
    print("=" * 70)


if __name__ == "__main__":
    run_mock_video()
