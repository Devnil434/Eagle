"""
run_pipeline.py — Integration demo for Phases 1+2+3+action recognition.

Runs detection → tracking → temporal action recognition → memory.

CLI:
    python scripts/run_pipeline.py --source 0
    python scripts/run_pipeline.py --source data/sample_videos/sample.mp4
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import redis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.action_recognition.inference import ActionRecognizer
from services.detection.detection import Detector
from services.memory.memory import MemoryService, MemoryStore
from services.memory.pipeline import process_tracked_frame
from services.tracking.cross_camera_reid import CrossCameraReID
from services.tracking.tracker import Tracker
from services.tracking.visualizer import draw_tracks

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline")


@dataclass
class PipelineResult:
    processed_frames: int = 0
    events: list[Any] = field(default_factory=list)
    action_summary: str = ""

    def __post_init__(self) -> None:
        if not self.action_summary and self.events:
            actions = []
            for e in self.events:
                label = getattr(e, "temporal_action", None) or getattr(e, "action_hint", None)
                if label is not None:
                    val = label.value if hasattr(label, "value") else str(label)
                    actions.append(val)
            unique: list[str] = []
            for a in actions:
                if not unique or unique[-1] != a:
                    unique.append(a)
            self.action_summary = " → ".join(unique) if unique else "unknown"


def process_frames(
    frames: list[np.ndarray],
    *,
    detector: Detector,
    tracker: Tracker,
    memory_service: MemoryService,
    memory_store: Optional[MemoryStore] = None,
    action_recognizer: Optional[ActionRecognizer] = None,
    camera_id: str = "cam_01",
) -> PipelineResult:
    """Process a list of frames through detection, tracking, actions, and memory."""
    store = memory_store or MemoryStore(redis_client=memory_service._r)
    all_events: list[Any] = []

    for frame_id, frame in enumerate(frames):
        det_frame = detector.detect(frame, frame_id=frame_id)
        det_frame.camera_id = camera_id
        tracked_frame = tracker.update(det_frame, frame)
        tracked_frame.camera_id = camera_id

        events = process_tracked_frame(
            tracked_frame,
            store,
            raw_frame=frame,
            action_recognizer=action_recognizer,
            memory_service=memory_service,
        )
        all_events.extend(events)

        for evt in tracker.drain_lifecycle_events():
            embedding = tracker._active_embeddings.get(evt.track_id)
            memory_service.handle_lifecycle_event(evt, embedding=embedding)

    return PipelineResult(
        processed_frames=len(frames),
        events=all_events,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Eagle surveillance pipeline")
    parser.add_argument("--source", default="0", help="Video path or camera index")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--camera-id", default="cam_01")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    redis_client = redis.Redis()
    reid = CrossCameraReID(redis_client)
    memory_service = MemoryService(redis_client, reid)
    memory_store = MemoryStore(redis_client=redis_client)

    detector = Detector(model_name=args.model)
    tracker = Tracker(fps=fps, camera_id=args.camera_id)
    action_recognizer = ActionRecognizer()

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det_frame = detector.detect(frame, frame_id=frame_id)
        det_frame.camera_id = args.camera_id
        tracked_frame = tracker.update(det_frame, frame)
        tracked_frame.camera_id = args.camera_id

        events = process_tracked_frame(
            tracked_frame,
            memory_store,
            raw_frame=frame,
            action_recognizer=action_recognizer,
            memory_service=memory_service,
        )

        annotated = draw_tracks(frame, tracked_frame)
        for evt in events:
            if not getattr(evt, "temporal_action", None):
                continue
            for t in tracked_frame.tracks:
                if t.track_id != evt.track_id:
                    continue
                x1, _, _, y2 = [int(v) for v in t.bbox]
                conf = getattr(evt, "temporal_action_confidence", 0.0)
                label = f"{evt.temporal_action} ({conf:.2f})"
                cv2.putText(
                    annotated, label, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                )
                break

        for evt in tracker.drain_lifecycle_events():
            embedding = tracker._active_embeddings.get(evt.track_id)
            memory_service.handle_lifecycle_event(evt, embedding=embedding)

        # Log sequences every 90 frames (~3s)
        if frame_id % 90 == 0:
            for track in tracked_frame.tracks:
                seq = memory_store.get_sequence(track.track_id, tracked_frame.camera_id)
                action_info = ""
                if events and getattr(events[-1], "temporal_action", None):
                    action_info = f" | temporal={events[-1].temporal_action} | hint={getattr(events[-1].action_hint, 'value', str(getattr(events[-1], 'action_hint', '')))}"
                
                logger.info(
                    f"Track #{track.track_id} | events={len(seq.events)} | "
                    f"summary={seq.action_summary} | dwell={seq.total_dwell:.1f}s{action_info}"
                )

        if not args.no_display:
            cv2.imshow("Eagle — Detection + Tracking + Actions", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

def process_frames(frames, detector=None, tracker=None, memory_service=None, show: bool = False):
    """Process a list of frames through detection, tracking, and memory.

    Returns a simple result object with attributes used by tests.
    """
    from dataclasses import dataclass

    @dataclass
    class Result:
        processed_frames: int = 0
        events: list = None
        action_summary: str = ""

    det = detector or Detector()
    trk = tracker or Tracker(fps=30)
    mem = memory_service or MemoryStore()

    res = Result(processed_frames=0, events=[])

    frame_id = 0
    for frame in frames:
        det_frame = det.detect(frame, frame_id=frame_id)
        tracked_frame = trk.update(det_frame, frame)
        # If memory_service implements handle_lifecycle_event (MemoryService),
        # use lifecycle events emitted by the tracker. Otherwise fall back to
        # the legacy process_tracked_frame which expects a MemoryStore.
        if hasattr(mem, "handle_lifecycle_event"):
            # Drain lifecycle events from tracker and store via MemoryService
            for evt in trk.drain_lifecycle_events():
                mem.handle_lifecycle_event(evt, embedding=None)
                res.events.append(evt)
        else:
            evts = process_tracked_frame(tracked_frame, mem)
            res.events.extend(evts or [])
        res.processed_frames += 1
        if tracked_frame.tracks:
            # simple action summary from first track sequence
            first_tid = tracked_frame.tracks[0].track_id
            if hasattr(mem, "get_sequence"):
                seq = mem.get_sequence(first_tid)
            elif hasattr(mem, "_r"):
                # MemoryService: create a temporary MemoryStore using same Redis
                from services.memory.memory import MemoryStore

                tmp = MemoryStore(redis_client=mem._r)
                seq = tmp.get_sequence(first_tid)
            else:
                seq = None

            if seq and getattr(seq, "action_summary", ""):
                res.action_summary = seq.action_summary
            else:
                # Fallback: build a simple summary from lifecycle events
                if res.events:
                    actions = [e.event.value for e in res.events]
                    unique = []
                    for a in actions:
                        if not unique or unique[-1] != a:
                            unique.append(a)
                    res.action_summary = " -> ".join(unique)
        frame_id += 1
    return res


if __name__ == "__main__":
    cap.release()
    cv2.destroyAllWindows()