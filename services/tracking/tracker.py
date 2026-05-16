"""
tracker.py — Wraps deep-sort-realtime (ByteTrack-style) to assign persistent
track IDs to YOLO detections coming from Phase 1.

Usage (standalone):
    from services.tracking.tracker import Tracker
    tracker = Tracker(fps=30)
    tracked_frame = tracker.update(detection_frame, raw_frame)

Usage (CLI demo):
    python tracker.py --source data/sample_videos/sample.mp4
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from libs.config.settings import settings

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from libs.schemas.detection import DetectionFrameSchema
from libs.schemas.tracking import (
    TrackedObject,
    TrackedFrame,
    TrackState,
    TrajectoryPoint,
    TrackLifecycleEvent,
)
from libs.observability.metrics import (
    active_tracks,
    frames_processed_total,
    track_dwell_seconds,
)
from libs.logging.track_event_logger import TrackEventLogger
from services.detection.zones import get_zones_for_point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tracker:
    """
    Stateful multi-object tracker. One instance per camera feed.
    """

    MAX_TRAJECTORY_LEN = 80
    FPS_DEFAULT = 30

    def __init__(
        self,
        fps: float = FPS_DEFAULT,
        max_age: int = settings.tracker_max_age,
        n_init: int = settings.tracker_n_init,
        max_cosine_distance: float = 0.4,
        camera_id: str = "cam_01",
        event_logger: TrackEventLogger | None = None,
        reid_similarity_threshold: float = 0.85,
    ) -> None:
        self.fps = fps
        self.camera_id = camera_id
        self.max_age = max_age
        self.REID_SIMILARITY_THRESHOLD = reid_similarity_threshold

        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=100,
        )

        self._active_tracks: dict[int, TrackedObject] = {}
        self._known_ids: set[int] = set()
        self._frame_id: int = 0
        self._lifecycle_queue: list[TrackLifecycleEvent] = []
        self._event_logger: TrackEventLogger | None = event_logger
        self._lost_embeddings: dict[int, dict] = {}
        self._active_embeddings: dict[int, np.ndarray] = {}

    # ── Public API ─────────────────────────────────────────────

    def update(
        self,
        det_frame: DetectionFrameSchema,
        raw_frame: np.ndarray,
    ) -> TrackedFrame:

        self._frame_id = det_frame.frame_id
        frames_processed_total.inc()

        ds_input = []
        for det in det_frame.detections:
            if det.label != "person":
                continue
            b = det.bbox
            left, top = b.x1, b.y1
            w, h = b.x2 - b.x1, b.y2 - b.y1
            ds_input.append(([left, top, w, h], float(det.confidence), "person"))

        raw_tracks = self._tracker.update_tracks(ds_input, frame=raw_frame)

        current_ids: set[int] = set()
        tracked_objects: list[TrackedObject] = []

        for t in raw_tracks:
            if not t.is_confirmed():
                continue

            tid = int(t.track_id)

            if hasattr(t, "features") and t.features:
                new_embedding = t.features[-1]
                self._active_embeddings[tid] = new_embedding

                for lost_id, data in list(self._lost_embeddings.items()):
                    age = self._frame_id - data["last_seen"]
                    if age > self.max_age:
                        continue

                    similarity = self._cosine_similarity(
                        new_embedding,
                        data["embedding"],
                    )

                    if similarity > self.REID_SIMILARITY_THRESHOLD:
                        tid = lost_id
                        t.track_id = lost_id
                        del self._lost_embeddings[lost_id]
                        logger.info("ReID matched: restored track #%s", lost_id)
                        break

            ltwh = t.to_ltwh()
            x1 = float(ltwh[0])
            y1 = float(ltwh[1])
            x2 = x1 + float(ltwh[2])
            y2 = y1 + float(ltwh[3])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            zones = [z.name for z in get_zones_for_point(cx, cy)]

            if tid not in self._known_ids:
                self._known_ids.add(tid)
                self._emit_lifecycle(TrackState.BORN, tid, zones, 0.0)
                logger.info(f"Track BORN: #{tid} zones={zones}")

            prev = self._active_tracks.get(tid)
            dwell_frames = (prev.dwell_time_frames + 1) if prev else 1
            dwell_secs = dwell_frames / self.fps

            prev_traj = prev.trajectory if prev else []
            new_point = TrajectoryPoint(x=cx, y=cy, frame_id=self._frame_id)
            trajectory = (prev_traj + [new_point])[-self.MAX_TRAJECTORY_LEN:]

            obj = TrackedObject(
                track_id=tid,
                label="person",
                bbox=[x1, y1, x2, y2],
                confidence=float(t.det_conf or 0.0),
                center=(cx, cy),
                dwell_time_frames=dwell_frames,
                dwell_time_seconds=round(dwell_secs, 2),
                state=TrackState.ACTIVE,
                trajectory=trajectory,
                zones_present=zones,
                last_seen_frame=self._frame_id,
            )

            self._active_tracks[tid] = obj
            current_ids.add(tid)
            tracked_objects.append(obj)

        active_tracks.set(len(tracked_objects))

        for obj in tracked_objects:
            track_dwell_seconds.observe(obj.dwell_time_seconds)

        for tid, prev_obj in list(self._active_tracks.items()):
            if tid not in current_ids:
                frames_since = self._frame_id - prev_obj.last_seen_frame

                self._emit_lifecycle(
                    TrackState.LOST,
                    tid,
                    prev_obj.zones_present,
                    prev_obj.dwell_time_seconds,
                )

                if frames_since >= self.max_age:
                    self._emit_lifecycle(
                        TrackState.DEAD,
                        tid,
                        prev_obj.zones_present,
                        prev_obj.dwell_time_seconds,
                    )
                    del self._active_tracks[tid]
                    self._active_embeddings.pop(tid, None)
                    logger.info(f"Track DEAD: #{tid}")

        return TrackedFrame(
            frame_id=self._frame_id,
            camera_id=self.camera_id,
            tracks=tracked_objects,
            timestamp_ms=time.time() * 1000,
            fps=self.fps,
        )

    def drain_lifecycle_events(self) -> list[TrackLifecycleEvent]:
        events = list(self._lifecycle_queue)
        self._lifecycle_queue.clear()
        return events

    def _emit_lifecycle(self, state, track_id, zones, dwell_secs):
        event = TrackLifecycleEvent(
            event=state,
            track_id=track_id,
            frame_id=self._frame_id,
            camera_id=self.camera_id,
            zones_present=zones,
            dwell_time_seconds=dwell_secs,
            timestamp_ms=time.time() * 1000,
        )
        self._lifecycle_queue.append(event)
        if self._event_logger:
            self._event_logger.log_event(event)

    def _cosine_similarity(self, a, b):
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)


# ─── CLI ─────────────────────────────────────────────

def main():
    from services.detection.detector import Detector
    from services.tracking.visualizer import draw_tracks

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default=settings.yolo_model)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    detector = Detector(model_name=args.model)

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or Tracker.FPS_DEFAULT
    tracker = Tracker(fps=fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det_frame = detector.detect(frame, frame_id=frame_id)
        tracked_frame = tracker.update(det_frame, frame)
        annotated = draw_tracks(frame, tracked_frame)

        for evt in tracker.drain_lifecycle_events():
            logger.info(f"{evt.event} track #{evt.track_id}")

        cv2.imshow("Tracking", annotated)
        if writer:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()