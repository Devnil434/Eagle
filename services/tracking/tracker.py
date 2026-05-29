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

from libs.config import settings
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── adjust sys.path so we can import sibling packages ──────────────────────
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

    Wraps DeepSort (ByteTrack-compatible) and adds:
    - Dwell time tracking
    - Trajectory accumulation (last N points)
    - Zone membership per track
    - Lifecycle event emission (BORN / LOST / DEAD)
    """

    MAX_TRAJECTORY_LEN = 80  # max trajectory points stored per track
    FPS_DEFAULT = 30

    def __init__(
        self,
        fps: float = FPS_DEFAULT,
        max_age: int = 30,  # frames before a lost track is marked DEAD
        n_init: int = 3,  # frames before a track is CONFIRMED
        max_cosine_distance: float = 0.4,
        camera_id: str = "cam_01",
        event_logger: TrackEventLogger | None = None,
        reid_similarity_threshold: float = 0.85,
        max_interpolation_gap: int = 10,  # Added with a sensible default
    ) -> None:
        """Initialize the tracker with DeepSort hyperparameters and interpolation constraints.

        Args:
            fps: Frame rate of the video source.
            max_age: Maximum frames to keep a lost track alive before dropping it.
            n_init: Number of consecutive frames needed to confirm a track.
            max_cosine_distance: Maximum threshold for visual appearance feature matching.
            camera_id: Unique identifier string for the source camera.
            event_logger: Optional logger interface for tracking state lifecycle events.
            reid_similarity_threshold: Minimum confidence needed to reconnect an ID via ReID.
            max_interpolation_gap: Maximum frame gap size allowed to fill missing trajectories.
        """
        self.fps = fps
        self.camera_id = camera_id
        self.max_age = max_age  # NEW
        self.REID_SIMILARITY_THRESHOLD = reid_similarity_threshold
        self.max_interpolation_gap = max_interpolation_gap  # Fixed missing attribute

        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=100,
        )
        # Internal state
        self._active_tracks: dict[int, TrackedObject] = {}
        self._known_ids: set[int] = set()
        self._frame_id: int = 0
        self._lifecycle_queue: list[TrackLifecycleEvent] = []
        self._event_logger: TrackEventLogger | None = event_logger
        self._lost_embeddings: dict[int, dict] = {}
        self._active_embeddings: dict[int, np.ndarray] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        det_frame: DetectionFrameSchema,
        raw_frame: np.ndarray,
    ) -> TrackedFrame:
        """
        Ingest a DetectionFrame, run ByteTrack, return TrackedFrame.

        Args:
            det_frame:  Output of Phase 1 detector (DetectionFrameSchema).
            raw_frame:  Original BGR frame — needed for appearance features.

        Returns:
            TrackedFrame with all confirmed tracks, dwell times, trajectories.
        """
        self._frame_id = det_frame.frame_id
        frames_processed_total.inc()

        # ── Convert Pydantic detections → DeepSort input format ───────────
        # DeepSort expects: list of ([left, top, w, h], confidence, label)
        ds_input = []
        for det in det_frame.detections:
            if det.label != "person":  # track persons only in this phase
                continue
            b = det.bbox
            left, top = b.x1, b.y1
            w, h = b.x2 - b.x1, b.y2 - b.y1
            ds_input.append(([left, top, w, h], float(det.confidence), "person"))

        # ── Run tracker ────────────────────────────────────────────────────
        raw_tracks = self._tracker.update_tracks(ds_input, frame=raw_frame)

        # ── Build TrackedObject list ───────────────────────────────────────
        current_ids: set[int] = set()
        tracked_objects: list[TrackedObject] = []

        for t in raw_tracks:
            if not t.is_confirmed():
                continue

            tid = int(t.track_id)

            # ── ReID matching ─────────────────────────────────────
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

            # ── Lifecycle: BORN ───────────────────────────────────────────
            if tid not in self._known_ids:
                self._known_ids.add(tid)
                self._emit_lifecycle(TrackState.BORN, tid, zones, 0.0)
                logger.info(f"Track BORN: #{tid} in zones={zones}")

            # ── Base Setup & Gap Calculation ──────────────────────────────
            prev = self._active_tracks.get(tid)
            prev_traj = prev.trajectory if prev else []
            
            # Compute gap_frames early so both Dwell Time and Trajectory can use it
            gap_frames = max(0, self._frame_id - prev.last_seen_frame - 1) if prev is not None else 0

            # ── Dwell time ────────────────────────────────────────────────
            if prev:
                # Add historic frames, the current frame, and the occlusion gap
                dwell_frames = prev.dwell_time_frames + 1 + gap_frames
            else:
                dwell_frames = 1
            
            dwell_secs = dwell_frames / self.fps

            # ── Trajectory ────────────────────────────────────────────────
            interpolated_points = []
            max_gap = self.max_interpolation_gap  # <-- Replaced self.config string access

            if prev is not None and 0 < gap_frames <= max_gap:
                # Added guard condition below to prevent IndexError crashes
                if prev.trajectory:
                    last_pos = {"x": prev.trajectory[-1].x, "y": prev.trajectory[-1].y}
                else:
                    last_pos = {"x": cx, "y": cy}  # Fallback to current center coordinates
                
                new_pos = {"x": cx, "y": cy}
                
                # Check if previous data contains w and h bounding box metrics
                if hasattr(prev, 'bbox') and len(prev.bbox) == 4:
                    # Calculate old width and height from bbox: [x1, y1, x2, y2]
                    last_pos["w"] = prev.bbox[2] - prev.bbox[0]
                    last_pos["h"] = prev.bbox[3] - prev.bbox[1]
                    # Current width and height
                    new_pos["w"] = x2 - x1
                    new_pos["h"] = y2 - y1
                
                # Synthesize intermediate points and wrap them into TrajectoryPoint instances
                interpolated_points = [
                    TrajectoryPoint(
                        x=p["x"],
                        y=p["y"],
                        frame_id=p["frame_id"],
                        interpolated=True,
                        w=p.get("w"),
                        h=p.get("h")
                    )
                    for p in _interpolate_trajectory(last_pos, new_pos, gap_frames, prev.last_seen_frame + 1)
                ] 
            
            # Generate the current frame real point
            new_point = TrajectoryPoint(x=cx, y=cy, frame_id=self._frame_id)
            
            # Merge old history, calculated mid-gap points, and current point cleanly
            trajectory = (prev_traj + interpolated_points + [new_point])[-self.MAX_TRAJECTORY_LEN :]

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

        # ── Lifecycle: LOST for tracks that disappeared ────────────────────
        for tid, prev_obj in list(self._active_tracks.items()):
            if tid not in current_ids:
                frames_since = self._frame_id - prev_obj.last_seen_frame
                track = None
                if frames_since == 1:
                    track = next((t for t in raw_tracks if int(t.track_id) == tid), None)

                embedding = None
                if frames_since == 1:
                    if track is not None and hasattr(track, "features") and track.features:
                        embedding = track.features[-1]
                    else:
                        embedding = self._active_embeddings.get(tid)

                if frames_since == 1 and embedding is not None:
                    self._lost_embeddings[tid] = {
                        "embedding": embedding,
                        "last_seen": self._frame_id,
                    }

                self._emit_lifecycle(
                    TrackState.LOST,
                    tid,
                    prev_obj.zones_present,
                    prev_obj.dwell_time_seconds,
                )
                if frames_since >= self._tracker.max_age:
                    self._emit_lifecycle(
                        TrackState.DEAD,
                        tid,
                        prev_obj.zones_present,
                        prev_obj.dwell_time_seconds,
                    )
                    del self._active_tracks[tid]
                    self._active_embeddings.pop(tid, None)
                    logger.info(f"Track DEAD: #{tid} after {prev_obj.dwell_time_seconds:.1f}s")
                    
        # ── Cleanup expired ReID embeddings ──────────────────
        expired_ids = [
            tid
            for tid, data in self._lost_embeddings.items()
            if self._frame_id - data["last_seen"] > self.max_age
        ]

        for tid in expired_ids:
            del self._lost_embeddings[tid]

        return TrackedFrame(
            frame_id=self._frame_id,
            camera_id=self.camera_id,
            tracks=tracked_objects,
            timestamp_ms=time.time() * 1000,
            fps=self.fps,
        )

    def drain_lifecycle_events(self) -> list[TrackLifecycleEvent]:
        """
        Pop and return all pending lifecycle events since last call.
        Called by the memory service to store BORN/LOST/DEAD events.
        """
        events = list(self._lifecycle_queue)
        self._lifecycle_queue.clear()
        return events

    # ── Internal ────────────────────────────────────────────────────────────

    def _emit_lifecycle(
        self,
        state: TrackState,
        track_id: int,
        zones: list[str],
        dwell_secs: float,
    ) -> None:
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
        if self._event_logger is not None:
            self._event_logger.log_event(event)

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        norm_product = np.linalg.norm(a) * np.linalg.norm(b)
        if norm_product == 0:
            return 0.0

        return float(np.dot(a, b) / norm_product)


# ─── CLI Demo ────────────────────────────────────────────────────────────────


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from services.detection.detector import Detector
    from services.tracking.visualizer import draw_tracks

    parser = argparse.ArgumentParser(description="Phase 2 — Tracking demo")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default=settings.detector_model, help="YOLO model name")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    detector = Detector(model_name=args.model)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
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

        # Drain lifecycle events (Phase 3 will store these in Redis)
        for evt in tracker.drain_lifecycle_events():
            logger.info(
                f"Lifecycle: {evt.event} track #{evt.track_id} "
                f"dwell={evt.dwell_time_seconds:.1f}s zones={evt.zones_present}"
            )

        cv2.imshow("Agentic Vision — Tracking", annotated)
        if writer:
            writer.write(annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_id += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def _interpolate_trajectory(
    last_pos: dict, 
    new_pos: dict, 
    gap_frames: int, 
    start_frame_id: int
) -> list:
    """Fills trajectory gaps using linear interpolation for temporary missed detections."""
    if gap_frames <= 0:
        return []

    interpolated_points = []
    total_steps = gap_frames + 1
    
    x_step = (new_pos['x'] - last_pos['x']) / total_steps
    y_step = (new_pos['y'] - last_pos['y']) / total_steps
    
    for i in range(1, gap_frames + 1):
        point = {
            "frame_id": start_frame_id + (i - 1),
            "x": round(last_pos['x'] + (x_step * i), 2),
            "y": round(last_pos['y'] + (y_step * i), 2),
            "interpolated": True
        }
        
        if all(k in last_pos and k in new_pos for k in ('w', 'h')):
            point['w'] = round(last_pos['w'] + (((new_pos['w'] - last_pos['w']) / total_steps) * i), 2)
            point['h'] = round(last_pos['h'] + (((new_pos['h'] - last_pos['h']) / total_steps) * i), 2)
            
        interpolated_points.append(point)
        
    return interpolated_points

if __name__ == "__main__":
    main()