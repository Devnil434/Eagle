"""
run_pipeline.py — Integration demo for Phases 1+2+3.
Runs detection → tracking → memory for each frame of a video.
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

import cv2, time, logging
from services.detection.detector import Detector
from services.tracking.tracker   import Tracker
from services.tracking.visualizer import draw_tracks
from services.memory.memory       import MemoryStore
from services.memory.pipeline     import process_tracked_frame

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline")

SOURCE = "data/sample_videos/sample.mp4"

cap      = cv2.VideoCapture(SOURCE)
fps      = cap.get(cv2.CAP_PROP_FPS) or 30
detector = Detector()
tracker  = Tracker(fps=fps)
store    = MemoryStore()

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phase 1 — Detection
    det_frame     = detector.detect(frame, frame_id=frame_id)

    # Phase 2 — Tracking
    tracked_frame = tracker.update(det_frame, frame)

    # Phase 3 — Memory
    events        = process_tracked_frame(tracked_frame, store)

    # Log sequences every 90 frames (~3s)
    if frame_id % 90 == 0:
        for track in tracked_frame.tracks:
            seq = store.get_sequence(track.track_id)
            logger.info(
                f"Track #{track.track_id} | events={len(seq.events)} | "
                f"summary={seq.action_summary} | dwell={seq.total_dwell:.1f}s"
            )

    annotated = draw_tracks(frame, tracked_frame)
    cv2.imshow("Agentic Vision — Phase 1+2+3", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()