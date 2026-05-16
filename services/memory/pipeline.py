"""
pipeline.py — Connects Phase 2 tracker output to Phase 3 memory store.

Called once per frame from the main loop (or FastAPI ingest endpoint).
"""
from __future__ import annotations

import time
from libs.schemas.tracking import TrackedFrame
from libs.schemas.memory   import TrackEvent
from services.memory.action_classifier import classify_action
from services.memory.memory import MemoryStore

# Shared state for action classifier (tracks zone-entry history)
_zone_entry_registry: dict[int, set[str]] = {}
_prev_objects: dict[int, object] = {}


def process_tracked_frame(tracked: TrackedFrame, store: MemoryStore) -> list[TrackEvent]:
    """
    Convert a TrackedFrame into TrackEvents and write them to Redis.

    Args:
        tracked: Output of Phase 2 Tracker.update()
        store:   MemoryStore instance connected to Redis

    Returns:
        List of TrackEvent objects that were stored (useful for logging/testing).
    """
    events: list[TrackEvent] = []

    for obj in tracked.tracks:
        prev = _prev_objects.get(obj.track_id)
        hint = classify_action(obj, prev, _zone_entry_registry)

        event = TrackEvent(
            track_id           = obj.track_id,
            camera_id          = tracked.camera_id,
            frame_id           = tracked.frame_id,
            timestamp_ms       = time.time() * 1000,
            zone               = obj.zones_present[0] if obj.zones_present else None,
            action_hint        = hint,
            bbox               = obj.bbox,
            center             = obj.center,
            dwell_time_seconds = obj.dwell_time_seconds,
            confidence         = obj.confidence,
        )

        store.store_event(event)
        events.append(event)
        _prev_objects[obj.track_id] = obj

    return events