"""
pipeline.py — Connects Phase 2 tracker output to Phase 3 memory store.

Called once per frame from the main loop (or FastAPI ingest endpoint).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from libs.schemas.tracking import TrackedFrame
from libs.schemas.memory import TrackEvent
from services.memory.action_classifier import classify_action
from services.memory.action_bridge import apply_actions_to_memory_events, publish_actions_to_redis
from services.memory.memory import MemoryStore

if TYPE_CHECKING:
    from services.action_recognition.inference import ActionRecognizer
    from services.memory.memory import MemoryService

# Shared state for action classifier (tracks zone-entry history)
_zone_entry_registry: dict[int, set[str]] = {}
_prev_objects: dict[int, object] = {}


def process_tracked_frame(
    tracked: TrackedFrame,
    store: MemoryStore,
    raw_frame: Optional[np.ndarray] = None,
    action_recognizer: Optional["ActionRecognizer"] = None,
    memory_service: Optional["MemoryService"] = None,
) -> list[TrackEvent]:
    """
    Convert a TrackedFrame into TrackEvents and write them to Redis.

    When *action_recognizer* and *raw_frame* are provided, temporal action
    labels from action_model.onnx are attached and published to track records
    for the dashboard API.
    """
    events: list[TrackEvent] = []

    for obj in tracked.tracks:
        prev = _prev_objects.get(obj.track_id)
        hint = classify_action(obj, prev, _zone_entry_registry)

        event = TrackEvent(
            track_id=obj.track_id,
            camera_id=tracked.camera_id,
            frame_id=tracked.frame_id,
            timestamp_ms=time.time() * 1000,
            zone=obj.zones_present[0] if obj.zones_present else None,
            action_hint=hint,
            bbox=obj.bbox,
            center=obj.center,
            dwell_time_seconds=obj.dwell_time_seconds,
            confidence=obj.confidence,
        )
        events.append(event)
        _prev_objects[obj.track_id] = obj

    if action_recognizer is not None and raw_frame is not None:
        action_result = action_recognizer.update(tracked, raw_frame)
        events = apply_actions_to_memory_events(events, action_result)
        if memory_service is not None:
            publish_actions_to_redis(memory_service, tracked.camera_id, action_result)

    for evt in events:
        store.store_event(evt)

    return events
