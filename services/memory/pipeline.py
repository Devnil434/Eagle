"""
pipeline.py — Connects Phase 2 tracker output to Phase 3 memory store.

Called once per frame from the main loop (or FastAPI ingest endpoint).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from libs.schemas.tracking import TrackedFrame
from libs.schemas.memory   import TrackEvent
from services.memory.action_classifier import classify_action
from services.memory.memory import MemoryStore
from services.memory.kafka_producer import KafkaEventProducer
from libs.config.settings import settings

if TYPE_CHECKING:
    from services.action_recognition.inference import ActionRecognizer
    from services.memory.memory import MemoryService

# Shared state for action classifier (tracks zone-entry history)
_zone_entry_registry: dict[int, set[str]] = {}
_zone_entry_counts: dict[int, dict[str, int]] = {}
_last_repeated_approach: dict[int, float] = {}
_prev_objects: dict[int, object] = {}

# Global Kafka producer instance
_kafka_producer: KafkaEventProducer | None = None
if settings.use_kafka:
    _kafka_producer = KafkaEventProducer()


def process_tracked_frame(
    tracked: TrackedFrame,
    store: MemoryStore,
    raw_frame: Optional[np.ndarray] = None,
    action_recognizer: Optional["ActionRecognizer"] = None,
    memory_service: Optional["MemoryService"] = None,
) -> list[TrackEvent]:
    """
    Convert a TrackedFrame into TrackEvents and persist them to storage systems.

    This function processes tracking results from the detection pipeline and:
    - Classifies actions for each detected object
    - Creates structured TrackEvent objects
    - Stores events in Redis via MemoryStore or publishes to Kafka
    - Optionally applies action recognition results for temporal reasoning

    Args:
        tracked (TrackedFrame): Frame containing detected and tracked objects.
        store (MemoryStore): Redis-backed event storage system.
        raw_frame (Optional[np.ndarray]): Original video frame for action recognition.
        action_recognizer (Optional[ActionRecognizer]): Model used for temporal action detection.
        memory_service (Optional[MemoryService]): Global memory system for advanced reasoning.

    Returns:
        list[TrackEvent]: List of processed tracking events generated from this frame.

    Raises:
        Exception: If action recognition or event processing fails (handled internally and logged).
    """
    events: list[TrackEvent] = []

    for obj in tracked.tracks:
        prev = _prev_objects.get(obj.track_id)
        current_time_ms = time.time() * 1000
        hint = classify_action(
            obj, 
            prev, 
            _zone_entry_registry,
            zone_entry_counts=_zone_entry_counts,
            last_repeated_approach=_last_repeated_approach,
            current_time_ms=current_time_ms
        )

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

        if settings.use_kafka and _kafka_producer and _kafka_producer.producer is not None:
            _kafka_producer.produce_event(event)
        else:
            store.store_event(event)

        events.append(event)
        _prev_objects[obj.track_id] = obj

    if action_recognizer is not None and raw_frame is not None:
        try:
            from services.memory.action_bridge import (
                apply_actions_to_memory_events,
                publish_actions_to_redis,
            )
            action_result = action_recognizer.update(tracked, raw_frame)
            events = apply_actions_to_memory_events(events, action_result)
            if memory_service is not None:
                publish_actions_to_redis(memory_service, tracked.camera_id, action_result)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Action recognition failed: %s", exc)

    return events