"""
Bridge temporal action recognition output into Redis track records and memory events.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from libs.schemas.action_recognition import ActionFrameResult, ActionPrediction
from libs.schemas.memory import ActionHint, TrackEvent

if TYPE_CHECKING:
    from services.memory.memory import MemoryService

logger = logging.getLogger(__name__)

TRACK_TTL_SECONDS = 86_400


def _prediction_to_hint(pred: ActionPrediction) -> ActionHint:
    # ActionHint is a coarse vocabulary used by the legacy reasoning layer
    # (memory/action_classifier.py). It predates temporal action recognition
    # and only has WALKING, STANDING, LINGERING, and UNKNOWN.
    #
    # The mapping below is intentionally lossy:
    #   - "running"  → WALKING   (both are locomotion; no RUNNING hint exists)
    #   - "fighting" → UNKNOWN   (no FIGHTING hint; alert is raised separately via ActionAlert)
    #   - "falling"  → UNKNOWN   (same — handled by AlertSeverity.HIGH, not ActionHint)
    #
    # Downstream consumers that need the full action label should read
    # TrackEvent.temporal_action (set just before this call in apply_actions_to_memory_events)
    # rather than action_hint.
    mapping = {
        "walking": ActionHint.WALKING,
        "running": ActionHint.WALKING,
        "fighting": ActionHint.UNKNOWN,
        "loitering": ActionHint.LINGERING,
        "falling": ActionHint.UNKNOWN,
        "suspicious_stationary": ActionHint.STANDING,
        "unknown": ActionHint.UNKNOWN,
    }
    return mapping.get(pred.action.value, ActionHint.UNKNOWN)


def apply_actions_to_memory_events(
    events: list[TrackEvent],
    action_result: ActionFrameResult,
) -> list[TrackEvent]:
    """Attach temporal action labels to TrackEvents for the reasoning layer."""
    by_track = {p.track_id: p for p in action_result.predictions}
    enriched: list[TrackEvent] = []
    for evt in events:
        pred = by_track.get(evt.track_id)
        if pred is None:
            enriched.append(evt)
            continue
        data = evt.model_dump()
        data["temporal_action"] = pred.action.value
        data["temporal_action_confidence"] = pred.confidence
        data["temporal_action_source"] = pred.source
        data["action_hint"] = _prediction_to_hint(pred)
        enriched.append(TrackEvent(**data))
    return enriched


def publish_actions_to_redis(
    memory_service: "MemoryService",
    camera_id: str,
    action_result: ActionFrameResult,
) -> None:
    """Write current per-track action labels into track:{camera_id}:{track_id} records."""
    for pred in action_result.predictions:
        key = f"track:{camera_id}:{pred.track_id}"
        raw = memory_service._r.get(key)
        if raw is None:
            record: dict = {
                "camera_id": camera_id,
                "track_id": pred.track_id,
                "state": "ACTIVE",
            }
        else:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                record = {"camera_id": camera_id, "track_id": pred.track_id}

        record.update(
            {
                "current_action": pred.action.value,
                "action_confidence": pred.confidence,
                "action_source": pred.source,
                "action_updated_frame": action_result.frame_id,
            }
        )
        memory_service._r.setex(key, TRACK_TTL_SECONDS, json.dumps(record))
