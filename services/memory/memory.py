"""
memory.py — Persistent memory layer for Eagle surveillance.

Stores lifecycle events (BORN / LOST / DEAD) in Redis and orchestrates
cross-camera ReID so every track gets a ``global_id`` alongside its local
``track_id``.

Redis key schema (extended for global IDs)
------------------------------------------
- ``track:{camera_id}:{track_id}``   → JSON blob    ← per-track state
- ``event:{camera_id}:{frame_id}``   → JSON list     ← lifecycle history
- ``embed:{camera_id}:{track_id}``   → JSON blob     ← ReID embedding (TTL 7 s)
- ``identity:{global_id}``           → JSON list     ← cross-cam tokens (TTL 1 h)

Usage
-----
    import redis
    from services.memory.memory import MemoryService
    from services.tracking.cross_camera_reid import CrossCameraReID

    r    = redis.Redis()
    reid = CrossCameraReID(r)
    mem  = MemoryService(r, reid)

    # In your tracking loop:
    for event in tracker.drain_lifecycle_events():
        global_id = mem.handle_lifecycle_event(event, embedding=vec)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import numpy as np

from libs.observability.metrics import redis_write_latency
from libs.schemas.tracking import TrackLifecycleEvent, TrackState
from services.tracking.cross_camera_reid import CrossCameraReID

logger = logging.getLogger(__name__)

# ── Redis TTLs ────────────────────────────────────────────────────────────────
TRACK_TTL_SECONDS = 86_400  # 24 h — keep per-track state for a full day
EVENT_TTL_SECONDS = 86_400


class MemoryService:
    """
    Writes tracking lifecycle events to Redis and assigns global identities
    via ReID.

    Parameters
    ----------
    redis_client:
        Connected ``redis.Redis`` (or FakeRedis for tests).
    reid:
        ``CrossCameraReID`` instance sharing the same Redis client.
    """

    def __init__(self, redis_client, reid: CrossCameraReID) -> None:
        self._r = redis_client
        self._reid = reid

    # ── Public API ────────────────────────────────────────────────────────────

    def handle_lifecycle_event(
        self,
        event: TrackLifecycleEvent,
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Process a single lifecycle event and return the assigned global_id.

        - BORN  → attempt ReID match; mint or reuse a global_id; store track record.
        - LOST  → store embedding for future cross-camera matching; update record.
        - DEAD  → mark track as dead; update record.

        Args:
            event:     TrackLifecycleEvent from Tracker.drain_lifecycle_events().
            embedding: Appearance feature vector.  Required for BORN/LOST events
                       if cross-camera ReID is desired; may be None in tests.

        Returns:
            The global_id string if one was assigned, else None.
        """
        global_id: Optional[str] = None

        if event.event == TrackState.BORN:
            global_id = self._handle_born(event, embedding)

        elif event.event == TrackState.LOST:
            global_id = self._handle_lost(event, embedding)

        elif event.event == TrackState.DEAD:
            self._handle_dead(event)

        # Always append the raw event to the event log
        self._append_event(event, global_id)
        return global_id

    def get_track_record(self, camera_id: str, track_id: int) -> Optional[dict]:
        """Retrieve the stored track record, or None if not found."""
        raw = self._r.get(self._track_key(camera_id, track_id))
        return json.loads(raw) if raw else None

    def get_identity(self, global_id: str) -> list[str]:
        """Proxy to CrossCameraReID.get_identity."""
        return self._reid.get_identity(global_id)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _handle_born(
        self,
        event: TrackLifecycleEvent,
        embedding: Optional[np.ndarray],
    ) -> str:
        if embedding is not None:
            reid_result = self._reid.match_or_create(
                camera_id=event.camera_id,
                track_id=event.track_id,
                embedding=embedding,
            )
            global_id = reid_result.global_id
        else:
            # No embedding available → mint a placeholder global_id
            import uuid

            global_id = str(uuid.uuid4())
            logger.warning(
                "BORN event for cam=%s track=%d has no embedding; "
                "cross-camera ReID disabled for this track.",
                event.camera_id,
                event.track_id,
            )

        record = {
            "camera_id": event.camera_id,
            "track_id": event.track_id,
            "global_id": global_id,
            "state": TrackState.ACTIVE.value,
            "born_frame": event.frame_id,
            "born_timestamp_ms": event.timestamp_ms,
            "last_seen_frame": event.frame_id,
            "last_seen_ms": event.timestamp_ms,
            "dwell_time_seconds": event.dwell_time_seconds,
            "zones_present": event.zones_present,
        }
        with redis_write_latency.time():
            self._r.setex(
                self._track_key(event.camera_id, event.track_id),
                TRACK_TTL_SECONDS,
                json.dumps(record),
            )
        logger.info("BORN  cam=%s track=%d gid=%s", event.camera_id, event.track_id, global_id)
        return global_id

    def _handle_lost(
        self,
        event: TrackLifecycleEvent,
        embedding: Optional[np.ndarray],
    ) -> Optional[str]:
        record = self._load_record(event.camera_id, event.track_id)
        global_id = record.get("global_id") if record else None

        # Store embedding so another camera can match against it within 5 s
        if embedding is not None:
            self._reid.store_embedding(
                camera_id=event.camera_id,
                track_id=event.track_id,
                embedding=embedding,
                global_id=global_id,
            )

        self._update_record(event, TrackState.LOST.value)
        logger.info("LOST  cam=%s track=%d gid=%s", event.camera_id, event.track_id, global_id)
        return global_id

    def _handle_dead(self, event: TrackLifecycleEvent) -> None:
        self._update_record(event, TrackState.DEAD.value)
        logger.info("DEAD  cam=%s track=%d", event.camera_id, event.track_id)

    # ── Redis helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _track_key(camera_id: str, track_id: int) -> str:
        return f"track:{camera_id}:{track_id}"

    @staticmethod
    def _event_key(camera_id: str, frame_id: int) -> str:
        return f"event:{camera_id}:{frame_id}"

    def _load_record(self, camera_id: str, track_id: int) -> Optional[dict]:
        raw = self._r.get(self._track_key(camera_id, track_id))
        return json.loads(raw) if raw else None

    def _update_record(self, event: TrackLifecycleEvent, state: str) -> None:
        record = self._load_record(event.camera_id, event.track_id) or {}
        record.update(
            {
                "state": state,
                "last_seen_frame": event.frame_id,
                "last_seen_ms": event.timestamp_ms,
                "dwell_time_seconds": event.dwell_time_seconds,
                "zones_present": event.zones_present,
            }
        )
        self._r.setex(
            self._track_key(event.camera_id, event.track_id),
            TRACK_TTL_SECONDS,
            json.dumps(record),
        )

    def _append_event(
        self,
        event: TrackLifecycleEvent,
        global_id: Optional[str],
    ) -> None:
        key = self._event_key(event.camera_id, event.frame_id)
        raw = self._r.get(key)
        evts: list[dict] = json.loads(raw) if raw else []
        evts.append(
            {
                "event": event.event.value,
                "track_id": event.track_id,
                "global_id": global_id,
                "frame_id": event.frame_id,
                "timestamp_ms": event.timestamp_ms,
                "dwell_time_seconds": event.dwell_time_seconds,
                "zones_present": event.zones_present,
            }
        )
        with redis_write_latency.time():
            self._r.setex(
                key,
                EVENT_TTL_SECONDS,
                json.dumps(evts),
            )
