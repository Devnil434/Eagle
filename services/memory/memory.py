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
from typing import Optional

import numpy as np

from libs.observability.metrics import redis_write_latency
from libs.schemas.tracking import TrackLifecycleEvent, TrackState
from libs.schemas.memory import TrackEvent, TrackSequence, ActionHint
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
        """
        Initialise MemoryService with a Redis client and ReID engine.

        Args:
            redis_client: Connected redis.Redis or FakeRedis instance.
            reid:         CrossCameraReID instance for global ID assignment.
        """
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
        """
        Handle a BORN lifecycle event.

        Attempts ReID match to reuse an existing global_id; mints a new
        UUID if no embedding is available.

        Args:
            event:     The BORN TrackLifecycleEvent.
            embedding: Appearance feature vector, or None.

        Returns:
            Assigned global_id string.
        """
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
        """
        Handle a LOST lifecycle event.

        Stores the appearance embedding so another camera can match against
        it within the ReID TTL window.

        Args:
            event:     The LOST TrackLifecycleEvent.
            embedding: Appearance feature vector, or None.

        Returns:
            The existing global_id if found, else None.
        """
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
        """
        Handle a DEAD lifecycle event.

        Marks the track record as DEAD in Redis.

        Args:
            event: The DEAD TrackLifecycleEvent.
        """
        self._update_record(event, TrackState.DEAD.value)
        logger.info("DEAD  cam=%s track=%d", event.camera_id, event.track_id)

    # ── Redis helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _track_key(camera_id: str, track_id: int) -> str:
        """
        Generate Redis key for storing per-track state.

        Args:
            camera_id (str): Camera identifier.
            track_id (int): Track identifier.

        Returns:
            str: Redis key in format track:{camera_id}:{track_id}
        """

        return f"track:{camera_id}:{track_id}"

    @staticmethod
    def _event_key(camera_id: str, frame_id: int) -> str:
        """
        Generate Redis key for storing per-frame lifecycle events.

        Args:
            camera_id (str): Camera identifier.
            frame_id (int): Frame number.

        Returns:
            str: Redis key in format event:{camera_id}:{frame_id}
        """

        return f"event:{camera_id}:{frame_id}"

    def _load_record(self, camera_id: str, track_id: int) -> Optional[dict]:
        """
        Load a track record from Redis and convert it to a Python dictionary.

        This method retrieves stored tracking information for a given
        camera_id and track_id combination.

        Args:
            camera_id (str): Camera identifier.
            track_id (int): Unique tracking ID.

        Returns:
            Optional[dict]: Track record if found, otherwise None.
        """

        raw = self._r.get(self._track_key(camera_id, track_id))
        return json.loads(raw) if raw else None

    def _update_record(self, event: TrackLifecycleEvent, state: str) -> None:
        """
        Update an existing track record in Redis with new lifecycle state.

        This updates:
        - Track state (LOST / DEAD / ACTIVE)
        - Last seen frame
        - Last seen timestamp
        - Dwell time
        - Zones visited

        Args:
            event (TrackLifecycleEvent): Lifecycle event containing update data.
            state (str): New state to assign to the track.

        Returns:
            None
        """

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
        """
        Append a lifecycle event to Redis event history.

        Stores per-frame event logs including:
        - Event type (BORN / LOST / DEAD)
        - Track ID
        - Global ID (if available)
        - Timestamp and metadata

        Args:
            event (TrackLifecycleEvent): Source lifecycle event.
            global_id (Optional[str]): Global identity assigned to track.

        Returns:
            None
        """
        
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


# Compatibility layer: lightweight event store used by tests and the pipeline.
MAX_EVENTS_PER_TRACK = 50


class MemoryStore:
    """Simple Redis-backed ring buffer for TrackEvent objects.

    This is intentionally minimal: it stores JSON-serialised events in a
    Redis list (oldest -> newest), trims to `MAX_EVENTS_PER_TRACK`, and
    exposes the methods used by unit tests and the pipeline.
    """

    def __init__(self, redis_client=None, prefix: str = "mem") -> None:
        """
        Initialise MemoryStore.

        Args:
            redis_client: Connected redis.Redis instance, or None to create one.
            prefix:       Key prefix used for all Redis keys (default: 'mem').
        """
        import redis

        self._r = redis_client or redis.Redis()
        self._prefix = prefix

    def _events_key(self, track_id: int) -> str:
        """Return the Redis list key for a track's event history."""
        return f"{self._prefix}:events:{track_id}"

    def _active_key(self, camera_id: str) -> str:
        """Return the Redis set key for active track IDs on a camera."""
        return f"{self._prefix}:active:{camera_id}"

    def _track_camera_key(self, track_id: int) -> str:
        """Return the Redis key mapping a track_id to its camera_id."""
        return f"{self._prefix}:track_camera:{track_id}"

    def store_event(self, evt: TrackEvent) -> None:
        """
        Persist a TrackEvent to Redis and maintain the active-tracks set.

        Trims the event list to MAX_EVENTS_PER_TRACK after each write.

        Args:
            evt: TrackEvent instance to store.
        """
        key = self._events_key(evt.track_id)
        payload = evt.model_dump() if hasattr(evt, "model_dump") else evt.dict()
        self._r.rpush(key, json.dumps(payload))
        self._r.ltrim(key, -MAX_EVENTS_PER_TRACK, -1)
        self._r.sadd(self._active_key(evt.camera_id), str(evt.track_id))
        self._r.set(self._track_camera_key(evt.track_id), evt.camera_id)
        self._r.expire(key, TRACK_TTL_SECONDS)

    def get_sequence(self, track_id: int, last_n: Optional[int] = None) -> TrackSequence:
        """
        Retrieve the event sequence for a track from Redis.

        Args:
            track_id: Integer track identifier.
            last_n:   If given, return only the most recent N events.

        Returns:
            TrackSequence containing the requested events.
        """
        key = self._events_key(track_id)
        raw = self._r.lrange(key, 0, -1)
        events: list[TrackEvent] = []
        for item in raw:
            data = json.loads(item)
            events.append(TrackEvent(**data))
        if last_n is not None:
            events = events[-last_n:]
        zones = list(dict.fromkeys(e.zone for e in events if e.zone is not None))
        return TrackSequence(track_id=track_id, events=events, zones_visited=zones)

    def get_zone_entry_count(self, track_id: int, zone: str) -> int:
        """
        Count how many times a track entered a specific zone.

        Args:
            track_id: Integer track identifier.
            zone:     Zone name string to filter on.

        Returns:
            Integer count of ZONE_ENTRY events for the given zone.
        """
        seq = self.get_sequence(track_id)
        return sum(
            1 for e in seq.events
            if e.zone == zone and e.action_hint == ActionHint.ZONE_ENTRY
        )

    def get_active_track_ids(self, camera_id: str) -> set[int]:
        """
        Return the set of currently active track IDs for a camera.

        Args:
            camera_id: Camera identifier string.

        Returns:
            Set of integer track IDs active on that camera.
        """
        members = self._r.smembers(self._active_key(camera_id))
        result: set[int] = set()
        for m in members:
            try:
                result.add(int(m))
            except Exception:
                continue
        return result

    def expire_track(self, track_id: int) -> None:
        """
        Remove all Redis state for a track and drop it from the active set.

        Args:
            track_id: Integer track identifier to expire.
        """
        cam = self._r.get(self._track_camera_key(track_id))
        if cam:
            try:
                cam = cam if isinstance(cam, str) else cam.decode()
            except Exception:
                pass
            self._r.srem(self._active_key(cam), str(track_id))
        self._r.delete(self._events_key(track_id))
        self._r.delete(self._track_camera_key(track_id))
