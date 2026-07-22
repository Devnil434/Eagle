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
from libs.config.settings import settings
from services.memory.baseline import ZoneBaseline
from services.tracking.cross_camera_reid import CrossCameraReID

logger = logging.getLogger(__name__)

# ── Redis TTLs (sourced from centralised settings) ───────────────────────────
TRACK_TTL_SECONDS = settings.track_ttl_seconds
EVENT_TTL_SECONDS = settings.track_ttl_seconds

# ── MemoryStore constants ─────────────────────────────────────────────────────
MAX_EVENTS_PER_TRACK = 50   # ring-buffer cap per track_id


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
        zone_anomalous: bool = False

        if event.event == TrackState.BORN:
            global_id = self._handle_born(event, embedding)

        elif event.event == TrackState.LOST:
            global_id, zone_anomalous = self._handle_lost(event, embedding)

        elif event.event == TrackState.DEAD:
            self._handle_dead(event)

        # Always append the raw event to the event log
        self._append_event(event, global_id, zone_anomalous)
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
    ) -> tuple[Optional[str], bool]:
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

        # Detect anomaly BEFORE updating baseline (avoid contaminating with outlier)
        # then update baseline for each zone this track visited
        zone_anomalous = False
        for zone in event.zones_present:
            baseline = ZoneBaseline(self._r, zone)
            if baseline.is_anomalous(event.dwell_time_seconds):
                zone_anomalous = True
            baseline.update(event.dwell_time_seconds)

        self._update_record(event, TrackState.LOST.value, zone_anomalous)
        logger.info("LOST  cam=%s track=%d gid=%s anomalous=%s",
                    event.camera_id, event.track_id, global_id, zone_anomalous)
        return global_id, zone_anomalous

    def _handle_dead(self, event: TrackLifecycleEvent) -> None:
        self._update_record(event, TrackState.DEAD.value)
        logger.info("DEAD  cam=%s track=%d", event.camera_id, event.track_id)

    # ── Redis helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _track_key(camera_id: str, track_id: int) -> str:
        """Return the Redis key for a per-track state blob."""
        return f"track:{camera_id}:{track_id}"

    @staticmethod
    def _event_key(camera_id: str, frame_id: int) -> str:
        """Return the Redis key for a per-frame event list."""
        return f"event:{camera_id}:{frame_id}"

    def _load_record(self, camera_id: str, track_id: int) -> Optional[dict]:
        """Load and deserialise a track record from Redis, or return None."""
        raw = self._r.get(self._track_key(camera_id, track_id))
        return json.loads(raw) if raw else None

    def _update_record(
        self,
        event: TrackLifecycleEvent,
        state: str,
        anomalous: bool = False,
    ) -> None:
        """
        Update an existing track record's state and timing fields in Redis.

        Args:
            event: Source lifecycle event supplying updated field values.
            state: New state string (e.g. 'LOST', 'DEAD').
        """
        record = self._load_record(event.camera_id, event.track_id) or {}
        record.update(
            {
                "state": state,
                "last_seen_frame": event.frame_id,
                "last_seen_ms": event.timestamp_ms,
                "dwell_time_seconds": event.dwell_time_seconds,
                "zones_present": event.zones_present,
                "anomalous": anomalous,
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
        anomalous: bool = False,
    ) -> None:
        """
        Append a lifecycle event dict to the per-frame Redis event log.

        Args:
            event:     Source lifecycle event.
            global_id: Assigned global identity string, or None.
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
                "anomalous": anomalous,
            }
        )
        with redis_write_latency.time():
            self._r.setex(
                key,
                EVENT_TTL_SECONDS,
                json.dumps(evts),
            )


# Compatibility layer: lightweight event store used by tests and the pipeline.
MAX_EVENTS_PER_TRACK = settings.max_events_per_track


class MemoryStore:
    """
    Lightweight ring-buffer event store for per-track behavioural sequences.

    Stores ``TrackEvent`` objects (Phase 3 schema) in Redis lists capped at
    ``MAX_EVENTS_PER_TRACK`` entries.  Designed for the action-classifier →
    VLM/LLM reasoning pipeline.

    Redis key schema
    ----------------
    - ``seq:{camera_id}:{track_id}``                    → JSON list of TrackEvent dicts
    - ``zones:{camera_id}:{track_id}``                  → Redis set of zone names visited
    - ``zone_count:{camera_id}:{track_id}:{zone}``      → integer entry count
    - ``active:{camera_id}``                            → Redis set of active track_ids

    Parameters
    ----------
    redis_client:
        Connected ``redis.Redis`` (or FakeRedis for tests).
    camera_id:
        Default camera identifier used when none is supplied per-event.
    """

    def __init__(self, redis_client, camera_id: str = "cam_01") -> None:
        self._r = redis_client
        self._camera_id = camera_id

    # ── Key helpers ───────────────────────────────────────────────────────────

    def _seq_key(self, track_id: int, camera_id: Optional[str] = None) -> str:
        cam = camera_id or self._camera_id
        return f"seq:{cam}:{track_id}"

    def _zones_key(self, track_id: int, camera_id: Optional[str] = None) -> str:
        cam = camera_id or self._camera_id
        return f"zones:{cam}:{track_id}"

    def _zone_count_key(self, track_id: int, zone: str, camera_id: Optional[str] = None) -> str:
        cam = camera_id or self._camera_id
        return f"zone_count:{cam}:{track_id}:{zone}"

    def _active_key(self, camera_id: Optional[str] = None) -> str:
        cam = camera_id or self._camera_id
        return f"active:{cam}"

    def store_event(self, event) -> None:
        """
        Append a ``TrackEvent`` to the ring buffer for its track.

        Enforces the ``MAX_EVENTS_PER_TRACK`` cap by trimming the oldest
        entry whenever the list exceeds the limit.  Also maintains the
        zones-visited set, per-zone entry counts, and the active-tracks set.

        Args:
            event: ``TrackEvent`` instance (from ``libs.schemas.memory``).
        """

        key = self._seq_key(event.track_id)
        serialised = event.model_dump_json()

        pipe = self._r.pipeline()
        pipe.rpush(key, serialised)
        pipe.ltrim(key, -MAX_EVENTS_PER_TRACK, -1)
        pipe.sadd(self._active_key(), str(event.track_id))

        if event.zone:
            pipe.sadd(self._zones_key(event.track_id), event.zone)
            if event.action_hint == ActionHint.ZONE_ENTRY:
                pipe.incr(self._zone_count_key(event.track_id, event.zone))

        pipe.execute()

    def get_sequence(self, track_id: int, last_n: Optional[int] = None, camera_id: Optional[str] = None):
        """
        Return a ``TrackSequence`` for the given track.

        Args:
            track_id: Track identifier.
            last_n:   If given, return only the most recent *n* events.

        Returns:
            ``TrackSequence`` (empty if the track has no stored events).
        """
        key = self._seq_key(track_id, camera_id)
        raw_list = self._r.lrange(key, -last_n, -1) if last_n else self._r.lrange(key, 0, -1)
        events: list[TrackEvent] = []
        for raw in raw_list:
            try:
                data = json.loads(raw if isinstance(raw, str) else raw.decode())
                events.append(TrackEvent(**data))
            except Exception:
                continue

        zones_raw = self._r.smembers(self._zones_key(track_id, camera_id))
        zones_visited = [z if isinstance(z, str) else z.decode() for z in zones_raw]
        total_dwell = sum(e.dwell_time_seconds for e in events)

        return TrackSequence(
            track_id=track_id,
            camera_id=camera_id or self._camera_id,
            events=events,
            zones_visited=zones_visited,
            total_dwell=total_dwell,
        )

    def get_zone_entry_count(self, track_id: int, zone: str, camera_id: Optional[str] = None) -> int:
        """Return the number of times *track_id* has entered *zone*."""
        raw = self._r.get(self._zone_count_key(track_id, zone, camera_id))
        if raw is None:
            return 0
        return int(raw if isinstance(raw, (int, str)) else raw.decode())

    def get_active_track_ids(self, camera_id: str) -> set[int]:
        """Return the set of track IDs currently marked active for *camera_id*."""
        members = self._r.smembers(self._active_key(camera_id))
        return {int(m if isinstance(m, (int, str)) else m.decode()) for m in members}

    def expire_track(self, track_id: int, camera_id: Optional[str] = None) -> None:
        """Remove all stored data for *track_id* and deregister it as active."""
        pipe = self._r.pipeline()
        pipe.delete(self._seq_key(track_id, camera_id))
        pipe.delete(self._zones_key(track_id, camera_id))
        pipe.srem(self._active_key(camera_id), str(track_id))
        pipe.execute()
