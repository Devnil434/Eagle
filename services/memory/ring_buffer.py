from __future__ import annotations
import json
from typing import Optional

from libs.schemas.memory import TrackEvent, TrackSequence

MAX_EVENTS_PER_TRACK = 50


class MemoryStore:
    """Lightweight Redis-backed ring buffer used by tests."""

    def __init__(self, redis_client=None, prefix: str = "mem", camera_id: str = "cam_01") -> None:
        import redis

        self._r = redis_client or redis.Redis()
        self._prefix = prefix
        self._camera_id = camera_id

    def _events_key(self, track_id: int) -> str:
        return f"{self._prefix}:events:{track_id}"

    def store_event(self, evt: TrackEvent) -> None:
        key = self._events_key(evt.track_id)
        payload = evt.model_dump() if hasattr(evt, "model_dump") else evt.dict()
        self._r.rpush(key, json.dumps(payload))
        self._r.ltrim(key, -MAX_EVENTS_PER_TRACK, -1)

    def get_sequence(self, track_id: int, last_n: Optional[int] = None) -> TrackSequence:
        key = self._events_key(track_id)
        raw_list = self._r.lrange(key, 0, -1)
        events = []
        for raw in raw_list:
            try:
                data = json.loads(raw if isinstance(raw, str) else raw.decode())
                events.append(TrackEvent(**data))
            except Exception:
                continue

        total_dwell = sum(e.dwell_time_seconds for e in events)
        zones_visited = [e.zone for e in events if e.zone]

        return TrackSequence(
            track_id=track_id,
            camera_id=self._camera_id,
            events=events,
            zones_visited=zones_visited,
            total_dwell=total_dwell,
        )

    # Alerts storage (simple sorted set by timestamp)
    def store_alert(self, alert_json: str, timestamp_ms: float, camera_id: str = "cam_01") -> None:
        key = f"alerts:{camera_id}"
        # Use score = timestamp_ms
        self._r.zadd(key, {alert_json: timestamp_ms})

    def get_alerts(self, camera_id: str = "cam_01", limit: int = 10) -> list[str]:
        key = f"alerts:{camera_id}"
        items = self._r.zrevrange(key, 0, limit - 1)
        return [i if isinstance(i, str) else i.decode() for i in items]
