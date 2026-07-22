from __future__ import annotations
import json
from typing import Optional

from libs.schemas.memory import TrackEvent, TrackSequence
from libs.config.settings import settings

MAX_EVENTS_PER_TRACK = settings.max_events_per_track


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

    def get_alert_by_id(self, alert_id: str) -> Optional[str]:
        """Return the raw alert JSON for a given alert_id or None."""
        # Scan recent alerts across camera sets — simple linear search
        pattern = "alerts:*"
        for key in self._r.keys(pattern):
            items = self._r.zrange(key, 0, -1)
            for raw in items:
                raw_s = raw if isinstance(raw, str) else raw.decode()
                try:
                    payload = json.loads(raw_s)
                    if payload.get("alert_id") == alert_id:
                        return raw_s
                except Exception:
                    continue
        return None

    def store_feedback(self, alert_id: str, verdict: str, operator_id: str, notes: str, timestamp_ms: float) -> None:
        """Store feedback as a Redis hash at key feedback:{alert_id}."""
        key = f"feedback:{alert_id}"
        self._r.hset(key, mapping={
            "verdict": verdict,
            "operator_id": operator_id,
            "notes": notes,
            "timestamp_ms": timestamp_ms,
        })

    def get_feedback(self, alert_id: str) -> Optional[str]:
        """Return the verdict string for an alert, or None."""
        key = f"feedback:{alert_id}"
        if not self._r.exists(key):
            return None
        verdict = self._r.hget(key, "verdict")
        return verdict if isinstance(verdict, str) else (verdict.decode() if verdict else None)
