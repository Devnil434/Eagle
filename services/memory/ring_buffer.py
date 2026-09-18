from __future__ import annotations
import heapq
import json
from itertools import islice
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

    def get_alerts_in_range(
        self,
        start_ms: float,
        end_ms: float,
        camera_id: Optional[str] = None,
        limit: int = 5_000,
    ) -> list[str]:
        """Return raw alert JSON for the window [start_ms, end_ms], oldest first.

        Args:
            start_ms:  Inclusive lower bound (epoch milliseconds).
            end_ms:    Inclusive upper bound (epoch milliseconds).
            camera_id: Restrict to one camera; None aggregates every camera.
            limit:     Hard cap on returned alerts, protecting the caller from
                       loading an unbounded window into memory.
        """
        if start_ms > end_ms or limit <= 0:
            return []

        keys = [f"alerts:{camera_id}"] if camera_id else self._alert_keys()

        # Redis returns each camera's slice pre-sorted by score, so merging keeps
        # the timeline chronological across cameras.  Trimming to `limit` after
        # each merge holds the accumulator at `limit` however many cameras exist,
        # instead of letting it grow to `limit x cameras` before a final cut.
        # The result is unchanged: the earliest `limit` alerts overall can only
        # come from the earliest `limit` of each camera.
        scored: list[tuple[float, str]] = []
        for key in keys:
            items = self._r.zrangebyscore(
                key, start_ms, end_ms, start=0, num=limit, withscores=True
            )
            batch = [
                (score, raw if isinstance(raw, str) else raw.decode())
                for raw, score in items
            ]
            scored = list(
                islice(heapq.merge(scored, batch, key=lambda pair: pair[0]), limit)
            )

        return [raw for _, raw in scored]

    def _alert_keys(self) -> list[str]:
        """Discover per-camera alert keys via SCAN (never KEYS, which blocks)."""
        keys = []
        for key in self._r.scan_iter(match="alerts:*"):
            keys.append(key if isinstance(key, str) else key.decode())
        return keys

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

    def get_feedback_bulk(self, alert_ids: list[str]) -> dict[str, str]:
        """Fetch verdicts for many alerts in a single round-trip.

        Reports resolve feedback for every alert in the window, so the per-alert
        `get_feedback` would cost one round-trip each.  Alerts without feedback
        are omitted from the result.
        """
        if not alert_ids:
            return {}

        pipe = self._r.pipeline()
        for alert_id in alert_ids:
            pipe.hget(f"feedback:{alert_id}", "verdict")
        verdicts = pipe.execute()

        resolved: dict[str, str] = {}
        for alert_id, verdict in zip(alert_ids, verdicts):
            if verdict:
                resolved[alert_id] = (
                    verdict if isinstance(verdict, str) else verdict.decode()
                )
        return resolved
