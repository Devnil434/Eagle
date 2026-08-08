"""
Prevent the same (track_id, zone) pair from firing alerts every few seconds.
Uses Redis SETEX with a configurable window.
"""
from __future__ import annotations
import os
import logging
from typing import Optional
import redis

logger = logging.getLogger(__name__)

REDIS_URL            = os.getenv("REDIS_URL", "redis://localhost:6379")
DEDUP_WINDOW_SECONDS = int(os.getenv("ALERT_DEDUP_WINDOW_SECONDS", "300"))


class AlertDeduplicator:
    """
    Blocks duplicate alerts within DEDUP_WINDOW_SECONDS.
    Key: dedup:{track_id}:{zone}  → TTL = DEDUP_WINDOW_SECONDS
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        window_seconds: int = DEDUP_WINDOW_SECONDS,
    ) -> None:
        self._r      = redis_client or redis.from_url(REDIS_URL, decode_responses=True)
        self._window = window_seconds

    def is_duplicate(self, track_id: int, zone: str) -> bool:
        return bool(self._r.exists(f"dedup:{track_id}:{zone}"))

    def mark_alerted(self, track_id: int, zone: str) -> None:
        self._r.setex(f"dedup:{track_id}:{zone}", self._window, "1")
        logger.debug("Dedup key set  track=%d zone=%s  TTL=%ds",
                     track_id, zone, self._window)

    def reset(self, track_id: int, zone: str) -> None:
        self._r.delete(f"dedup:{track_id}:{zone}")
