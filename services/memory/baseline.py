"""
baseline.py — Adaptive anomaly baseline per surveillance zone.

Uses Welford's online algorithm to maintain a running mean and variance of
dwell times without batch recomputation.  Statistics are persisted in Redis
under ``zone:{name}:stats`` so they survive restarts.

Anomaly rule
------------
    dwell > mean + 2.5 * std
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Optional

_ZONE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

STATS_TTL = 0          # 0 = no expiry — stats should persist indefinitely
ANOMALY_THRESHOLD = 2.5
MIN_COUNT_FOR_ANOMALY = 10   # need enough samples before flagging outliers


@dataclass
class WelfordStats:
    count: int   = 0
    mean:  float = 0.0
    m2:    float = 0.0   # sum of squared deviations (Welford accumulator)

    @property
    def variance(self) -> float:
        # Sample variance (Bessel's correction) — correct for anomaly detection
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


class ZoneBaseline:
    """
    Per-zone adaptive dwell-time baseline backed by Redis.

    Parameters
    ----------
    redis_client:
        Connected ``redis.Redis`` (or FakeRedis for tests).
    zone_name:
        Logical zone identifier, e.g. ``"restricted_exit"``.
    """

    def __init__(self, redis_client, zone_name: str) -> None:
        if not _ZONE_NAME_RE.match(zone_name):
            raise ValueError(
                f"Invalid zone name '{zone_name}'. "
                "Only alphanumeric characters, hyphens, and underscores are allowed (max 64 chars)."
            )
        self._r    = redis_client
        self._zone = zone_name
        self._key  = f"zone:{zone_name}:stats"
        self._stats: Optional[WelfordStats] = None   # lazy-loaded

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, dwell: float) -> None:
        """Ingest one dwell-time observation and persist updated stats."""
        s = self._load()
        s.count += 1
        delta   = dwell - s.mean
        s.mean += delta / s.count
        delta2  = dwell - s.mean
        s.m2   += delta * delta2
        self._save(s)

    def is_anomalous(self, dwell: float) -> bool:
        """
        Return True when *dwell* exceeds mean + 2.5 * std.

        Returns False until MIN_COUNT_FOR_ANOMALY samples have been collected
        so early noise doesn't produce false positives.
        Returns False when std == 0 (all identical values) to avoid flagging
        any noise as anomalous.
        """
        s = self._load()
        if s.count < MIN_COUNT_FOR_ANOMALY:
            return False
        if s.std == 0:
            return False
        return dwell > s.mean + ANOMALY_THRESHOLD * s.std

    def get_stats(self) -> dict:
        """Return serialisable stats dict for the API response."""
        s = self._load()
        return {
            "zone":     self._zone,
            "count":    s.count,
            "mean":     round(s.mean, 4),
            "variance": round(s.variance, 4),
            "std":      round(s.std, 4),
            "m2":       round(s.m2, 6),
        }

    # ── Redis helpers ─────────────────────────────────────────────────────────

    def _load(self) -> WelfordStats:
        if self._stats is not None:
            return self._stats
        raw = self._r.get(self._key)
        if raw is None:
            self._stats = WelfordStats()
        else:
            d = json.loads(raw)
            self._stats = WelfordStats(
                count = d["count"],
                mean  = d["mean"],
                m2    = d["m2"],
            )
        return self._stats

    def _save(self, s: WelfordStats) -> None:
        payload = json.dumps({"count": s.count, "mean": s.mean, "m2": s.m2})
        if STATS_TTL:
            self._r.setex(self._key, STATS_TTL, payload)
        else:
            self._r.set(self._key, payload)
