"""
apps/backend/routes/zones.py — Zone adaptive baseline statistics endpoint.

GET /zones/{name}/stats
    Returns Welford running statistics for the named zone.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from services.memory.baseline import ZoneBaseline, _ZONE_NAME_RE

router = APIRouter(prefix="/zones", tags=["zones"])


class ZoneStatsResponse(BaseModel):
    zone:     str
    count:    int
    mean:     float
    variance: float
    std:      float
    m2:       float


def _get_redis(request: Request):
    try:
        return request.app.state.redis
    except AttributeError:
        raise HTTPException(status_code=503, detail="Redis not initialised in app.state")


@router.get("/{name}/stats", response_model=ZoneStatsResponse)
def get_zone_stats(name: str, redis=Depends(_get_redis)) -> ZoneStatsResponse:
    """
    Return adaptive dwell-time statistics for *name* zone.

    Statistics are computed incrementally via Welford's algorithm and
    persisted in Redis under ``zone:{name}:stats``.
    """
    if not _ZONE_NAME_RE.match(name):
        raise HTTPException(status_code=422, detail="Invalid zone name")
    stats = ZoneBaseline(redis, name).get_stats()
    return ZoneStatsResponse(**stats)
