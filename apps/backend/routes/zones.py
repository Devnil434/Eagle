"""
apps/backend/routes/zones.py — Zone CRUD + adaptive baseline statistics endpoints.

CRUD
----
GET    /zones            List all zones
GET    /zones/{zone_id}  Get a single zone by ID
POST   /zones            Save zones (replaces current config)
DELETE /zones/{zone_id}  Delete a zone by ID

Stats
-----
GET    /zones/{name}/stats
    Returns Welford running statistics for the named zone.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from services.memory.baseline import ZoneBaseline, _ZONE_NAME_RE

router = APIRouter(prefix="/zones", tags=["zones"])

logger = logging.getLogger(__name__)

# ── Config path resolution ────────────────────────────────────────────────────

DEFAULT_ZONES_CONFIG = Path(__file__).resolve().parents[2] / "config" / "zones.yaml"


def _resolve_zones_path() -> Path:
    env_path = os.environ.get("ZONES_CONFIG_PATH")
    return Path(env_path) if env_path else DEFAULT_ZONES_CONFIG


# ── Models ────────────────────────────────────────────────────────────────────


class ZonePoint(BaseModel):
    x: float
    y: float


class ZoneModel(BaseModel):
    id: str
    name: str
    polygon: list[ZonePoint]
    alert_on_entry: bool = False
    color_hex: str = "#FF0000"


class ZoneListResponse(BaseModel):
    zones: list[ZoneModel]


class ZoneStatsResponse(BaseModel):
    zone:     str
    count:    int
    mean:     float
    variance: float
    std:      float
    m2:       float


# ── YAML helpers ──────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"camera_id": "unknown", "zones": []}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "zones" not in data:
        data["zones"] = []
    return data


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _zones_path() -> Path:
    return _resolve_zones_path()


# ── Dependency helpers ────────────────────────────────────────────────────────


def _get_redis(request: Request):
    try:
        return request.app.state.redis
    except AttributeError:
        raise HTTPException(status_code=503, detail="Redis not initialised in app.state")


# ── CRUD routes ───────────────────────────────────────────────────────────────


@router.get("", response_model=ZoneListResponse)
def list_zones() -> ZoneListResponse:
    """Return all zones from the YAML config."""
    path = _zones_path()
    data = _load_yaml(path)
    zones = []
    for z in data.get("zones", []):
        polygon = [ZonePoint(x=p[0], y=p[1]) for p in z.get("polygon", [])]
        zones.append(
            ZoneModel(
                id=str(z.get("id", z.get("name", ""))),
                name=z.get("name", ""),
                polygon=polygon,
                alert_on_entry=z.get("alert_on_entry", False),
                color_hex=z.get("color_hex", "#FF0000"),
            )
        )
    return ZoneListResponse(zones=zones)


@router.get("/{zone_id}", response_model=ZoneModel)
def get_zone(zone_id: str) -> ZoneModel:
    """Return a single zone by its ID (or name)."""
    path = _zones_path()
    data = _load_yaml(path)
    for z in data.get("zones", []):
        zid = str(z.get("id", z.get("name", "")))
        if zid == zone_id or z.get("name") == zone_id:
            polygon = [ZonePoint(x=p[0], y=p[1]) for p in z.get("polygon", [])]
            return ZoneModel(
                id=zid,
                name=z.get("name", ""),
                polygon=polygon,
                alert_on_entry=z.get("alert_on_entry", False),
                color_hex=z.get("color_hex", "#FF0000"),
            )
    raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")


@router.post("", response_model=ZoneListResponse)
def save_zones(payload: ZoneListResponse) -> ZoneListResponse:
    """Replace all zones with the provided list and persist to YAML."""
    path = _zones_path()
    data = _load_yaml(path)
    data["camera_id"] = data.get("camera_id", "unknown")
    data["zones"] = []
    for z in payload.zones:
        data["zones"].append(
            {
                "id": z.id,
                "name": z.name,
                "polygon": [[p.x, p.y] for p in z.polygon],
                "alert_on_entry": z.alert_on_entry,
                "color_hex": z.color_hex,
            }
        )
    _save_yaml(path, data)
    logger.info("Saved %d zone(s) to %s", len(data["zones"]), path)
    return payload


@router.delete("/{zone_id}")
def delete_zone(zone_id: str) -> dict[str, str]:
    """Delete a zone by ID (or name) and persist the change."""
    path = _zones_path()
    data = _load_yaml(path)
    original_len = len(data.get("zones", []))
    data["zones"] = [
        z for z in data.get("zones", [])
        if str(z.get("id", z.get("name", ""))) != zone_id and z.get("name") != zone_id
    ]
    if len(data["zones"]) == original_len:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")
    _save_yaml(path, data)
    logger.info("Deleted zone '%s' from %s", zone_id, path)
    return {"detail": f"Zone '{zone_id}' deleted"}


# ── Stats route (existing) ────────────────────────────────────────────────────


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
