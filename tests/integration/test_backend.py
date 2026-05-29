"""
tests/integration/test_backend.py

Integration tests for the Eagle FastAPI backend (Phase 5).

Coverage
--------
- GET /health  → {"status": "ok", "redis": "connected"} when Redis is up
- GET /health  → {"status": "degraded", ...} when Redis is unreachable
- GET /tracks  → returns active track IDs for a camera
- GET /tracks  → returns empty list + error key when Redis is down
- GET /tracks  → filters out non-ACTIVE tracks
- GET /tracks  → respects camera_id query parameter
- GET /tracks  → default camera_id is "cam_01"

All tests use ``httpx.AsyncClient`` with the ASGI transport so no real
server or Redis instance is required.  Redis is replaced by fakeredis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

httpx = pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient


# ── App import (deferred so patches can be applied first) ─────────────────────

@pytest.fixture()
def fake_redis_client():
    """In-memory fakeredis instance (sync) shared across a test."""
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeRedis(decode_responses=False)


@pytest.fixture()
def app_with_redis(fake_redis_client):
    """
    Return the FastAPI app with its internal ``_redis`` replaced by a
    fakeredis instance so no real Redis server is needed.
    """
    import apps.backend.main as backend

    original = backend._redis
    backend._redis = fake_redis_client
    yield backend.app
    backend._redis = original


@pytest.fixture()
def app_without_redis():
    """
    Return the FastAPI app with Redis forcibly set to None to simulate
    an unreachable Redis server.
    """
    import apps.backend.main as backend

    original = backend._redis
    backend._redis = None
    yield backend.app
    backend._redis = original


# ── /health tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_ok_when_redis_connected(app_with_redis):
    """
    /health must return {"status": "ok", "redis": "connected"} when
    Redis responds to PING.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app_with_redis), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["redis"] == "connected"


@pytest.mark.asyncio
async def test_health_degraded_when_redis_unavailable(app_without_redis):
    """
    /health must return {"status": "degraded"} when Redis is unreachable.
    The ``_get_redis()`` helper returns None, so the endpoint short-circuits.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app_without_redis), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "redis" in body


@pytest.mark.asyncio
async def test_health_degraded_when_redis_ping_fails(app_with_redis):
    """
    /health must return degraded when Redis is initialised but PING raises.
    Simulates a mid-session Redis failure.
    """
    import apps.backend.main as backend

    broken = MagicMock()
    broken.ping.side_effect = Exception("connection reset")
    original = backend._redis
    backend._redis = broken

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app_with_redis), base_url="http://test"
        ) as client:
            response = await client.get("/health")
    finally:
        backend._redis = original

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "connection reset" in body["redis"]


@pytest.mark.asyncio
async def test_health_response_has_required_keys(app_with_redis):
    """Response body must always contain both 'status' and 'redis' keys."""
    async with AsyncClient(
        transport=ASGITransport(app=app_with_redis), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    body = response.json()
    assert "status" in body
    assert "redis" in body


# ── /tracks tests ─────────────────────────────────────────────────────────────

def _seed_track(redis_client, camera_id: str, track_id: int, state: str = "ACTIVE") -> None:
    """Write a minimal track record into fakeredis."""
    key = f"track:{camera_id}:{track_id}"
    record = {
        "camera_id": camera_id,
        "track_id": track_id,
        "state": state,
        "global_id": f"gid-{track_id}",
        "dwell_time_seconds": 0.0,
        "zones_present": [],
        "born_frame": 0,
        "last_seen_frame": 10,
    }
    redis_client.set(key, json.dumps(record).encode())


@pytest.mark.asyncio
async def test_tracks_returns_active_ids(app_with_redis, fake_redis_client):
    """
    /tracks must return the track_ids of all ACTIVE tracks for the camera.
    """
    _seed_track(fake_redis_client, "cam_01", 1, "ACTIVE")
    _seed_track(fake_redis_client, "cam_01", 3, "ACTIVE")
    _seed_track(fake_redis_client, "cam_01", 7, "ACTIVE")

    async with AsyncClient(
        transport=ASGITransport(app=app_with_redis), base_url="http://test"
    ) as client:
        response = await client.get("/tracks", params={"camera_id": "cam_01"})

    assert response.status_code == 200
    body = response.json()
    assert body["camera_id"] == "cam_01"
    assert sorted(body["track_ids"]) == [1, 3, 7]


@pytest.mark.asyncio
async def test_tracks_excludes_non_active_states(app_with_redis, fake_redis_client):
    """
    /tracks must only return ACTIVE tracks — LOST and DEAD tracks are excluded.
    """
    _seed_track(fake_redis_client, "cam_01", 10, "ACTIVE")
    _seed_track(fake_redis_client, "cam_01", 20, "LOST")
    _seed_track(fake_redis_client, "cam_01", 30, "DEAD")

    async with AsyncClient(
        transport=ASGITransport(app=app_with_redis), base_url="http://test"
    ) as client:
        response = await client.get("/tracks", params={"camera_id": "cam_01"})

    body = response.json()
    assert body["track_ids"] == [10]


@pytest.mark.asyncio
async def test_tracks_empty_when_no_tracks(app_with_redis):
    """
    /tracks must return an empty list when no tracks exist for the camera.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app_with_redis), base_url="http://test"
    ) as client:
        response = await client.get("/tracks", params={"camera_id": "cam_empty"})

    assert response.status_code == 200
    body = response.json()
    assert body["track_ids"] == []


@pytest.mark.asyncio
async def test_tracks_respects_camera_id_param(app_with_redis, fake_redis_client):
    """
    /tracks must only return tracks for the requested camera_id, not others.
    """
    _seed_track(fake_redis_client, "cam_01", 1, "ACTIVE")
    _seed_track(fake_redis_client, "cam_02", 99, "ACTIVE")

    async with AsyncClient(
        transport=ASGITransport(app=app_with_redis), base_url="http://test"
    ) as client:
        response = await client.get("/tracks", params={"camera_id": "cam_02"})

    body = response.json()
    assert body["camera_id"] == "cam_02"
    assert body["track_ids"] == [99]


@pytest.mark.asyncio
async def test_tracks_default_camera_id(app_with_redis, fake_redis_client):
    """
    /tracks without a camera_id param must default to 'cam_01'.
    """
    _seed_track(fake_redis_client, "cam_01", 5, "ACTIVE")

    async with AsyncClient(
        transport=ASGITransport(app=app_with_redis), base_url="http://test"
    ) as client:
        response = await client.get("/tracks")

    body = response.json()
    assert body["camera_id"] == "cam_01"
    assert 5 in body["track_ids"]


@pytest.mark.asyncio
async def test_tracks_degraded_when_redis_unavailable(app_without_redis):
    """
    /tracks must return an empty list and an 'error' key when Redis is down.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app_without_redis), base_url="http://test"
    ) as client:
        response = await client.get("/tracks", params={"camera_id": "cam_01"})

    assert response.status_code == 200
    body = response.json()
    assert body["track_ids"] == []
    assert "error" in body


@pytest.mark.asyncio
async def test_tracks_ids_are_sorted(app_with_redis, fake_redis_client):
    """
    /tracks must return track IDs in ascending sorted order.
    """
    for tid in [7, 2, 5, 1]:
        _seed_track(fake_redis_client, "cam_01", tid, "ACTIVE")

    async with AsyncClient(
        transport=ASGITransport(app=app_with_redis), base_url="http://test"
    ) as client:
        response = await client.get("/tracks", params={"camera_id": "cam_01"})

    body = response.json()
    assert body["track_ids"] == sorted(body["track_ids"])
