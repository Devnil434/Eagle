"""
tests/integration/test_bookmarks.py

Integration tests for the bookmarks and investigation workspace API.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import apps.backend.main as backend
from apps.backend.routes import bookmarks


@pytest.fixture()
def app():
    original = bookmarks._bookmarks, bookmarks._investigations, bookmarks._investigation_events, bookmarks._bookmark_notes
    bookmarks._bookmarks = {}
    bookmarks._investigations = {}
    bookmarks._investigation_events = {}
    bookmarks._bookmark_notes = {}
    yield backend.app
    bookmarks._bookmarks, bookmarks._investigations, bookmarks._investigation_events, bookmarks._bookmark_notes = original


@pytest.mark.asyncio
async def test_create_bookmark(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/bookmarks?operator_id=op-1",
            json={"alert_id": "alert-1", "camera_id": "cam_01", "track_id": 1, "label": "suspicious"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["alert_id"] == "alert-1"
    assert body["label"] == "suspicious"
    assert "id" in body


@pytest.mark.asyncio
async def test_list_bookmarks(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/bookmarks?operator_id=op-1",
            json={"alert_id": "alert-1", "camera_id": "cam_01", "track_id": 1, "label": "suspicious"},
        )
        response = await client.get("/bookmarks?operator_id=op-1")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["label"] == "suspicious"


@pytest.mark.asyncio
async def test_delete_bookmark(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create = await client.post(
            "/bookmarks?operator_id=op-1",
            json={"alert_id": "alert-1", "camera_id": "cam_01", "track_id": 1, "label": "suspicious"},
        )
        bm_id = create.json()["id"]
        response = await client.delete(f"/bookmarks/{bm_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] == bm_id


@pytest.mark.asyncio
async def test_create_investigation(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/bookmarks/investigations?operator_id=op-1",
            json={"name": "Incident A", "description": "Test investigation"},
        )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Incident A"
    assert body["event_count"] == 0


@pytest.mark.asyncio
async def test_add_event_to_investigation(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        inv = await client.post(
            "/bookmarks/investigations?operator_id=op-1",
            json={"name": "Incident A"},
        )
        inv_id = inv.json()["id"]
        response = await client.post(
            f"/bookmarks/investigations/{inv_id}/events?operator_id=op-1",
            json={"alert_id": "alert-2", "camera_id": "cam_02", "track_id": 2, "label": "loitering"},
        )
    assert response.status_code == 201
    body = response.json()
    assert body["alert_id"] == "alert-2"


@pytest.mark.asyncio
async def test_investigation_timeline(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        inv = await client.post(
            "/bookmarks/investigations?operator_id=op-1",
            json={"name": "Incident A"},
        )
        inv_id = inv.json()["id"]
        await client.post(
            f"/bookmarks/investigations/{inv_id}/events?operator_id=op-1",
            json={"alert_id": "alert-2", "camera_id": "cam_02", "track_id": 2, "label": "loitering"},
        )
        response = await client.get(f"/bookmarks/investigations/{inv_id}/timeline")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["label"] == "loitering"


@pytest.mark.asyncio
async def test_search_bookmarks(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/bookmarks?operator_id=op-1",
            json={"alert_id": "alert-1", "camera_id": "cam_01", "track_id": 1, "label": "suspicious"},
        )
        response = await client.get("/bookmarks/search?operator_id=op-1&q=susp")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["label"] == "suspicious"


@pytest.mark.asyncio
async def test_export_investigation(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        inv = await client.post(
            "/bookmarks/investigations?operator_id=op-1",
            json={"name": "Incident A"},
        )
        inv_id = inv.json()["id"]
        await client.post(
            f"/bookmarks/investigations/{inv_id}/events?operator_id=op-1",
            json={"alert_id": "alert-2", "camera_id": "cam_02", "track_id": 2, "label": "loitering"},
        )
        response = await client.get(f"/bookmarks/investigations/{inv_id}/export")
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "json"
    assert "events" in body["data"]
