"""
tests/integration/test_collaboration.py

Integration tests for the multi-operator collaboration workspace API.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import apps.backend.main as backend
from apps.backend.routes import collaboration


@pytest.fixture()
def app():
    original = (
        collaboration._workspaces,
        collaboration._workspace_events,
        collaboration._workspace_comments,
        collaboration._workspace_assignments,
        collaboration._workspace_activity,
    )
    collaboration._workspaces = {}
    collaboration._workspace_events = {}
    collaboration._workspace_comments = {}
    collaboration._workspace_assignments = {}
    collaboration._workspace_activity = {}
    yield backend.app
    (
        collaboration._workspaces,
        collaboration._workspace_events,
        collaboration._workspace_comments,
        collaboration._workspace_assignments,
        collaboration._workspace_activity,
    ) = original


@pytest.mark.asyncio
async def test_create_workspace(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/workspaces?operator_id=op-1",
            json={"name": "Ops Room A", "description": "Main ops room"},
        )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Ops Room A"
    assert body["owner_id"] == "op-1"


@pytest.mark.asyncio
async def test_join_and_leave_workspace(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create = await client.post(
            "/workspaces?operator_id=op-1",
            json={"name": "Ops Room A"},
        )
        ws_id = create.json()["id"]
        join = await client.post(f"/workspaces/{ws_id}/join?operator_id=op-2")
        assert join.status_code == 200
        leave = await client.post(f"/workspaces/{ws_id}/leave?operator_id=op-2")
        assert leave.status_code == 200


@pytest.mark.asyncio
async def test_add_event_and_comment(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create = await client.post(
            "/workspaces?operator_id=op-1",
            json={"name": "Ops Room A"},
        )
        ws_id = create.json()["id"]
        evt = await client.post(
            f"/workspaces/{ws_id}/events?operator_id=op-1",
            json={"alert_id": "alert-1", "camera_id": "cam_01", "track_id": 1, "label": "suspicious"},
        )
        assert evt.status_code == 201
        evt_id = evt.json()["id"]
        comment = await client.post(
            f"/workspaces/{ws_id}/events/{evt_id}/comments",
            json={"operator_id": "op-1", "content": "Looks bad"},
        )
        assert comment.status_code == 201


@pytest.mark.asyncio
async def test_assignment_and_activity(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        create = await client.post(
            "/workspaces?operator_id=op-1",
            json={"name": "Ops Room A"},
        )
        ws_id = create.json()["id"]
        await client.post(f"/workspaces/{ws_id}/join?operator_id=op-2")
        assign = await client.post(
            f"/workspaces/{ws_id}/assignments?assigned_by=op-1",
            json={"operator_id": "op-2", "incident_id": "alert-1", "description": "Review"},
        )
        assert assign.status_code == 201
        activity = await client.get(f"/workspaces/{ws_id}/activity")
        assert activity.status_code == 200
        body = activity.json()
        assert len(body) > 0
