"""
tests/test_camera_registry.py

Tests for the camera registry endpoints in apps/backend/routes/cameras.py.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import create_app


@pytest.fixture
def fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeRedis()


@pytest.fixture
def client(fake_redis):
    app = create_app()
    app.state.redis = fake_redis
    return TestClient(app)


class TestListCameras:
    def test_empty_registry(self, client):
        resp = client.get("/cameras/registry")
        assert resp.status_code == 200
        assert resp.json() == []


class TestRegisterCamera:
    def test_register_new_camera(self, client):
        resp = client.post("/cameras/registry", json={
            "camera_id": "cam_test",
            "label": "Test Camera",
            "location": "Lobby",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["camera_id"] == "cam_test"
        assert data["label"] == "Test Camera"
        assert data["status"] == "online"

    def test_register_duplicate_updates(self, client):
        client.post("/cameras/registry", json={"camera_id": "cam_01", "label": "Cam 1"})
        resp = client.post("/cameras/registry", json={"camera_id": "cam_01", "label": "Cam 1 Updated"})
        assert resp.status_code == 200
        assert resp.json()["label"] == "Cam 1 Updated"


class TestGetCamera:
    def test_get_existing(self, client):
        client.post("/cameras/registry", json={"camera_id": "cam_01", "label": "Front Door"})
        resp = client.get("/cameras/registry/cam_01")
        assert resp.status_code == 200
        assert resp.json()["label"] == "Front Door"

    def test_get_missing(self, client):
        resp = client.get("/cameras/registry/nonexistent")
        assert resp.status_code == 404


class TestDeleteCamera:
    def test_delete_existing(self, client):
        client.post("/cameras/registry", json={"camera_id": "cam_01", "label": "Front Door"})
        resp = client.delete("/cameras/registry/cam_01")
        assert resp.status_code == 200
        resp2 = client.get("/cameras/registry/cam_01")
        assert resp2.status_code == 404

    def test_delete_missing(self, client):
        resp = client.delete("/cameras/registry/nonexistent")
        assert resp.status_code == 404


class TestCameraHealth:
    def test_health_returns_camera(self, client):
        client.post("/cameras/registry", json={"camera_id": "cam_01", "label": "Front Door"})
        resp = client.get("/cameras/registry/cam_01/health")
        assert resp.status_code == 200
        assert resp.json()["camera_id"] == "cam_01"

    def test_health_missing(self, client):
        resp = client.get("/cameras/registry/nonexistent/health")
        assert resp.status_code == 404


class TestCameraFps:
    def test_update_fps(self, client):
        client.post("/cameras/registry", json={"camera_id": "cam_01", "label": "Front Door"})
        resp = client.post("/cameras/registry/cam_01/fps", json={"fps": 25.5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["fps"] == 25.5
        assert data["status"] == "online"

    def test_update_fps_zero_sets_offline(self, client):
        client.post("/cameras/registry", json={"camera_id": "cam_01", "label": "Front Door"})
        resp = client.post("/cameras/registry/cam_01/fps", json={"fps": 0})
        assert resp.status_code == 200
        assert resp.json()["status"] == "offline"

    def test_update_fps_missing_camera(self, client):
        resp = client.post("/cameras/registry/nonexistent/fps", json={"fps": 10})
        assert resp.status_code == 404
