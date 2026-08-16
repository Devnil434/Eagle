"""
tests/test_zones_api.py

Tests for the zone CRUD endpoints in apps/backend/routes/zones.py.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import create_app


@pytest.fixture
def tmp_zones_file(tmp_path):
    zones = tmp_path / "zones.yaml"
    zones.write_text(
        "camera_id: cam_01\nzones:\n  - name: zone_a\n    polygon: [[0,0],[10,0],[10,10],[0,10]]\n    alert_on_entry: true\n    color_hex: \"#ff0000\"\n",
        encoding="utf-8",
    )
    return zones


@pytest.fixture
def client(tmp_zones_file, monkeypatch):
    monkeypatch.setenv("ZONES_CONFIG_PATH", str(tmp_zones_file))
    app = create_app()
    return TestClient(app)


class TestListZones:
    def test_returns_zones(self, client):
        resp = client.get("/zones")
        assert resp.status_code == 200
        data = resp.json()
        assert "zones" in data
        assert len(data["zones"]) == 1
        assert data["zones"][0]["name"] == "zone_a"

    def test_empty_when_no_file(self, client, tmp_path, monkeypatch):
        missing = tmp_path / "missing.yaml"
        monkeypatch.setenv("ZONES_CONFIG_PATH", str(missing))
        app = create_app()
        c = TestClient(app)
        resp = c.get("/zones")
        assert resp.status_code == 200
        assert resp.json()["zones"] == []


class TestGetZone:
    def test_get_by_name(self, client):
        resp = client.get("/zones/zone_a")
        assert resp.status_code == 200
        assert resp.json()["name"] == "zone_a"

    def test_get_by_id(self, client):
        resp = client.get("/zones/zone_a")
        assert resp.status_code == 200

    def test_404_for_missing(self, client):
        resp = client.get("/zones/nonexistent")
        assert resp.status_code == 404


class TestSaveZones:
    def test_save_replaces_zones(self, client):
        payload = {
            "zones": [
                {
                    "id": "new_zone",
                    "name": "new_zone",
                    "polygon": [{"x": 0, "y": 0}, {"x": 5, "y": 0}, {"x": 5, "y": 5}],
                    "alert_on_entry": False,
                    "color_hex": "#00ff00",
                }
            ]
        }
        resp = client.post("/zones", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["zones"]) == 1
        assert resp.json()["zones"][0]["name"] == "new_zone"

    def test_save_persists_to_yaml(self, client, tmp_zones_file):
        payload = {
            "zones": [
                {
                    "id": "z1",
                    "name": "z1",
                    "polygon": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 2, "y": 2}],
                    "alert_on_entry": True,
                    "color_hex": "#ffffff",
                }
            ]
        }
        client.post("/zones", json=payload)
        content = tmp_zones_file.read_text(encoding="utf-8")
        assert "z1" in content
        assert "zone_a" not in content


class TestDeleteZone:
    def test_delete_existing(self, client):
        resp = client.delete("/zones/zone_a")
        assert resp.status_code == 200
        resp2 = client.get("/zones")
        assert resp2.json()["zones"] == []

    def test_delete_missing(self, client):
        resp = client.delete("/zones/nonexistent")
        assert resp.status_code == 404
