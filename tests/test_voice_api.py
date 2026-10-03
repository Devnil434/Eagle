"""
tests/test_voice_api.py

Tests for the voice query endpoints in apps/backend/routes/voice.py.
"""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import create_app
from apps.backend.deps import get_store


@pytest.fixture
def client(monkeypatch, tmp_path):
    history = tmp_path / "voice_history.yaml"
    monkeypatch.setenv("VOICE_HISTORY_PATH", str(history))
    app = create_app()
    return TestClient(app)


@pytest.fixture
def store_with_alerts(monkeypatch):
    import redis
    r = redis.Redis()
    key = "alerts:cam_01"
    alert = {
        "alert_id": "a1",
        "track_id": 1,
        "camera_id": "cam_01",
        "label": "Suspicious",
        "confidence": 0.9,
        "severity_score": 0.85,
        "reason": "person lingering near keypad",
        "key_signal": "lingering",
        "timestamp_ms": time.time() * 1000,
        "vlm_captions": [],
    }
    r.zadd(key, {json.dumps(alert): time.time() * 1000})
    yield
    r.delete(key)


class TestVoiceQuery:
    def test_empty_query_rejected(self, client):
        resp = client.post("/voice/query", json={"query": "   "})
        assert resp.status_code == 422

    def test_query_returns_matches(self, client, store_with_alerts):
        resp = client.post("/voice/query", json={"query": "lingering keypad"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_matches"] >= 1
        assert "suggestions" in data
        assert len(data["suggestions"]) == 3

    def test_query_no_matches(self, client, store_with_alerts):
        resp = client.post("/voice/query", json={"query": "completely unrelated"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_matches"] == 0

    def test_query_persists_history(self, client, store_with_alerts):
        client.post("/voice/query", json={"query": "lingering"})
        resp = client.get("/voice/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["queries"]) >= 1
        assert data["queries"][-1]["query"] == "lingering"


class TestVoiceHistory:
    def test_empty_history(self, client):
        resp = client.get("/voice/history")
        assert resp.status_code == 200
        assert resp.json()["queries"] == []
