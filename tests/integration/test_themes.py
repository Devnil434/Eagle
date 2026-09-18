"""
tests/integration/test_themes.py

Integration tests for the detection overlay theme management API.
"""
from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

import apps.backend.main as backend
from apps.backend.routes import themes


@pytest.fixture()
def app():
    original = themes.PRESETS.copy()
    themes.THEMES_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "config", "themes", "test_custom_themes.json")
    if os.path.exists(themes.THEMES_FILE):
        os.remove(themes.THEMES_FILE)
    yield backend.app
    if os.path.exists(themes.THEMES_FILE):
        os.remove(themes.THEMES_FILE)
    themes.PRESETS = original


@pytest.mark.asyncio
async def test_list_themes(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/themes")
    assert response.status_code == 200
    body = response.json()
    assert "default" in body
    assert "highContrast" in body
    assert body["default"]["isCustom"] is False


@pytest.mark.asyncio
async def test_get_preset_theme(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/themes/default")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Default"
    assert body["isCustom"] is False


@pytest.mark.asyncio
async def test_get_theme_not_found(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/themes/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_save_custom_theme(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/themes",
            json={
                "name": "MyTheme",
                "boundingBoxColors": {"palette": ["#ff0000"], "opacity": 1.0, "borderWidth": 4},
                "label": {"visible": True, "fontSize": 12, "fontFamily": "monospace", "backgroundColor": "rgba(0,0,0,0.6)", "textColor": "#ffffff"},
                "confidence": {"visible": True, "mode": "percentage", "color": "#ffffff"},
                "trackingTrails": {"enabled": True, "maxLength": 20, "trailOpacity": 0.5, "trailWidth": 2},
                "accessibility": {"highContrast": False, "reducedMotion": False, "largeText": False},
            },
        )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "MyTheme"
    assert body["isCustom"] is True


@pytest.mark.asyncio
async def test_delete_custom_theme(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/themes",
            json={
                "name": "ToDelete",
                "boundingBoxColors": {"palette": ["#ff0000"], "opacity": 1.0, "borderWidth": 4},
                "label": {"visible": True, "fontSize": 12, "fontFamily": "monospace", "backgroundColor": "rgba(0,0,0,0.6)", "textColor": "#ffffff"},
                "confidence": {"visible": True, "mode": "percentage", "color": "#ffffff"},
                "trackingTrails": {"enabled": True, "maxLength": 20, "trailOpacity": 0.5, "trailWidth": 2},
                "accessibility": {"highContrast": False, "reducedMotion": False, "largeText": False},
            },
        )
        response = await client.delete("/themes/ToDelete")
    assert response.status_code == 200
    assert response.json()["deleted"] == "ToDelete"
