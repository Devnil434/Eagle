"""
tests/integration/test_rules.py

Integration tests for the alert rules API.

The endpoint reports what the pipeline is currently enforcing, so the tests
point the loader at a temporary file and assert the response reflects it.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import apps.backend.main as backend
from libs.config.rule_loader import RuleConfigLoader
from services.rules import provider

RULES_YAML = """
rules:
  - id: restricted_door_person
    description: Person at the secure door.
    object_types: [person]
    zones: [restricted_door]
    min_confidence: 0.6
    cooldown_seconds: 90
  - id: after_hours_vehicle
    object_types: [vehicle]
    time_windows:
      - start: "19:00"
        end: "07:00"
  - id: noisy_corridor
    enabled: false
    zones: [safe_corridor]
"""


@pytest.fixture()
def app(tmp_path: Path):
    path = tmp_path / "alert_rules.yaml"
    path.write_text(textwrap.dedent(RULES_YAML), encoding="utf-8")

    original = provider._loader
    provider._loader = RuleConfigLoader(path, reload_interval=0)
    yield backend.app
    provider._loader = original


@pytest.fixture()
def app_without_rules(tmp_path: Path):
    original = provider._loader
    provider._loader = RuleConfigLoader(tmp_path / "absent.yaml", reload_interval=0)
    yield backend.app
    provider._loader = original


async def get(app, url: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(url)


@pytest.mark.asyncio
async def test_list_rules_returns_them_in_evaluation_order(app):
    response = await get(app, "/rules")

    assert response.status_code == 200
    assert [rule["id"] for rule in response.json()] == [
        "restricted_door_person",
        "after_hours_vehicle",
        "noisy_corridor",
    ]


@pytest.mark.asyncio
async def test_list_rules_exposes_enabled_state(app):
    body = await get(app, "/rules")

    states = {rule["id"]: rule["enabled"] for rule in body.json()}
    assert states == {
        "restricted_door_person": True,
        "after_hours_vehicle": True,
        "noisy_corridor": False,
    }


@pytest.mark.asyncio
async def test_enabled_only_filters_disabled_rules(app):
    response = await get(app, "/rules?enabled_only=true")

    ids = [rule["id"] for rule in response.json()]
    assert "noisy_corridor" not in ids
    assert len(ids) == 2


@pytest.mark.asyncio
async def test_rule_fields_survive_serialisation(app):
    response = await get(app, "/rules/restricted_door_person")

    assert response.status_code == 200
    rule = response.json()
    assert rule["object_types"] == ["person"]
    assert rule["zones"] == ["restricted_door"]
    assert rule["min_confidence"] == 0.6
    assert rule["cooldown_seconds"] == 90
    assert rule["description"] == "Person at the secure door."


@pytest.mark.asyncio
async def test_time_windows_are_serialised(app):
    response = await get(app, "/rules/after_hours_vehicle")

    window = response.json()["time_windows"][0]
    assert window["start"].startswith("19:00")
    assert window["end"].startswith("07:00")


@pytest.mark.asyncio
async def test_unknown_rule_returns_404(app):
    response = await get(app, "/rules/does_not_exist")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_no_rules_file_returns_an_empty_list(app_without_rules):
    """Absence of config is a valid state, not a server error."""
    response = await get(app_without_rules, "/rules")

    assert response.status_code == 200
    assert response.json() == []
