"""
tests/integration/test_reports.py

Integration tests for GET /reports/summary.

The store dependency is overridden with a fakeredis-backed MemoryStore, so no
Redis server is required.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

import apps.backend.main as backend
from apps.backend.deps import get_store
from libs.schemas.reasoning import ReasoningResult
from services.memory.ring_buffer import MemoryStore

WINDOW_START = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def store() -> MemoryStore:
    return MemoryStore(redis_client=fakeredis.FakeRedis(decode_responses=True))


@pytest.fixture()
def app(store):
    backend.app.dependency_overrides[get_store] = lambda: store
    yield backend.app
    backend.app.dependency_overrides.pop(get_store, None)


def seed_alert(
    store: MemoryStore,
    *,
    alert_id: str,
    offset_seconds: float = 60,
    camera_id: str = "cam_01",
    label: str = "Suspicious",
    confidence: float = 0.88,
    severity: float = 0.9,
    zone: str | None = "restricted_door",
    object_labels: list[str] | None = None,
    reason: str = "Repeated keypad interaction suggests an entry attempt.",
) -> None:
    timestamp_ms = (WINDOW_START + timedelta(seconds=offset_seconds)).timestamp() * 1000
    alert = ReasoningResult(
        track_id       = 3,
        camera_id      = camera_id,
        label          = label,
        confidence     = confidence,
        reason         = reason,
        key_signal     = "near_keypad",
        timestamp_ms   = timestamp_ms,
        severity_score = severity,
        alert_id       = alert_id,
        zone           = zone,
        object_labels  = ["person"] if object_labels is None else object_labels,
    )
    store.store_alert(
        alert_json=alert.model_dump_json(), timestamp_ms=timestamp_ms, camera_id=camera_id
    )


def window_params(hours: float = 1.0) -> dict[str, str]:
    return {
        "start": WINDOW_START.isoformat(),
        "end": (WINDOW_START + timedelta(hours=hours)).isoformat(),
    }


async def get_report(app, **params):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get("/reports/summary", params={**window_params(), **params})


# ── Formats ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_markdown_is_the_default_format(app, store):
    seed_alert(store, alert_id="a1")

    response = await get_report(app)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "# Incident Summary Report" in response.text


@pytest.mark.asyncio
async def test_json_format_returns_the_structured_summary(app, store):
    seed_alert(store, alert_id="a1", zone="keypad_area", object_labels=["person", "backpack"])

    response = await get_report(app, format="json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["total_alerts"] == 1
    assert body["suspicious_alerts"] == 1
    assert body["timeline"][0]["alert_id"] == "a1"
    assert {g["key"] for g in body["by_object_type"]} == {"person", "backpack"}
    assert [g["key"] for g in body["by_zone"]] == ["keypad_area"]


@pytest.mark.asyncio
async def test_pdf_format_returns_a_pdf_document(app, store):
    pytest.importorskip("fpdf")
    seed_alert(store, alert_id="a1")

    response = await get_report(app, format="pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


@pytest.mark.asyncio
async def test_unsupported_format_is_rejected_before_reaching_the_store(app):
    response = await get_report(app, format="docx")

    assert response.status_code == 422


# ── Window handling ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_only_alerts_inside_the_window_are_reported(app, store):
    seed_alert(store, alert_id="inside", offset_seconds=60)
    seed_alert(store, alert_id="before", offset_seconds=-7_200)
    seed_alert(store, alert_id="after", offset_seconds=7_200)

    body = (await get_report(app, format="json")).json()

    assert [e["alert_id"] for e in body["timeline"]] == ["inside"]


@pytest.mark.asyncio
async def test_window_defaults_to_the_recent_past_when_omitted(app, store):
    """With no bounds the endpoint reports the last `report_default_window_hours`."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/reports/summary", params={"format": "json"})

    assert response.status_code == 200
    body = response.json()
    assert body["window"]["end_ms"] > body["window"]["start_ms"]
    assert body["total_alerts"] == 0


@pytest.mark.asyncio
async def test_inverted_window_is_rejected(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/reports/summary",
            params={
                "start": (WINDOW_START + timedelta(hours=2)).isoformat(),
                "end": WINDOW_START.isoformat(),
            },
        )

    assert response.status_code == 422
    assert "earlier" in response.json()["detail"]


@pytest.mark.asyncio
async def test_excessive_window_is_rejected(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/reports/summary",
            params={
                "start": WINDOW_START.isoformat(),
                "end": (WINDOW_START + timedelta(days=400)).isoformat(),
            },
        )

    assert response.status_code == 422
    assert "narrower" in response.json()["detail"]


@pytest.mark.asyncio
async def test_naive_timestamps_are_interpreted_as_utc(app, store):
    seed_alert(store, alert_id="a1", offset_seconds=60)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/reports/summary",
            params={
                "start": "2026-06-10T12:00:00",
                "end": "2026-06-10T13:00:00",
                "format": "json",
            },
        )

    assert response.json()["total_alerts"] == 1


# ── Filtering ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_camera_filter_restricts_the_report(app, store):
    seed_alert(store, alert_id="a1", camera_id="cam_01")
    seed_alert(store, alert_id="a2", camera_id="cam_02")

    body = (await get_report(app, format="json", camera_id="cam_02")).json()

    assert body["cameras"] == ["cam_02"]
    assert [e["alert_id"] for e in body["timeline"]] == ["a2"]


@pytest.mark.asyncio
async def test_all_cameras_covered_by_default(app, store):
    seed_alert(store, alert_id="a1", camera_id="cam_01")
    seed_alert(store, alert_id="a2", camera_id="cam_02")

    body = (await get_report(app, format="json")).json()

    assert body["cameras"] == ["cam_01", "cam_02"]
    assert body["total_alerts"] == 2


@pytest.mark.asyncio
async def test_dismissed_alerts_can_be_excluded(app, store):
    seed_alert(store, alert_id="kept", offset_seconds=60)
    seed_alert(store, alert_id="dismissed", offset_seconds=120)
    store.store_feedback("dismissed", "dismissed", "op_1", "", 0.0)

    body = (await get_report(app, format="json", include_dismissed=False)).json()

    assert [e["alert_id"] for e in body["timeline"]] == ["kept"]


@pytest.mark.asyncio
async def test_operator_verdicts_appear_in_the_report(app, store):
    seed_alert(store, alert_id="a1")
    store.store_feedback("a1", "confirmed", "op_1", "", 0.0)

    body = (await get_report(app, format="json")).json()

    assert body["timeline"][0]["feedback"] == "confirmed"


@pytest.mark.asyncio
async def test_top_n_limits_the_spotlight(app, store):
    for i in range(4):
        seed_alert(store, alert_id=f"a{i}", offset_seconds=60 + i, severity=i / 10)

    body = (await get_report(app, format="json", top_n=2)).json()

    assert len(body["top_suspicious"]) == 2
    assert len(body["timeline"]) == 4


@pytest.mark.asyncio
async def test_top_n_is_bounded(app):
    response = await get_report(app, top_n=0)

    assert response.status_code == 422


# ── Robustness ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_window_reports_zero_alerts(app):
    response = await get_report(app)

    assert response.status_code == 200
    assert "No alerts were recorded in this window." in response.text


@pytest.mark.asyncio
async def test_corrupt_alert_does_not_sink_the_report(app, store):
    """One unreadable record must not deny the operator the rest of the window."""
    seed_alert(store, alert_id="good", offset_seconds=60)
    timestamp_ms = (WINDOW_START + timedelta(seconds=90)).timestamp() * 1000
    store.store_alert(alert_json="{not valid json", timestamp_ms=timestamp_ms)

    body = (await get_report(app, format="json")).json()

    assert [e["alert_id"] for e in body["timeline"]] == ["good"]


@pytest.mark.asyncio
async def test_legacy_alerts_without_provenance_are_grouped_as_unknown(app, store):
    """Alerts predating provenance capture still appear, bucketed explicitly."""
    timestamp_ms = (WINDOW_START + timedelta(seconds=60)).timestamp() * 1000
    legacy = {
        "track_id": 9,
        "camera_id": "cam_01",
        "label": "Suspicious",
        "confidence": 0.7,
        "reason": "Loitering by the door.",
        "key_signal": "lingering",
        "timestamp_ms": timestamp_ms,
        "vlm_captions": [],
        "severity_score": 0.5,
        "alert_id": "legacy-1",
    }
    store.store_alert(alert_json=json.dumps(legacy), timestamp_ms=timestamp_ms)

    body = (await get_report(app, format="json")).json()

    assert body["total_alerts"] == 1
    assert [g["key"] for g in body["by_zone"]] == ["unknown"]
    assert [g["key"] for g in body["by_object_type"]] == ["unknown"]


@pytest.mark.asyncio
async def test_download_flag_sets_an_attachment_filename(app, store):
    seed_alert(store, alert_id="a1")

    response = await get_report(app, download=True)

    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert disposition.endswith('.md"')


@pytest.mark.asyncio
async def test_reports_are_served_inline_by_default(app, store):
    seed_alert(store, alert_id="a1")

    response = await get_report(app)

    assert response.headers["content-disposition"].startswith("inline;")
