"""
Unit tests for incident summary aggregation and rendering.

The aggregator is pure, so these tests need no Redis and no HTTP client.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from libs.schemas.reasoning import ReasoningResult
from services.reporting import build_summary, get_renderer
from services.reporting.models import TimeWindow
from services.reporting.renderers import ReportRenderError

BASE_MS = 1_718_000_000_000.0


def make_alert(
    *,
    alert_id: str = "alert-1",
    track_id: int = 1,
    camera_id: str = "cam_01",
    label: str = "Suspicious",
    confidence: float = 0.8,
    severity: float = 0.5,
    offset_ms: float = 0.0,
    zone: str | None = "restricted_door",
    object_labels: list[str] | None = None,
    reason: str = "Lingering near the keypad for an extended period.",
) -> ReasoningResult:
    return ReasoningResult(
        track_id       = track_id,
        camera_id      = camera_id,
        label          = label,
        confidence     = confidence,
        reason         = reason,
        key_signal     = "lingering",
        timestamp_ms   = BASE_MS + offset_ms,
        severity_score = severity,
        alert_id       = alert_id,
        zone           = zone,
        object_labels  = ["person"] if object_labels is None else object_labels,
    )


@pytest.fixture
def window() -> TimeWindow:
    return TimeWindow(start_ms=BASE_MS, end_ms=BASE_MS + 3_600_000)


# ── Totals and window ─────────────────────────────────────────────────────────

def test_empty_window_produces_zeroed_summary(window):
    summary = build_summary([], window)

    assert summary.total_alerts == 0
    assert summary.suspicious_alerts == 0
    assert summary.normal_alerts == 0
    assert summary.cameras == []
    assert summary.by_object_type == []
    assert summary.timeline == []
    assert summary.confidence.stdev is None


def test_counts_split_by_verdict(window):
    alerts = [
        make_alert(alert_id="a1", label="Suspicious"),
        make_alert(alert_id="a2", label="Normal"),
        make_alert(alert_id="a3", label="Normal"),
    ]
    summary = build_summary(alerts, window)

    assert summary.total_alerts == 3
    assert summary.suspicious_alerts == 1
    assert summary.normal_alerts == 2


def test_actionable_count_follows_schema_rule(window):
    """Actionable means Suspicious with confidence >= 0.65 per ReasoningResult."""
    alerts = [
        make_alert(alert_id="a1", label="Suspicious", confidence=0.90),
        make_alert(alert_id="a2", label="Suspicious", confidence=0.40),
        make_alert(alert_id="a3", label="Normal", confidence=0.99),
    ]
    summary = build_summary(alerts, window)

    assert summary.actionable_alerts == 1


def test_timeline_sorted_chronologically_regardless_of_input_order(window):
    alerts = [
        make_alert(alert_id="late", offset_ms=5_000),
        make_alert(alert_id="early", offset_ms=1_000),
        make_alert(alert_id="middle", offset_ms=3_000),
    ]
    summary = build_summary(alerts, window)

    assert [e.alert_id for e in summary.timeline] == ["early", "middle", "late"]


def test_cameras_are_deduplicated_and_sorted(window):
    alerts = [
        make_alert(alert_id="a1", camera_id="cam_02"),
        make_alert(alert_id="a2", camera_id="cam_01"),
        make_alert(alert_id="a3", camera_id="cam_02"),
    ]
    summary = build_summary(alerts, window)

    assert summary.cameras == ["cam_01", "cam_02"]


# ── Confidence statistics ─────────────────────────────────────────────────────

def test_confidence_stats_across_multiple_alerts(window):
    alerts = [
        make_alert(alert_id="a1", confidence=0.90),
        make_alert(alert_id="a2", confidence=0.60),
        make_alert(alert_id="a3", confidence=0.30),
    ]
    stats = build_summary(alerts, window).confidence

    assert stats.mean == pytest.approx(0.60, abs=1e-4)
    assert stats.median == pytest.approx(0.60, abs=1e-4)
    assert stats.minimum == pytest.approx(0.30, abs=1e-4)
    assert stats.maximum == pytest.approx(0.90, abs=1e-4)
    assert stats.stdev == pytest.approx(0.30, abs=1e-4)


def test_stdev_undefined_for_single_sample(window):
    """One sample has no standard deviation; it must not raise."""
    stats = build_summary([make_alert(confidence=0.72)], window).confidence

    assert stats.stdev is None
    assert stats.mean == pytest.approx(0.72)


def test_confidence_tier_counts_match_schema_thresholds(window):
    alerts = [
        make_alert(alert_id="a1", confidence=0.80),   # high  (>= 0.75)
        make_alert(alert_id="a2", confidence=0.75),   # high  (boundary)
        make_alert(alert_id="a3", confidence=0.50),   # medium (boundary)
        make_alert(alert_id="a4", confidence=0.10),   # low
    ]
    stats = build_summary(alerts, window).confidence

    assert (stats.high_tier_count, stats.medium_tier_count, stats.low_tier_count) == (2, 1, 1)


# ── Grouping ──────────────────────────────────────────────────────────────────

def test_object_grouping_counts_each_class(window):
    alerts = [
        make_alert(alert_id="a1", object_labels=["person", "backpack"]),
        make_alert(alert_id="a2", object_labels=["person"]),
    ]
    groups = {g.key: g for g in build_summary(alerts, window).by_object_type}

    assert groups["person"].count == 2
    assert groups["backpack"].count == 1


def test_alerts_without_object_labels_bucket_as_unknown(window):
    """Alerts written before provenance capture must still be represented."""
    summary = build_summary([make_alert(object_labels=[])], window)

    assert [g.key for g in summary.by_object_type] == ["unknown"]
    assert summary.by_object_type[0].count == 1


def test_alerts_without_zone_bucket_as_unknown(window):
    summary = build_summary([make_alert(zone=None)], window)

    assert [g.key for g in summary.by_zone] == ["unknown"]


def test_groups_ordered_by_count_then_key(window):
    alerts = [
        make_alert(alert_id="a1", zone="zone_b"),
        make_alert(alert_id="a2", zone="zone_b"),
        make_alert(alert_id="a3", zone="zone_a"),
        make_alert(alert_id="a4", zone="zone_c"),
    ]
    keys = [g.key for g in build_summary(alerts, window).by_zone]

    assert keys == ["zone_b", "zone_a", "zone_c"]


def test_group_stats_computed_per_bucket(window):
    alerts = [
        make_alert(alert_id="a1", zone="dock", label="Suspicious",
                   confidence=0.90, severity=0.80),
        make_alert(alert_id="a2", zone="dock", label="Normal",
                   confidence=0.50, severity=0.20),
    ]
    dock = next(g for g in build_summary(alerts, window).by_zone if g.key == "dock")

    assert dock.count == 2
    assert dock.suspicious_count == 1
    assert dock.mean_confidence == pytest.approx(0.70)
    assert dock.max_confidence == pytest.approx(0.90)
    assert dock.mean_severity == pytest.approx(0.50)


# ── Suspicious ranking ────────────────────────────────────────────────────────

def test_top_suspicious_ranked_by_severity_and_excludes_normal(window):
    alerts = [
        make_alert(alert_id="mild", label="Suspicious", severity=0.30),
        make_alert(alert_id="worst", label="Suspicious", severity=0.95),
        make_alert(alert_id="mid", label="Suspicious", severity=0.60),
        make_alert(alert_id="benign", label="Normal", severity=0.99),
    ]
    top = build_summary(alerts, window).top_suspicious

    assert [e.alert_id for e in top] == ["worst", "mid", "mild"]


def test_top_n_limits_spotlight_without_truncating_timeline(window):
    alerts = [
        make_alert(alert_id=f"a{i}", severity=i / 10, offset_ms=i * 1000)
        for i in range(5)
    ]
    summary = build_summary(alerts, window, top_n=2)

    assert len(summary.top_suspicious) == 2
    assert len(summary.timeline) == 5


# ── Feedback ──────────────────────────────────────────────────────────────────

def test_operator_feedback_attached_to_timeline(window):
    alerts = [make_alert(alert_id="a1"), make_alert(alert_id="a2")]
    summary = build_summary(alerts, window, feedback={"a1": "confirmed"})

    verdicts = {e.alert_id: e.feedback for e in summary.timeline}
    assert verdicts == {"a1": "confirmed", "a2": None}


def test_truncated_flag_is_passed_through(window):
    assert build_summary([], window, truncated=True).truncated is True


# ── Rendering ─────────────────────────────────────────────────────────────────

def test_markdown_report_contains_every_required_section(window):
    alerts = [make_alert(alert_id="a1", object_labels=["person", "backpack"])]
    summary = build_summary(alerts, window, generated_at_ms=BASE_MS)

    report = get_renderer("markdown").render(summary).decode()

    for heading in (
        "# Incident Summary Report",
        "## Overview",
        "## Confidence",
        "## Object Summary",
        "## Zone Summary",
        "## Suspicious Activities",
        "## Event Timeline",
    ):
        assert heading in report
    assert "restricted_door" in report
    assert "backpack" in report


def test_markdown_report_states_when_window_is_empty(window):
    report = get_renderer("markdown").render(build_summary([], window)).decode()

    assert "No alerts were recorded in this window." in report


def test_markdown_escapes_pipes_so_tables_survive(window):
    alerts = [make_alert(reason="Left bag | then walked away")]
    report = get_renderer("markdown").render(build_summary(alerts, window)).decode()

    assert r"Left bag \| then walked away" in report


def test_markdown_flags_truncation_to_the_operator(window):
    summary = build_summary([make_alert()], window, truncated=True)
    report = get_renderer("markdown").render(summary).decode()

    assert "alert cap" in report


def test_json_report_is_the_full_summary(window):
    summary = build_summary([make_alert(alert_id="a1")], window)

    payload = json.loads(get_renderer("json").render(summary))

    assert payload["total_alerts"] == 1
    assert payload["timeline"][0]["alert_id"] == "a1"
    assert payload["window"]["start_ms"] == BASE_MS


def test_pdf_report_is_a_valid_pdf_document(window):
    pytest.importorskip("fpdf")
    summary = build_summary([make_alert()], window)

    body = get_renderer("pdf").render(summary)

    assert body.startswith(b"%PDF-")


def test_pdf_survives_characters_outside_latin1(window):
    """LLM and VLM text carries typographic characters the core fonts lack."""
    pytest.importorskip("fpdf")
    alerts = [make_alert(reason="Approach → keypad ×3, ‘testing’ the panel…")]
    summary = build_summary(alerts, window)

    assert get_renderer("pdf").render(summary).startswith(b"%PDF-")


def test_unknown_format_is_rejected():
    with pytest.raises(ReportRenderError):
        get_renderer("docx")
