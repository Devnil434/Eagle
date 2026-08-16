"""
Aggregation of stored alerts into an IncidentSummary.

Deliberately pure: no Redis, no HTTP, no templates, no clock.  Everything the
result depends on arrives as an argument, which keeps the statistics unit
testable and makes the summary reproducible for a given window.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Mapping, Optional, Sequence

from libs.schemas.reasoning import ReasoningResult
from services.reporting.models import (
    UNKNOWN_GROUP,
    ConfidenceStats,
    GroupStat,
    IncidentSummary,
    TimelineEntry,
    TimeWindow,
)


def build_summary(
    alerts: Sequence[ReasoningResult],
    window: TimeWindow,
    *,
    feedback: Optional[Mapping[str, str]] = None,
    top_n: int = 10,
    truncated: bool = False,
    generated_at_ms: float = 0.0,
) -> IncidentSummary:
    """Roll a window of alerts up into a renderable summary.

    Args:
        alerts:          Alerts inside `window`, in any order.
        window:          The reported time window.
        feedback:        alert_id → operator verdict, for alerts that have one.
        top_n:           How many of the most severe suspicious alerts to spotlight.
        truncated:       True when the window held more alerts than were fetched.
        generated_at_ms: Report generation time, injected so output is deterministic.
    """
    verdicts = feedback or {}
    entries = sorted(
        (_to_timeline_entry(alert, verdicts) for alert in alerts),
        key=lambda entry: entry.timestamp_ms,
    )

    suspicious = [e for e in entries if e.label == "Suspicious"]

    return IncidentSummary(
        generated_at_ms   = generated_at_ms,
        window            = window,
        cameras           = sorted({e.camera_id for e in entries}),
        total_alerts      = len(entries),
        suspicious_alerts = len(suspicious),
        normal_alerts     = len(entries) - len(suspicious),
        actionable_alerts = sum(1 for a in alerts if a.is_actionable),
        confidence        = _confidence_stats(alerts),
        by_object_type    = _group_by_object_type(entries),
        by_zone           = _group_by_zone(entries),
        top_suspicious    = sorted(
            suspicious, key=lambda e: (-e.severity_score, -e.confidence)
        )[:top_n],
        timeline          = entries,
        truncated         = truncated,
    )


def _to_timeline_entry(
    alert: ReasoningResult, verdicts: Mapping[str, str]
) -> TimelineEntry:
    alert_id = alert.alert_id or ""
    return TimelineEntry(
        timestamp_ms   = alert.timestamp_ms,
        alert_id       = alert_id,
        track_id       = alert.track_id,
        camera_id      = alert.camera_id,
        label          = alert.label,
        zone           = alert.zone,
        object_labels  = list(alert.object_labels),
        confidence     = alert.confidence,
        severity_score = alert.severity_score,
        reason         = alert.reason,
        key_signal     = alert.key_signal,
        feedback       = verdicts.get(alert_id),
    )


def _confidence_stats(alerts: Sequence[ReasoningResult]) -> ConfidenceStats:
    if not alerts:
        return ConfidenceStats()

    values = [a.confidence for a in alerts]
    tiers = [a.confidence_tier for a in alerts]

    return ConfidenceStats(
        mean    = round(statistics.fmean(values), 4),
        median  = round(statistics.median(values), 4),
        minimum = round(min(values), 4),
        maximum = round(max(values), 4),
        stdev   = round(statistics.stdev(values), 4) if len(values) > 1 else None,
        high_tier_count   = tiers.count("high"),
        medium_tier_count = tiers.count("medium"),
        low_tier_count    = tiers.count("low"),
    )


def _group_by_object_type(entries: Sequence[TimelineEntry]) -> list[GroupStat]:
    """Group by detected object class.

    An alert carrying several classes contributes to each of them, so group
    counts can legitimately sum to more than the total alert count.
    """
    buckets: dict[str, list[TimelineEntry]] = defaultdict(list)
    for entry in entries:
        for label in entry.object_labels or [UNKNOWN_GROUP]:
            buckets[label].append(entry)
    return _to_sorted_stats(buckets)


def _group_by_zone(entries: Sequence[TimelineEntry]) -> list[GroupStat]:
    buckets: dict[str, list[TimelineEntry]] = defaultdict(list)
    for entry in entries:
        buckets[entry.zone or UNKNOWN_GROUP].append(entry)
    return _to_sorted_stats(buckets)


def _to_sorted_stats(buckets: Mapping[str, list[TimelineEntry]]) -> list[GroupStat]:
    stats = [
        GroupStat(
            key              = key,
            count            = len(group),
            suspicious_count = sum(1 for e in group if e.label == "Suspicious"),
            mean_confidence  = round(statistics.fmean(e.confidence for e in group), 4),
            max_confidence   = round(max(e.confidence for e in group), 4),
            mean_severity    = round(statistics.fmean(e.severity_score for e in group), 4),
        )
        for key, group in buckets.items()
    ]
    # Busiest group first; key breaks ties so output is stable across runs.
    return sorted(stats, key=lambda s: (-s.count, s.key))
