"""
Incident summary reporting.

Turns stored alerts from a time window into an operator-facing report. The layer
is split so each piece has one reason to change:

    models      — the data contract shared by aggregation and rendering
    aggregator  — pure statistics, no I/O
    renderers   — output formats (Markdown, JSON, PDF) behind one Protocol
    templates   — the single definition of report content

Reports reuse the reason text the reasoning layer already produced, so
generating one costs no model inference.
"""
from services.reporting.aggregator import build_summary
from services.reporting.models import (
    ConfidenceStats,
    GroupStat,
    IncidentSummary,
    TimelineEntry,
    TimeWindow,
)
from services.reporting.renderers import (
    SUPPORTED_FORMATS,
    ReportFormat,
    ReportRenderError,
    ReportRenderer,
    get_renderer,
)

__all__ = [
    "build_summary",
    "ConfidenceStats",
    "GroupStat",
    "IncidentSummary",
    "TimelineEntry",
    "TimeWindow",
    "get_renderer",
    "ReportFormat",
    "ReportRenderer",
    "ReportRenderError",
    "SUPPORTED_FORMATS",
]
