"""
GET /reports/summary — AI-generated incident summary for a time window.

Aggregates the alerts already produced by the reasoning layer, so a report costs
no model inference and cannot fail on a provider outage: the natural-language
assessments are the ones written when each alert fired.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from apps.backend.deps import get_store
from libs.config.settings import settings
from libs.schemas.reasoning import ReasoningResult
from services.memory.ring_buffer import MemoryStore
from services.reporting import (
    ReportFormat,
    ReportRenderError,
    TimeWindow,
    build_summary,
    get_renderer,
)

logger = logging.getLogger("eagle.reports")

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get(
    "/summary",
    summary="Incident summary report for a time window",
    response_class=Response,
    responses={
        200: {
            "content": {
                "text/markdown": {},
                "application/json": {},
                "application/pdf": {},
            },
            "description": "Rendered incident summary.",
        },
        422: {"description": "Invalid or excessive time window."},
        503: {"description": "Requested format unavailable in this deployment."},
    },
)
def summary_report(
    start: Optional[datetime] = Query(
        None,
        description="ISO-8601 start of the window. Defaults to "
                    "`end` minus `report_default_window_hours`.",
    ),
    end: Optional[datetime] = Query(
        None, description="ISO-8601 end of the window. Defaults to now."
    ),
    camera_id: Optional[str] = Query(
        None, description="Restrict to one camera. Omit to cover all cameras."
    ),
    report_format: ReportFormat = Query(
        "markdown", alias="format", description="Output format."
    ),
    top_n: int = Query(
        10, ge=1, le=100, description="How many suspicious activities to spotlight."
    ),
    include_dismissed: bool = Query(
        True, description="Include alerts an operator dismissed."
    ),
    download: bool = Query(
        False, description="Serve as a file attachment rather than inline."
    ),
    store: MemoryStore = Depends(get_store),
) -> Response:
    """Summarise detected activity between `start` and `end`."""
    window = _resolve_window(start, end)

    raw_alerts = store.get_alerts_in_range(
        start_ms  = window.start_ms,
        end_ms    = window.end_ms,
        camera_id = camera_id,
        limit     = settings.report_max_alerts,
    )
    alerts = [parsed for raw in raw_alerts if (parsed := _parse_alert(raw))]

    verdicts = store.get_feedback_bulk([a.alert_id for a in alerts if a.alert_id])
    if not include_dismissed:
        alerts = [a for a in alerts if verdicts.get(a.alert_id or "") != "dismissed"]

    summary = build_summary(
        alerts,
        window,
        feedback        = verdicts,
        top_n           = top_n,
        truncated       = len(raw_alerts) >= settings.report_max_alerts,
        generated_at_ms = time.time() * 1000,
    )

    try:
        renderer = get_renderer(report_format)
        body = renderer.render(summary)
    except ReportRenderError as exc:
        logger.warning("Report rendering failed (format=%s): %s", report_format, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info(
        "Report generated  format=%s  camera=%s  alerts=%d  truncated=%s",
        report_format, camera_id or "all", summary.total_alerts, summary.truncated,
    )
    return Response(
        content=body,
        media_type=renderer.media_type,
        headers=_content_disposition(window, renderer.extension, download),
    )


def _resolve_window(start: Optional[datetime], end: Optional[datetime]) -> TimeWindow:
    """Apply defaults, then validate ordering and span.

    Naive datetimes are read as UTC, matching the epoch-millisecond timestamps
    the reasoning layer writes.
    """
    end = end or datetime.now(tz=timezone.utc)
    start = start or end - timedelta(hours=settings.report_default_window_hours)

    start, end = _as_utc(start), _as_utc(end)

    if start >= end:
        raise HTTPException(
            status_code=422, detail="`start` must be strictly earlier than `end`."
        )

    max_span = timedelta(hours=settings.report_max_window_hours)
    if end - start > max_span:
        raise HTTPException(
            status_code=422,
            detail=f"Window exceeds the {settings.report_max_window_hours:g}h maximum. "
                   "Request a narrower range.",
        )

    return TimeWindow(start_ms=start.timestamp() * 1000, end_ms=end.timestamp() * 1000)


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def _parse_alert(raw: str) -> Optional[ReasoningResult]:
    """Deserialise a stored alert, skipping any record we cannot read.

    A single corrupt entry must not deny the operator the rest of the report.
    """
    try:
        return ReasoningResult(**json.loads(raw))
    except Exception as exc:
        logger.warning("Skipping unreadable alert in report window: %s", exc)
        return None


def _content_disposition(
    window: TimeWindow, extension: str, download: bool
) -> dict[str, str]:
    stamp = datetime.fromtimestamp(
        window.start_ms / 1000, tz=timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    disposition = "attachment" if download else "inline"
    filename = f"eagle-incident-summary-{stamp}.{extension}"
    return {"Content-Disposition": f'{disposition}; filename="{filename}"'}
