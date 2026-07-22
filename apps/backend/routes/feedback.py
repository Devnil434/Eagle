"""
POST /feedback/{alert_id}
Operators confirm or dismiss an alert.  Stored in Redis as hash.
"""
from __future__ import annotations
import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from apps.backend.deps    import get_store
from apps.backend.schemas import FeedbackRequest, FeedbackResponse
from services.memory.ring_buffer import MemoryStore

logger = logging.getLogger("eagle.feedback")

router = APIRouter()

@router.post("/{alert_id}", response_model=FeedbackResponse)
def record_feedback(
    alert_id: str,
    body:     FeedbackRequest,
    store:    MemoryStore = Depends(get_store),
) -> FeedbackResponse:
    """
    Record operator feedback.  Key: feedback:{alert_id}
    The feedback is attached to alert responses in GET /alerts.
    """
    raw = store.get_alert_by_id(alert_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id!r} not found")

    store.store_feedback(
        alert_id    = alert_id,
        verdict     = body.verdict,
        operator_id = body.operator_id,
        notes       = body.notes or "",
        timestamp_ms = time.time() * 1000,
    )
    logger.info("Feedback recorded  alert_id=%s  verdict=%s  op=%s",
                alert_id, body.verdict, body.operator_id)

    return FeedbackResponse(alert_id=alert_id, verdict=body.verdict, recorded=True)