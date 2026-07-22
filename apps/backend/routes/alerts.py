"""
GET /alerts             — paginated alert list
GET /alerts/{alert_id}  — single alert
GET /alerts/stream      — SSE real-time stream
"""
from __future__ import annotations
import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from apps.backend.deps    import get_store
from apps.backend.schemas import AlertResponse
from libs.config.settings import settings
from services.memory.ring_buffer  import MemoryStore
from services.reasoning.pipeline  import alert_queue   # shared asyncio.Queue

logger = logging.getLogger("eagle.alerts")

router = APIRouter()

def _parse_alert(raw: str) -> AlertResponse | None:
    try:
        d = json.loads(raw)
        return AlertResponse(**d)
    except Exception as exc:
        logger.warning("Failed to parse alert JSON: %s", exc)
        return None

@router.get("", response_model=list[AlertResponse])
def list_alerts(
    camera_id: str   = Query("cam_01"),
    limit:     int   = Query(20, ge=1, le=settings.max_alerts_page),
    store:     MemoryStore = Depends(get_store),
) -> list[AlertResponse]:
    """Return the most recent `limit` alerts for a camera, newest first."""
    raws    = store.get_alerts(camera_id=camera_id, limit=limit)
    results = []
    for raw in raws:
        a = _parse_alert(raw)
        if a:
            # Attach feedback if exists
            fb = store.get_feedback(a.alert_id)
            if fb:
                a.feedback = fb
            results.append(a)
    return results

@router.get("/stream")
async def stream_alerts(camera_id: str = Query("cam_01")) -> EventSourceResponse:
    """
    SSE endpoint.  Dashboard subscribes to this for real-time alert push.
    Each event is a JSON-encoded AlertResponse.
    """

    async def generator():
        while True:
            try:
                result = await asyncio.wait_for(alert_queue.get(), timeout=15.0)
                if result.camera_id == camera_id:
                    yield {"data": result.model_dump_json()}
                else:
                    # Put back — another camera subscription might need it
                    # (In production use per-camera queues)
                    await alert_queue.put(result)
            except asyncio.TimeoutError:
                yield {"event": "heartbeat", "data": ""}   # keep connection alive

    return EventSourceResponse(generator())

@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: str,
    store:    MemoryStore = Depends(get_store),
) -> AlertResponse:
    """Fetch a single alert by its UUID."""
    raw = store.get_alert_by_id(alert_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id!r} not found")
    a  = _parse_alert(raw)
    if a is None:
        raise HTTPException(status_code=500, detail="Alert data corrupted")
    fb = store.get_feedback(alert_id)
    if fb:
        a.feedback = fb
    return a
