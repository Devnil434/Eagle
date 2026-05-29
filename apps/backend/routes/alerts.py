import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from apps.backend.core.redis import redis_client
from libs.schemas.alert import Alert
from apps.backend.services.alert_broker import publish_alert
from apps.backend.core.queue import alert_queue

router = APIRouter()

CHANNEL_NAME = "eagle_alerts"


# =========================
# SSE STREAM ENDPOINT
# =========================
@router.get("/alerts/stream")
async def stream_alerts(request: Request):

    async def event_generator():

        print("✅ SSE CONNECTED")

        while True:

            if await request.is_disconnected():
                break

            try:
                alert = await asyncio.wait_for(alert_queue.get(), timeout=10)

                yield f"data: {alert}\n\n"

            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
# =========================
# TEST ALERT ENDPOINT
# =========================
@router.post("/alerts/test")
async def test_alert():

    alert = Alert(
        id=str(uuid.uuid4()),
        camera_id="cam_01",
        track_id=7,
        label="Suspicious",
        confidence=0.92,
        reason="Repeated loitering near restricted exit",
        timestamp=datetime.utcnow(),
        zone="Exit A"
    )

    await publish_alert(alert)

    return {"status": "sent"}