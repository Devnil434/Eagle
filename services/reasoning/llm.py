from libs.schemas.alert import Alert
from apps.backend.services.alert_broker import publish_alert

from datetime import datetime
import uuid


async def process_reasoning_result(result):

    if result["label"] == "Suspicious":

        alert = Alert(
            id=str(uuid.uuid4()),
            camera_id=result["camera_id"],
            track_id=result["track_id"],

            label=result["label"],
            confidence=result["confidence"],
            reason=result["reason"],

            timestamp=datetime.utcnow(),

            zone=result.get("zone")
        )

        await publish_alert(alert)