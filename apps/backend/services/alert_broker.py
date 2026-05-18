import json
from apps.backend.core.redis import redis_client
from apps.backend.core.queue import alert_queue

CHANNEL_NAME = "eagle_alerts"


async def publish_alert(alert):

    await redis_client.publish(
        "eagle_alerts",
        alert.model_dump_json()
    )

    # 🔥 ALSO push to SSE queue
    await alert_queue.put(alert.model_dump_json())