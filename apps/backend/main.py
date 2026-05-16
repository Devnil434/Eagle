import os
import cv2
import redis
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import generate_latest

from libs.observability.metrics import frames_processed_total

app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    r = redis.from_url(REDIS_URL)
    r.ping()
    print(f"[INFO] Connected to Redis at {REDIS_URL}")
except (redis.RedisError, redis.ConnectionError) as e:
    print(f"[WARN] Redis not available: {e}")
    r = None


@app.get("/health")
def health():
    redis_status = "healthy"
    if r is not None:
        try:
            r.ping()
        except Exception:
            redis_status = "unhealthy"
    else:
        redis_status = "unavailable"
    return {"status": "ok" if redis_status == "healthy" else "degraded", "redis": redis_status}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
