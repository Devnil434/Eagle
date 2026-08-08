import os

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import generate_latest

from apps.backend.routes.cameras import identity_router, router as cameras_router
from services.memory.memory import MemoryService
from services.tracking.cross_camera_reid import CrossCameraReID

app = FastAPI(title="Eagle Surveillance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    redis_client.ping()
    reid_engine = CrossCameraReID(redis_client)
    memory_service = MemoryService(redis_client, reid_engine)
    app.state.redis = redis_client
    app.state.reid = reid_engine
    app.state.memory = memory_service
    print(f"[INFO] Connected to Redis at {REDIS_URL}")
except (redis.RedisError, redis.ConnectionError) as e:
    print(f"[WARN] Redis not available: {e}")
    redis_client = None
    app.state.redis = None
    app.state.reid = None
    app.state.memory = None

app.include_router(cameras_router)
app.include_router(identity_router)


@app.get("/health")
def health():
    redis_status = "healthy"
    if redis_client is not None:
        try:
            redis_client.ping()
        except Exception:
            redis_status = "unhealthy"
    else:
        redis_status = "unavailable"
    return {"status": "ok" if redis_status == "healthy" else "degraded", "redis": redis_status}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
