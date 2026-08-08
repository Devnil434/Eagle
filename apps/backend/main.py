from __future__ import annotations
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from libs.config.settings import settings
from apps.backend.routes import alerts, ingest, snapshot, cameras, feedback

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("eagle.backend")

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "Eagle — Agentic Vision API",
        description = "Surveillance reasoning pipeline REST + SSE interface",
        version     = "0.5.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.cors_origins,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    app.include_router(ingest.router,   prefix="/ingest",   tags=["ingest"])
    app.include_router(alerts.router,   prefix="/alerts",   tags=["alerts"])
    app.include_router(snapshot.router, prefix="/snapshot", tags=["snapshot"])
    app.include_router(cameras.router,  prefix="/cameras",  tags=["cameras"])
    app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])

    # Prometheus metrics scrape endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok", "version": "0.5.0"}

    return app

app = create_app()