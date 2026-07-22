"""
apps/backend/routes/cameras.py — FastAPI router for camera and identity endpoints.

Endpoints
---------
GET  /cameras/{camera_id}/tracks
    Returns all active track records for a given camera, each enriched with
    its global_id for cross-camera identity lookup.

GET  /cameras/{camera_id}/tracks/{track_id}
    Returns a single track record including global_id and track_id.

GET  /identities/{global_id}
    Returns the full list of cam:track tokens associated with a global identity.

POST /cameras/{camera_id}/tracks/{track_id}/embedding
    Accepts an appearance embedding and triggers ReID matching.
    Used by the tracking service to push embeddings on BORN/LOST events.

Dependencies
------------
The router expects the following FastAPI app-state objects set at startup:

    app.state.redis     = redis.Redis(...)
    app.state.reid      = CrossCameraReID(app.state.redis)
    app.state.memory    = MemoryService(app.state.redis, app.state.reid)
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras", tags=["cameras"])


# ── Request / Response models ─────────────────────────────────────────────────

class TrackResponse(BaseModel):
    """Single track record returned by the API."""
    camera_id:           str
    track_id:            int
    global_id:           Optional[str]     = Field(None, description="Cross-camera identity")
    state:               str               = Field(..., description="ACTIVE | LOST | DEAD")
    dwell_time_seconds:  float             = 0.0
    zones_present:       list[str]         = Field(default_factory=list)
    born_frame:          Optional[int]     = None
    last_seen_frame:     Optional[int]     = None
    current_action:      Optional[str]     = Field(None, description="Temporal action label")
    action_confidence:   Optional[float]   = None
    action_source:       Optional[str]     = Field(None, description="heuristic | model")


class IdentityResponse(BaseModel):
    """Cross-camera identity mapping."""
    global_id:  str
    tokens:     list[str] = Field(
        ...,
        description='List of "cam_id:track_id" tokens that share this identity',
        examples=[["cam_01:3", "cam_02:7"]],
    )


class EmbeddingRequest(BaseModel):
    """Payload for pushing a new appearance embedding."""
    embedding:  list[float] = Field(..., description="L2-normalised appearance vector")
    event_type: str         = Field(
        "BORN",
        description="Lifecycle event that triggered this push: BORN or LOST",
    )


class EmbeddingResponse(BaseModel):
    global_id:    str
    is_new:       bool
    matched_cam:  Optional[str]  = None
    matched_track: Optional[int] = None
    similarity:   float          = 0.0


# ── Dependency helpers ────────────────────────────────────────────────────────

def _get_redis(request: Request):
    try:
        return request.app.state.redis
    except AttributeError:
        raise HTTPException(status_code=503, detail="Redis not initialised in app.state")


def _get_reid(request: Request):
    try:
        return request.app.state.reid
    except AttributeError:
        raise HTTPException(status_code=503, detail="ReID engine not initialised in app.state")


def _get_memory(request: Request):
    try:
        return request.app.state.memory
    except AttributeError:
        raise HTTPException(status_code=503, detail="MemoryService not initialised in app.state")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/{camera_id}/tracks", response_model=list[TrackResponse])
def list_tracks(camera_id: str, redis=Depends(_get_redis)) -> list[TrackResponse]:
    """
    Return all stored track records for *camera_id*.

    Each record includes the ``global_id`` that links it to tracks on other cameras.
    """
    pattern  = f"track:{camera_id}:*"
    all_keys: list[bytes] = redis.keys(pattern)

    results: list[TrackResponse] = []
    for raw_key in all_keys:
        raw = redis.get(raw_key)
        if raw is None:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Corrupt track record at key %s", raw_key)
            continue
        results.append(_record_to_response(data))

    return results


@router.get("/{camera_id}/tracks/{track_id}", response_model=TrackResponse)
def get_track(camera_id: str, track_id: int, redis=Depends(_get_redis)) -> TrackResponse:
    """
    Return a single track record.  404 if not found or expired.
    """
    key = f"track:{camera_id}:{track_id}"
    raw = redis.get(key)
    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=f"Track {camera_id}:{track_id} not found (may have expired)",
        )
    data = json.loads(raw)
    return _record_to_response(data)


@router.post(
    "/{camera_id}/tracks/{track_id}/embedding",
    response_model=EmbeddingResponse,
)
def push_embedding(
    camera_id:   str,
    track_id:    int,
    body:        EmbeddingRequest,
    reid=Depends(_get_reid),
) -> EmbeddingResponse:
    """
    Accept an appearance embedding from the tracking service and run ReID.

    - For **BORN** events: match against recently-lost tracks on other cameras.
    - For **LOST** events: store the embedding for future matching.

    Returns the assigned ``global_id`` and match metadata.
    """
    import numpy as np

    embedding = np.array(body.embedding, dtype=np.float32)

    if body.event_type.upper() == "BORN":
        result = reid.match_or_create(
            camera_id = camera_id,
            track_id  = track_id,
            embedding = embedding,
        )
        return EmbeddingResponse(
            global_id     = result.global_id,
            is_new        = result.is_new,
            matched_cam   = result.matched_cam,
            matched_track = result.matched_track,
            similarity    = result.similarity,
        )

    elif body.event_type.upper() == "LOST":
        reid.store_embedding(
            camera_id = camera_id,
            track_id  = track_id,
            embedding = embedding,
        )
        return EmbeddingResponse(global_id="", is_new=False)

    else:
        raise HTTPException(
            status_code=422,
            detail=f"event_type must be BORN or LOST, got '{body.event_type}'",
        )


# ── Identity router (mounted separately at /identities) ──────────────────────

identity_router = APIRouter(prefix="/identities", tags=["identities"])


@identity_router.get("/{global_id}", response_model=IdentityResponse)
def get_identity(global_id: str, reid=Depends(_get_reid)) -> IdentityResponse:
    """
    Return all camera:track tokens that share the given global identity.

    Example response::

        {
          "global_id": "3fa85f64-...",
          "tokens": ["cam_01:3", "cam_02:7"]
        }
    """
    tokens = reid.get_identity(global_id)
    if not tokens:
        raise HTTPException(
            status_code=404,
            detail=f"Identity '{global_id}' not found or expired",
        )
    return IdentityResponse(global_id=global_id, tokens=tokens)


# ── Private helpers ───────────────────────────────────────────────────────────

def _record_to_response(data: dict) -> TrackResponse:
    return TrackResponse(
        camera_id           = data.get("camera_id", ""),
        track_id            = int(data.get("track_id", 0)),
        global_id           = data.get("global_id"),
        state               = data.get("state", "UNKNOWN"),
        dwell_time_seconds  = float(data.get("dwell_time_seconds", 0.0)),
        zones_present       = data.get("zones_present", []),
        born_frame          = data.get("born_frame"),
        last_seen_frame     = data.get("last_seen_frame"),
        current_action      = data.get("current_action"),
        action_confidence   = data.get("action_confidence"),
        action_source       = data.get("action_source"),
    )
