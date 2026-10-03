"""
apps/backend/routes/voice.py — Voice query and speech-to-text endpoints.

POST /voice/query
    Accept a natural-language query and search matching alerts.

GET /voice/history
    Return the recent voice query history.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from apps.backend.deps import get_store
from services.memory.ring_buffer import MemoryStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

# ── Persistence for voice query history ──────────────────────────────────────

DEFAULT_HISTORY_PATH = Path(__file__).resolve().parents[2] / "data" / "voice_history.yaml"


def _resolve_history_path() -> Path:
    env_path = os.environ.get("VOICE_HISTORY_PATH")
    return Path(env_path) if env_path else DEFAULT_HISTORY_PATH


HISTORY_PATH = _resolve_history_path()
_HISTORY_LOCK = None


def _get_history_lock():
    global _HISTORY_LOCK
    if _HISTORY_LOCK is None:
        import threading
        _HISTORY_LOCK = threading.Lock()
    return _HISTORY_LOCK


def _load_history() -> list[dict[str, Any]]:
    path = _resolve_history_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("queries", [])
    except Exception as exc:
        logger.warning("Failed to load voice history: %s", exc)
        return []


def _save_history(queries: list[dict[str, Any]]) -> None:
    path = _resolve_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_history_lock()
    with lock:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"queries": queries[-100:]}, f, default_flow_style=False)


# ── Models ───────────────────────────────────────────────────────────────────


class VoiceQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query text")
    language: str = Field("en-US", description="BCP-47 language tag, e.g. en-US, es-ES")
    camera_id: str = Field("cam_01", description="Camera to search alerts for")


class VoiceQueryResponse(BaseModel):
    query: str
    language: str
    total_matches: int
    alerts: list[dict[str, Any]]
    suggestions: list[str]


class VoiceHistoryResponse(BaseModel):
    queries: list[dict[str, Any]]


# ── Search logic ──────────────────────────────────────────────────────────────


def _search_alerts(store: MemoryStore, query: str, camera_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search alerts by matching query tokens against alert fields."""
    raw_alerts = store.get_alerts(camera_id=camera_id, limit=limit)
    results = []
    tokens = query.lower().split()
    for raw in raw_alerts:
        try:
            alert = json.loads(raw if isinstance(raw, str) else raw.decode())
        except Exception:
            continue
        haystack = " ".join(str(v) for v in alert.values()).lower()
        score = sum(1 for t in tokens if t in haystack)
        if score > 0:
            alert["_match_score"] = score
            results.append(alert)
    results.sort(key=lambda a: a.get("_match_score", 0), reverse=True)
    return results[:limit]


def _generate_suggestions(query: str) -> list[str]:
    """Generate simple follow-up suggestions based on the query."""
    base = query.strip().rstrip("?.")
    return [
        f"{base} in last hour",
        f"{base} confirmed",
        f"{base} dismissed",
    ]


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/query", response_model=VoiceQueryResponse)
def voice_query(
    body: VoiceQueryRequest,
    store: MemoryStore = Depends(get_store),
) -> VoiceQueryResponse:
    """Search alerts using a natural-language query."""
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty")

    results = _search_alerts(store, body.query.strip(), body.camera_id)
    suggestions = _generate_suggestions(body.query)

    entry = {
        "query": body.query,
        "language": body.language,
        "camera_id": body.camera_id,
        "total_matches": len(results),
        "timestamp": time.time(),
    }
    history = _load_history()
    history.append(entry)
    _save_history(history)

    return VoiceQueryResponse(
        query=body.query,
        language=body.language,
        total_matches=len(results),
        alerts=results,
        suggestions=suggestions,
    )


@router.get("/history", response_model=VoiceHistoryResponse)
def voice_history() -> VoiceHistoryResponse:
    """Return recent voice query history."""
    return VoiceHistoryResponse(queries=_load_history())
