"""
Event Bookmarking and Investigation Workspace API

Endpoints:
  POST   /bookmarks                    - Bookmark an event
  GET    /bookmarks                    - List bookmarks
  DELETE /bookmarks/{id}               - Remove bookmark
  POST   /investigations               - Create investigation folder
  GET    /investigations               - List investigations
  POST   /investigations/{inv_id}/events - Add event to investigation
  GET    /investigations/{inv_id}/events - List events in investigation
  POST   /bookmarks/{id}/notes         - Add notes to bookmark
  GET    /bookmarks/{id}/notes         - List notes for bookmark
  GET    /investigations/{inv_id}/timeline - Timeline view
  GET    /bookmarks/search             - Search bookmarks
  GET    /investigations/{id}/export   - Export workspace
"""
from __future__ import annotations
import logging
import uuid
import time
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bookmarks", tags=["bookmarks"])

# In-memory stores
_bookmarks: dict[str, dict] = {}
_investigations: dict[str, dict] = {}
_investigation_events: dict[str, list] = {}
_bookmark_notes: dict[str, list] = {}


# Schemas

class BookmarkRequest(BaseModel):
    alert_id: str
    camera_id: str
    track_id: int
    label: str
    notes: Optional[str] = None


class BookmarkResponse(BaseModel):
    id: str
    alert_id: str
    camera_id: str
    track_id: int
    label: str
    notes: Optional[str] = None
    created_at: float
    note_count: int = 0


class InvestigationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""


class InvestigationResponse(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str
    created_at: float
    updated_at: float
    event_count: int = 0


class AddEventRequest(BaseModel):
    alert_id: str
    camera_id: str
    track_id: int
    label: str
    notes: Optional[str] = None


class NoteRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000)


class NoteResponse(BaseModel):
    id: str
    bookmark_id: str
    content: str
    created_at: float


class TimelineEvent(BaseModel):
    id: str
    alert_id: str
    camera_id: str
    track_id: int
    label: str
    timestamp: float
    notes: Optional[str] = None


class ExportResponse(BaseModel):
    investigation_id: str
    name: str
    exported_at: float
    format: str
    data: str


# Helpers

def _log_debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)


def _get_bookmark(bookmark_id: str) -> dict:
    bm = _bookmarks.get(bookmark_id)
    if not bm:
        raise HTTPException(status_code=404, detail=f"Bookmark {bookmark_id!r} not found")
    return bm


def _get_investigation(inv_id: str) -> dict:
    inv = _investigations.get(inv_id)
    if not inv:
        raise HTTPException(status_code=404, detail=f"Investigation {inv_id!r} not found")
    return inv


# Routes

@router.post("", response_model=BookmarkResponse)
def create_bookmark(body: BookmarkRequest, operator_id: str = Query(...)):
    bm_id = str(uuid.uuid4())
    now = time.time()
    bm = {
        "id": bm_id,
        "alert_id": body.alert_id,
        "camera_id": body.camera_id,
        "track_id": body.track_id,
        "label": body.label,
        "notes": body.notes,
        "owner_id": operator_id,
        "created_at": now,
    }
    _bookmarks[bm_id] = bm
    _bookmark_notes[bm_id] = []
    _log_debug("Bookmark created  id=%s  alert=%s", bm_id, body.alert_id)
    return BookmarkResponse(**bm, note_count=0)


@router.get("", response_model=list[BookmarkResponse])
def list_bookmarks(operator_id: str = Query(...), limit: int = Query(50, ge=1, le=200)):
    results = []
    for bm_id, bm in _bookmarks.items():
        if bm.get("owner_id") == operator_id:
            results.append(BookmarkResponse(**bm, note_count=len(_bookmark_notes.get(bm_id, []))))
    return results[-limit:][::-1]


@router.delete("/{bookmark_id}")
def delete_bookmark(bookmark_id: str):
    _get_bookmark(bookmark_id)
    del _bookmarks[bookmark_id]
    _bookmark_notes.pop(bookmark_id, None)
    return {"deleted": bookmark_id}


@router.post("/investigations", response_model=InvestigationResponse, status_code=201)
def create_investigation(body: InvestigationRequest, operator_id: str = Query(...)):
    inv_id = str(uuid.uuid4())
    now = time.time()
    inv = {
        "id": inv_id,
        "name": body.name,
        "description": body.description,
        "owner_id": operator_id,
        "created_at": now,
        "updated_at": now,
    }
    _investigations[inv_id] = inv
    _investigation_events[inv_id] = []
    _log_debug("Investigation created  id=%s  name=%s", inv_id, body.name)
    return InvestigationResponse(**inv, event_count=0)


@router.get("/investigations", response_model=list[InvestigationResponse])
def list_investigations(operator_id: str = Query(...)):
    results = []
    for inv_id, inv in _investigations.items():
        if inv.get("owner_id") == operator_id:
            results.append(InvestigationResponse(**inv, event_count=len(_investigation_events.get(inv_id, []))))
    return results


@router.get("/investigations/{inv_id}", response_model=InvestigationResponse)
def get_investigation(inv_id: str):
    inv = _get_investigation(inv_id)
    return InvestigationResponse(**inv, event_count=len(_investigation_events.get(inv_id, [])))


@router.post("/investigations/{inv_id}/events", response_model=BookmarkResponse, status_code=201)
def add_event_to_investigation(inv_id: str, body: AddEventRequest, operator_id: str = Query(...)):
    _get_investigation(inv_id)
    bm_id = str(uuid.uuid4())
    now = time.time()
    bm = {
        "id": bm_id,
        "alert_id": body.alert_id,
        "camera_id": body.camera_id,
        "track_id": body.track_id,
        "label": body.label,
        "notes": body.notes,
        "owner_id": operator_id,
        "created_at": now,
    }
    _bookmarks[bm_id] = bm
    _bookmark_notes[bm_id] = []
    _investigation_events[inv_id].append(bm_id)
    _investigations[inv_id]["updated_at"] = now
    _log_debug("Event added to investigation  inv=%s  bm=%s", inv_id, bm_id)
    return BookmarkResponse(**bm, note_count=0)


@router.get("/investigations/{inv_id}/events", response_model=list[BookmarkResponse])
def list_investigation_events(inv_id: str):
    _get_investigation(inv_id)
    bm_ids = _investigation_events.get(inv_id, [])
    results = []
    for bm_id in bm_ids:
        bm = _bookmarks.get(bm_id)
        if bm:
            results.append(BookmarkResponse(**bm, note_count=len(_bookmark_notes.get(bm_id, []))))
    return results


@router.post("/bookmarks/{bookmark_id}/notes", response_model=NoteResponse, status_code=201)
def add_note_to_bookmark(bookmark_id: str, body: NoteRequest):
    _get_bookmark(bookmark_id)
    note_id = str(uuid.uuid4())
    note = {
        "id": note_id,
        "bookmark_id": bookmark_id,
        "content": body.content,
        "created_at": time.time(),
    }
    _bookmark_notes[bookmark_id].append(note)
    _log_debug("Note added to bookmark  bm=%s  note=%s", bookmark_id, note_id)
    return NoteResponse(**note)


@router.get("/bookmarks/{bookmark_id}/notes", response_model=list[NoteResponse])
def list_bookmark_notes(bookmark_id: str):
    _get_bookmark(bookmark_id)
    return _bookmark_notes.get(bookmark_id, [])


@router.get("/investigations/{inv_id}/timeline", response_model=list[TimelineEvent])
def get_investigation_timeline(inv_id: str):
    _get_investigation(inv_id)
    bm_ids = _investigation_events.get(inv_id, [])
    events = []
    for bm_id in bm_ids:
        bm = _bookmarks.get(bm_id)
        if bm:
            events.append(TimelineEvent(
                id=bm_id,
                alert_id=bm["alert_id"],
                camera_id=bm["camera_id"],
                track_id=bm["track_id"],
                label=bm["label"],
                timestamp=bm["created_at"],
                notes=bm.get("notes"),
            ))
    events.sort(key=lambda e: e.timestamp)
    return events


@router.get("/search", response_model=list[BookmarkResponse])
def search_bookmarks(q: str = Query(..., min_length=1), operator_id: str = Query(...)):
    q_lower = q.lower()
    results = []
    for bm_id, bm in _bookmarks.items():
        if bm.get("owner_id") != operator_id:
            continue
        haystack = f"{bm['label']} {bm.get('notes', '')} {bm['alert_id']} {bm['camera_id']}".lower()
        if q_lower in haystack:
            results.append(BookmarkResponse(**bm, note_count=len(_bookmark_notes.get(bm_id, []))))
    return results[::-1]


@router.get("/investigations/{inv_id}/export", response_model=ExportResponse)
def export_investigation(inv_id: str):
    inv = _get_investigation(inv_id)
    bm_ids = _investigation_events.get(inv_id, [])
    events = []
    for bm_id in bm_ids:
        bm = _bookmarks.get(bm_id)
        if bm:
            events.append({
                "id": bm_id,
                "alert_id": bm["alert_id"],
                "camera_id": bm["camera_id"],
                "track_id": bm["track_id"],
                "label": bm["label"],
                "notes": bm.get("notes"),
                "created_at": bm["created_at"],
            })
    export_data = {
        "investigation_id": inv_id,
        "name": inv["name"],
        "description": inv["description"],
        "exported_at": time.time(),
        "events": events,
    }
    return ExportResponse(
        investigation_id=inv_id,
        name=inv["name"],
        exported_at=export_data["exported_at"],
        format="json",
        data=json.dumps(export_data, indent=2),
    )
