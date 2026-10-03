"""
Multi-Operator Collaboration Workspace API

Endpoints:
  GET    /workspaces                    - List workspaces for operator
  POST   /workspaces                    - Create a new workspace
  GET    /workspaces/{ws_id}            - Get workspace details
  POST   /workspaces/{ws_id}/join       - Join a workspace
  POST   /workspaces/{ws_id}/leave      - Leave a workspace
  POST   /workspaces/{ws_id}/events     - Add an annotated event
  GET    /workspaces/{ws_id}/events     - List annotated events
  POST   /workspaces/{ws_id}/events/{evt_id}/comments - Add comment
  GET    /workspaces/{ws_id}/events/{evt_id}/comments - List comments
  POST   /workspaces/{ws_id}/assignments - Assign incident to operator
  GET    /workspaces/{ws_id}/assignments - List assignments
  GET    /workspaces/{ws_id}/activity   - Activity history
"""
from __future__ import annotations
import logging
import uuid
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces", tags=["collaboration"])

# In-memory store (replace with Redis/DB in production)
_workspaces: dict[str, dict] = {}
_workspace_events: dict[str, list] = {}
_workspace_comments: dict[str, dict[str, list]] = {}
_workspace_assignments: dict[str, list] = {}
_workspace_activity: dict[str, list] = {}


class CreateWorkspaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str
    members: list[str]
    created_at: float
    updated_at: float


class JoinLeaveResponse(BaseModel):
    workspace_id: str
    operator_id: str
    action: str
    success: bool


class AnnotatedEvent(BaseModel):
    alert_id: str
    camera_id: str
    track_id: int
    label: str
    notes: Optional[str] = None
    timestamp: float = Field(default_factory=lambda: time.time())


class AnnotatedEventResponse(BaseModel):
    id: str
    alert_id: str
    camera_id: str
    track_id: int
    label: str
    notes: Optional[str] = None
    added_by: str
    timestamp: float
    comment_count: int = 0


class CommentRequest(BaseModel):
    operator_id: str
    content: str = Field(..., min_length=1, max_length=1000)


class CommentResponse(BaseModel):
    id: str
    event_id: str
    operator_id: str
    content: str
    timestamp: float


class AssignmentRequest(BaseModel):
    operator_id: str
    incident_id: str = Field(..., description="Alert or event ID")
    description: Optional[str] = None


class AssignmentResponse(BaseModel):
    id: str
    workspace_id: str
    incident_id: str
    assigned_to: str
    assigned_by: str
    description: Optional[str]
    status: str = "open"
    created_at: float


class ActivityResponse(BaseModel):
    id: str
    operator_id: str
    action: str
    details: str
    timestamp: float


# Helpers

def _log_activity(ws_id: str, operator_id: str, action: str, details: str):
    entry = {
        "id": str(uuid.uuid4()),
        "operator_id": operator_id,
        "action": action,
        "details": details,
        "timestamp": time.time(),
    }
    if ws_id not in _workspace_activity:
        _workspace_activity[ws_id] = []
    _workspace_activity[ws_id].insert(0, entry)
    _workspace_activity[ws_id] = _workspace_activity[ws_id][:100]


def _get_ws(ws_id: str) -> dict:
    ws = _workspaces.get(ws_id)
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace {ws_id!r} not found")
    return ws


def _require_member(ws: dict, operator_id: str):
    if operator_id not in ws["members"]:
        raise HTTPException(status_code=403, detail="Operator is not a member of this workspace")


# Routes

@router.get("", response_model=list[WorkspaceResponse])
def list_workspaces(operator_id: str = Query(...)):
    results = []
    for ws_id, ws in _workspaces.items():
        if operator_id in ws["members"]:
            results.append(WorkspaceResponse(id=ws_id, **ws))
    return results


@router.post("", response_model=WorkspaceResponse, status_code=201)
def create_workspace(body: CreateWorkspaceRequest, operator_id: str = Query(...)):
    ws_id = str(uuid.uuid4())
    now = time.time()
    ws = {
        "name": body.name,
        "description": body.description,
        "owner_id": operator_id,
        "members": [operator_id],
        "created_at": now,
        "updated_at": now,
    }
    _workspaces[ws_id] = ws
    _workspace_events[ws_id] = []
    _workspace_comments[ws_id] = {}
    _workspace_assignments[ws_id] = []
    _workspace_activity[ws_id] = []
    _log_activity(ws_id, operator_id, "created", f"Created workspace '{body.name}'")
    logger.info("Workspace created  id=%s  name=%s  owner=%s", ws_id, body.name, operator_id)
    return WorkspaceResponse(id=ws_id, **ws)


@router.get("/{ws_id}", response_model=WorkspaceResponse)
def get_workspace(ws_id: str):
    ws = _get_ws(ws_id)
    return WorkspaceResponse(id=ws_id, **ws)


@router.post("/{ws_id}/join", response_model=JoinLeaveResponse)
def join_workspace(ws_id: str, operator_id: str = Query(...)):
    ws = _get_ws(ws_id)
    if operator_id in ws["members"]:
        return JoinLeaveResponse(workspace_id=ws_id, operator_id=operator_id, action="join", success=True)
    ws["members"].append(operator_id)
    ws["updated_at"] = time.time()
    _log_activity(ws_id, operator_id, "joined", "Joined the workspace")
    return JoinLeaveResponse(workspace_id=ws_id, operator_id=operator_id, action="join", success=True)


@router.post("/{ws_id}/leave", response_model=JoinLeaveResponse)
def leave_workspace(ws_id: str, operator_id: str = Query(...)):
    ws = _get_ws(ws_id)
    if operator_id not in ws["members"]:
        raise HTTPException(status_code=400, detail="Operator is not a member")
    ws["members"].remove(operator_id)
    ws["updated_at"] = time.time()
    _log_activity(ws_id, operator_id, "left", "Left the workspace")
    return JoinLeaveResponse(workspace_id=ws_id, operator_id=operator_id, action="leave", success=True)


@router.post("/{ws_id}/events", response_model=AnnotatedEventResponse, status_code=201)
def add_event(ws_id: str, body: AnnotatedEvent, operator_id: str = Query(...)):
    ws = _get_ws(ws_id)
    _require_member(ws, operator_id)
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "alert_id": body.alert_id,
        "camera_id": body.camera_id,
        "track_id": body.track_id,
        "label": body.label,
        "notes": body.notes,
        "added_by": operator_id,
        "timestamp": body.timestamp,
    }
    if ws_id not in _workspace_events:
        _workspace_events[ws_id] = []
    _workspace_events[ws_id].append(evt)
    _workspace_comments[ws_id][evt_id] = []
    _log_activity(ws_id, operator_id, "added_event", f"Added event {evt_id} ({body.label})")
    return AnnotatedEventResponse(**evt, comment_count=0)


@router.get("/{ws_id}/events", response_model=list[AnnotatedEventResponse])
def list_events(ws_id: str, limit: int = Query(50, ge=1, le=200)):
    _get_ws(ws_id)
    evts = _workspace_events.get(ws_id, [])[-limit:]
    results = []
    for e in evts:
        cc = len(_workspace_comments.get(ws_id, {}).get(e["id"], []))
        results.append(AnnotatedEventResponse(**e, comment_count=cc))
    return results[::-1]


@router.post("/{ws_id}/events/{evt_id}/comments", response_model=CommentResponse, status_code=201)
def add_comment(ws_id: str, evt_id: str, body: CommentRequest):
    _get_ws(ws_id)
    if ws_id not in _workspace_events or evt_id not in [e["id"] for e in _workspace_events.get(ws_id, [])]:
        raise HTTPException(status_code=404, detail=f"Event {evt_id!r} not found")
    cid = str(uuid.uuid4())
    comment = {
        "id": cid,
        "event_id": evt_id,
        "operator_id": body.operator_id,
        "content": body.content,
        "timestamp": time.time(),
    }
    if ws_id not in _workspace_comments:
        _workspace_comments[ws_id] = {}
    if evt_id not in _workspace_comments[ws_id]:
        _workspace_comments[ws_id][evt_id] = []
    _workspace_comments[ws_id][evt_id].append(comment)
    _log_activity(ws_id, body.operator_id, "commented", f"Commented on event {evt_id}")
    return CommentResponse(**comment)


@router.get("/{ws_id}/events/{evt_id}/comments", response_model=list[CommentResponse])
def list_comments(ws_id: str, evt_id: str):
    _get_ws(ws_id)
    return _workspace_comments.get(ws_id, {}).get(evt_id, [])


@router.post("/{ws_id}/assignments", response_model=AssignmentResponse, status_code=201)
def create_assignment(ws_id: str, body: AssignmentRequest, assigned_by: str = Query(...)):
    ws = _get_ws(ws_id)
    _require_member(ws, body.operator_id)
    aid = str(uuid.uuid4())
    assignment = {
        "id": aid,
        "workspace_id": ws_id,
        "incident_id": body.incident_id,
        "assigned_to": body.operator_id,
        "assigned_by": assigned_by,
        "description": body.description,
        "status": "open",
        "created_at": time.time(),
    }
    if ws_id not in _workspace_assignments:
        _workspace_assignments[ws_id] = []
    _workspace_assignments[ws_id].append(assignment)
    _log_activity(ws_id, assigned_by, "assigned", f"Assigned {body.incident_id} to {body.operator_id}")
    return AssignmentResponse(**assignment)


@router.get("/{ws_id}/assignments", response_model=list[AssignmentResponse])
def list_assignments(ws_id: str, status: Optional[str] = None):
    _get_ws(ws_id)
    results = _workspace_assignments.get(ws_id, [])
    if status:
        results = [a for a in results if a["status"] == status]
    return [AssignmentResponse(**a) for a in results]


@router.get("/{ws_id}/activity", response_model=list[ActivityResponse])
def get_activity(ws_id: str, limit: int = Query(50, ge=1, le=200)):
    _get_ws(ws_id)
    return _workspace_activity.get(ws_id, [])[:limit]
