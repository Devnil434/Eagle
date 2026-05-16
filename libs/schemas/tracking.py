"""
Pydantic schemas for tracking output.
These are the contracts between the tracking service and everything downstream.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class TrackState(str, Enum):
    BORN     = "BORN"      # first frame this track_id appeared
    ACTIVE   = "ACTIVE"    # confirmed, currently visible
    LOST     = "LOST"      # not seen for up to max_age frames
    DEAD     = "DEAD"      # expired — will not be reassigned


class TrajectoryPoint(BaseModel):
    x: float
    y: float
    frame_id: int


class TrackedObject(BaseModel):
    track_id: int                          = Field(..., description="Persistent ID across frames")
    label: str                             = Field(..., description="COCO class label, e.g. 'person'")
    bbox: list[float]                      = Field(..., description="[x1, y1, x2, y2] absolute pixels")
    confidence: float                      = Field(..., ge=0.0, le=1.0)
    center: tuple[float, float]
    dwell_time_frames: int                 = Field(0, description="Frames this track has been active")
    dwell_time_seconds: float              = Field(0.0)
    state: TrackState                      = TrackState.ACTIVE
    trajectory: list[TrajectoryPoint]      = Field(default_factory=list)
    zones_present: list[str]               = Field(default_factory=list)
    last_seen_frame: int                   = 0


class TrackedFrame(BaseModel):
    frame_id: int
    camera_id: str                         = "cam_01"
    tracks: list[TrackedObject]
    timestamp_ms: float
    fps: float                             = 0.0


class TrackLifecycleEvent(BaseModel):
    """Emitted on BORN / LOST / DEAD transitions — consumed by memory service."""
    event: TrackState
    track_id: int
    frame_id: int
    camera_id: str                         = "cam_01"
    zones_present: list[str]               = Field(default_factory=list)
    dwell_time_seconds: float              = 0.0
    timestamp_ms: float                    = 0.0

    def to_jsonl_dict(self) -> dict:
        """
        Return a dict matching the Issue #14 JSONL schema.

        Mapping rules
        -------------
        * Internal ``BORN`` → output ``BIRTH`` (per issue spec).
        * ``zone`` is the first entry in ``zones_present``, or ``"unknown"``.
        * ``dwell_time_sec`` is included only for ``LOST`` events.
        * ISO-8601 ``timestamp`` derived from ``timestamp_ms``.
        """
        from datetime import datetime, timezone

        event_name = "BIRTH" if self.event == TrackState.BORN else self.event.value
        ts_iso = datetime.fromtimestamp(
            self.timestamp_ms / 1000.0, tz=timezone.utc
        ).isoformat()

        record: dict = {
            "event": event_name,
            "track_id": self.track_id,
            "timestamp": ts_iso,
        }

        if self.event == TrackState.BORN:
            record["zone"] = self.zones_present[0] if self.zones_present else "unknown"
        elif self.event == TrackState.LOST:
            record["dwell_time_sec"] = round(self.dwell_time_seconds, 2)

        return record