"""
Event schemas for the temporal memory layer.

Every time a tracked person enters a zone, dwells, or interacts with an object,
one TrackEvent is written to Redis. The LLM reasoning layer reads a sequence
of these to infer intent.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ActionHint(str, Enum):
    """
    Coarse action categories inferred from tracker state.
    These are NOT ML predictions — they're deterministic rules:
      - dwell > 5s in a zone      → LINGERING
      - centroid barely moved     → STANDING
      - fast centroid movement    → WALKING
      - zone entered for 1st time → ZONE_ENTRY
      - near keypad object bbox   → NEAR_KEYPAD
    """
    WALKING    = "walking"
    STANDING   = "standing"
    LINGERING  = "lingering"       # dwell > 5s
    ZONE_ENTRY = "zone_entry"      # first frame inside zone
    ZONE_EXIT  = "zone_exit"
    NEAR_KEYPAD = "near_keypad"
    REPEATED_APPROACH = "repeated_approach"   # entered zone > 2×
    UNKNOWN    = "unknown"


class TrackEvent(BaseModel):
    """A single behavioral event for one track at one point in time."""
    track_id:    int
    camera_id:   str              = "cam_01"
    frame_id:    int
    timestamp_ms: float
    zone:        Optional[str]    = None   # zone name or None if not in any zone
    action_hint: ActionHint       = ActionHint.UNKNOWN
    bbox:        list[float]      = Field(default_factory=list)    # [x1,y1,x2,y2]
    center:      tuple[float, float] = (0.0, 0.0)
    dwell_time_seconds: float     = 0.0
    confidence:  float            = 0.0


class TrackSequence(BaseModel):
    """A temporal sequence of events for one track — input to the VLM/LLM."""
    track_id:     int
    camera_id:    str              = "cam_01"
    events:       list[TrackEvent] = Field(default_factory=list)
    total_dwell:  float            = 0.0   # seconds
    zones_visited: list[str]       = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if len(self.events) < 2:
            return 0.0
        return (self.events[-1].timestamp_ms - self.events[0].timestamp_ms) / 1000

    @property
    def action_summary(self) -> str:
        """Human-readable summary of the action sequence (for LLM prompt)."""
        actions = [e.action_hint.value for e in self.events]
        unique  = []
        for a in actions:
            if not unique or unique[-1] != a:
                unique.append(a)
        return " → ".join(unique)