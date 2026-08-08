"""
libs/schemas/action_recognition.py — Pydantic schemas for action recognition.

Defines:
    - ActionLabel     : enum of supported activity classes
    - ActionPrediction: per-track prediction with confidence
    - ActionAlert     : alertable action with severity and message
    - ActionFrameResult: all predictions + alerts for a single frame
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Action labels ─────────────────────────────────────────────────────────────

class ActionLabel(str, Enum):
    WALKING               = "walking"
    RUNNING               = "running"
    FIGHTING              = "fighting"
    LOITERING             = "loitering"
    FALLING               = "falling"
    SUSPICIOUS_STATIONARY = "suspicious_stationary"
    UNKNOWN               = "unknown"


# Labels that should trigger an alert
ALERTABLE_ACTIONS: set[ActionLabel] = {
    ActionLabel.FIGHTING,
    ActionLabel.RUNNING,
    ActionLabel.LOITERING,
    ActionLabel.FALLING,
    ActionLabel.SUSPICIOUS_STATIONARY,
}


class AlertSeverity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# Severity mapping per action
SEVERITY_MAP: dict[ActionLabel, AlertSeverity] = {
    ActionLabel.FIGHTING:              AlertSeverity.HIGH,
    ActionLabel.FALLING:               AlertSeverity.HIGH,
    ActionLabel.RUNNING:               AlertSeverity.MEDIUM,
    ActionLabel.LOITERING:             AlertSeverity.MEDIUM,
    ActionLabel.SUSPICIOUS_STATIONARY: AlertSeverity.LOW,
    ActionLabel.WALKING:               AlertSeverity.LOW,
    ActionLabel.UNKNOWN:               AlertSeverity.LOW,
}


# ── Per-track prediction ──────────────────────────────────────────────────────

class ActionPrediction(BaseModel):
    """Single per-track action prediction for one frame."""

    track_id:   int
    action:     ActionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    source:     str   = "heuristic"   # "heuristic" | "model"
    timestamp:  datetime = Field(default_factory=datetime.utcnow)

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ── Alert ─────────────────────────────────────────────────────────────────────

class ActionAlert(BaseModel):
    """Alert generated when a suspicious action is detected."""

    track_id:  int
    action:    ActionLabel
    severity:  AlertSeverity
    message:   str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "track_id":  self.track_id,
            "action":    self.action.value,
            "severity":  self.severity.value,
            "message":   self.message,
            "timestamp": self.timestamp.isoformat(),
        }


# ── Frame-level result ────────────────────────────────────────────────────────

class ActionFrameResult(BaseModel):
    """All action predictions and alerts for a single processed frame."""

    frame_id:    int
    predictions: list[ActionPrediction] = Field(default_factory=list)
    alerts:      list[ActionAlert]      = Field(default_factory=list)

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0
