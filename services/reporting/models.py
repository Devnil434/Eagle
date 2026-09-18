"""
Data contract for incident summary reports.

These models are the single interface between aggregation and rendering: the
aggregator is the only thing that builds them, and every renderer consumes them.
Adding an output format therefore requires no change here.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

UNKNOWN_GROUP = "unknown"


class TimeWindow(BaseModel):
    """Inclusive reporting window in epoch milliseconds."""

    start_ms: float
    end_ms:   float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.end_ms - self.start_ms) / 1000)


class ConfidenceStats(BaseModel):
    """Aggregate confidence distribution across the reported alerts."""

    mean:    float = 0.0
    median:  float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    # None for fewer than two samples, where standard deviation is undefined.
    stdev:   Optional[float] = None
    high_tier_count:   int = 0
    medium_tier_count: int = 0
    low_tier_count:    int = 0


class GroupStat(BaseModel):
    """Rolled-up statistics for one object type or one zone."""

    key:              str
    count:            int   = 0
    suspicious_count: int   = 0
    mean_confidence:  float = 0.0
    max_confidence:   float = 0.0
    mean_severity:    float = 0.0


class TimelineEntry(BaseModel):
    """One alert as it appears in the report timeline."""

    timestamp_ms:   float
    alert_id:       str
    track_id:       int
    camera_id:      str
    label:          Literal["Suspicious", "Normal"]
    zone:           Optional[str] = None
    object_labels:  list[str]     = Field(default_factory=list)
    confidence:     float         = 0.0
    severity_score: float         = 0.0
    reason:         str           = ""
    key_signal:     str           = ""
    feedback:       Optional[str] = None


class IncidentSummary(BaseModel):
    """Everything a rendered incident report needs, in presentation order."""

    generated_at_ms: float
    window:          TimeWindow
    cameras:         list[str] = Field(default_factory=list)

    total_alerts:      int = 0
    suspicious_alerts: int = 0
    normal_alerts:     int = 0
    actionable_alerts: int = 0

    confidence:     ConfidenceStats = Field(default_factory=ConfidenceStats)
    by_object_type: list[GroupStat]  = Field(default_factory=list)
    by_zone:        list[GroupStat]  = Field(default_factory=list)
    top_suspicious: list[TimelineEntry] = Field(default_factory=list)
    timeline:       list[TimelineEntry] = Field(default_factory=list)

    # True when the window held more alerts than the configured cap, meaning the
    # report describes a prefix of the window rather than all of it.
    truncated: bool = False
