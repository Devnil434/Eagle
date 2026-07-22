from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
import time

class IngestRequest(BaseModel):
    """Payload for POST /ingest — a single detection frame."""
    camera_id:   str              = "cam_01"
    track_id:    int
    frame_b64:   Optional[str]    = None   # base64 JPEG, optional
    bbox:        list[float]      = Field(default_factory=list)  # [x1,y1,x2,y2]
    label:       str              = "person"
    confidence:  float            = 1.0
    zones:       list[str]        = Field(default_factory=list)
    timestamp_ms: float           = Field(default_factory=lambda: time.time() * 1000)

class IngestResponse(BaseModel):
    accepted:   bool
    track_id:   int
    queued:     bool = False    # True if reasoning was scheduled
    message:    str  = "ok"

class AlertResponse(BaseModel):
    alert_id:      str
    track_id:      int
    camera_id:     str
    label:         Literal["Suspicious", "Normal"]
    confidence:    float
    severity_score: float
    reason:        str
    key_signal:    str
    timestamp_ms:  float
    vlm_captions:  list[str] = Field(default_factory=list)
    feedback:      Optional[str] = None    # "confirmed" | "dismissed" | None

class FeedbackRequest(BaseModel):
    operator_id: str = "anonymous"
    verdict:     Literal["confirmed", "dismissed"]
    notes:       Optional[str] = None

class FeedbackResponse(BaseModel):
    alert_id:   str
    verdict:    str
    recorded:   bool
