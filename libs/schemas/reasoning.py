from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
import json


class ReasoningResult(BaseModel):
    track_id:       int
    camera_id:      str                              = "cam_01"
    label:          Literal["Suspicious", "Normal"]
    confidence:     float                            = Field(..., ge=0.0, le=1.0)
    reason:         str                              = Field(..., max_length=300)
    key_signal:     str                              = ""
    timestamp_ms:   float                            = 0.0
    vlm_captions:   list[str]                        = Field(default_factory=list)
    severity_score: float                            = Field(0.0, ge=0.0, le=1.0)
    alert_id:       Optional[str]                    = None

    @property
    def confidence_tier(self) -> Literal["high", "medium", "low"]:
        if self.confidence >= 0.75:
            return "high"

        if self.confidence >= 0.50:
            return "medium"

        return "low"

    @property
    def is_actionable(self) -> bool:
        return self.label == "Suspicious" and self.confidence >= 0.65

    def model_dump_json(self, **kw) -> str:
        d = self.model_dump(**kw)
        d["confidence_tier"] = self.confidence_tier
        d["is_actionable"]   = self.is_actionable
        return json.dumps(d, default=str)


class GroundingResult(BaseModel):
    grounded:        bool
    invented_label:  Optional[str] = None
    checked_caption: str           = ""
