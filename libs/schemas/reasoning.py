from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, computed_field


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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_tier(self) -> Literal["high", "medium", "low"]:
        """Classify confidence into high/medium/low tiers for dashboard colour-coding."""
        if self.confidence >= 0.75:
            return "high"
        if self.confidence >= 0.50:
            return "medium"
        return "low"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_actionable(self) -> bool:
        """True when the alert warrants operator attention."""
        return self.label == "Suspicious" and self.confidence >= 0.65


class GroundingResult(BaseModel):
    grounded:        bool
    invented_label:  Optional[str] = None
    checked_caption: str           = ""
