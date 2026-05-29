"""
Pydantic schemas for detection service output.
These are the contracts between the detection service and tracking.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
import time


class BoundingBox(BaseModel):
    """Single bounding box in absolute pixel coordinates."""

    x1: float = Field(..., description="Left edge")
    y1: float = Field(..., description="Top edge")
    x2: float = Field(..., description="Right edge")
    y2: float = Field(..., description="Bottom edge")

    @property
    def center(self) -> tuple[float, float]:
        """Return (cx, cy) center of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Return area of the bounding box in square pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class DetectionSchema(BaseModel):
    """Single detection within a frame."""

    label: str = Field(..., description="COCO class label, e.g. 'person'")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: Optional[int] = Field(None, description="COCO class ID")
    center: tuple[float, float] = Field(..., description="Center coordinates (cx, cy)")
    zones_present: list[str] = Field(default_factory=list, description="Zones this detection is in")

class DetectionFrameSchema(BaseModel):
    """Collection of detections for a single frame."""

    frame_id: int = Field(..., description="Frame index")
    camera_id: str = Field("cam_01", description="Camera identifier")
    detections: list[DetectionSchema] = Field(default_factory=list)
    timestamp_ms: float = Field(
        default_factory=lambda: time.time() * 1000,
        description="Frame timestamp in milliseconds",
    )
