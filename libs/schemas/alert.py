from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class Alert(BaseModel):
    id: str
    camera_id: str
    track_id: int

    label: str
    confidence: float
    reason: str

    timestamp: datetime

    zone: Optional[str] = None