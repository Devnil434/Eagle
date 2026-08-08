from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class WorkflowStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYING = "RETRYING"


class FailureCategory(str, Enum):
    TIMEOUT = "timeout"
    NETWORK = "network"
    API_FAILURE = "api_failure"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


class WorkflowExecutionRecord(BaseModel):
    workflow_name: str
    task_id: Optional[str] = None
    status: WorkflowStatus
    failure_category: Optional[FailureCategory] = None
    retry_count: int = 0
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None