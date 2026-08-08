from __future__ import annotations

import json
import logging
from pathlib import Path
import os

from libs.schemas.workflow import WorkflowExecutionRecord

logger = logging.getLogger(__name__)


class WorkflowHistoryManager:

    DEFAULT_LOG_PATH = Path("data/logs/workflow_history.jsonl")

    def __init__(self, log_path: Path | None = None):

        self.log_path = log_path or self.DEFAULT_LOG_PATH

        os.makedirs(self.log_path.parent, exist_ok=True)

    def log_execution(self, record: WorkflowExecutionRecord) -> None:

        payload = record.dict()

        logger.info(
            "Workflow=%s | Status=%s | Retry=%d",
            record.workflow_name,
            record.status.value,
            record.retry_count,
        )

try:
    with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
            
except Exception as e:
            logger.warning(f"Failed to persist workflow history: {e}")