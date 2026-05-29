import os
import logging
import json
import redis
import time
from celery import Celery

# Import existing analysis logic
# Ensure these are available via root context
from services.reasoning.prompts import build_reasoning_prompt
from libs.observability.failure_classifier import classify_failure
from libs.observability.workflow_history import WorkflowHistoryManager

from libs.schemas.workflow import (
    WorkflowExecutionRecord,
    WorkflowStatus,
)

from libs.observability.metrics import (
    workflow_executions_total,
    workflow_failures_total,
    workflow_duration_seconds,
)

logger = logging.getLogger(__name__)
history_manager = WorkflowHistoryManager()

# Redis configuration - use 'redis' host as defined in docker-compose
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# Initialize Celery app
celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

@celery_app.task(
    name="analyze_sequence",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def analyze_sequence(track_id: int):
    """
    Asynchronous task wrapper for VLM/LLM analysis.
    Calls existing project logic for reasoning.
    """
    logger.info(f"Analyzing track sequence: {track_id}")
    start_time = time.time()
    retry_count = getattr(analyze_sequence.request, "retries", 0)
    workflow_executions_total.inc()

    # Initialize redis client to fetch track data
    try:
        r = redis.from_url(REDIS_URL)
        
        # Search for any matching track key (track:camera_id:track_id)
        pattern = f"track:*:{track_id}"
        keys = r.keys(pattern)
        
        if not keys:
            logger.warning(f"No track record found for track_id {track_id}")
            return {"status": "not_found", "track_id": track_id}
        
        raw_data = r.get(keys[0])
        if not raw_data:
            return {"status": "error", "message": "Empty record"}
            
        track_data = json.loads(raw_data)
        logger.info(f"Retrieved track data for reasoning: {track_data}")
        
        # Call existing project logic: build prompt based on track state
        # In the full pipeline, this would involve VLM captions, 
        # but here we use the available SceneGraph / Prompt logic.
        zones = track_data.get("zones_present", [])
        summary_text = f"Track {track_id} observed in zones: {', '.join(zones)}"
        
        # Use the existing prompt builder
        prompt = build_reasoning_prompt(summary_text)
        logger.info("Successfully generated reasoning prompt from existing logic.")
        
        # No mock inference return values - just confirming the logic was called
        duration = time.time() - start_time
        
        workflow_duration_seconds.observe(duration)

        history_manager.log_execution(
            WorkflowExecutionRecord(
                workflow_name="analyze_sequence",
                task_id=str(track_id),
                status=WorkflowStatus.SUCCESS,
                retry_count=retry_count,
                duration_seconds=duration,
            )
        )

        return {
            "status": "success",
            "track_id": track_id,
            "analysis_triggered": True
        }
        
    except Exception as e:
            workflow_failures_total.inc()

            duration = time.time() - start_time

            workflow_duration_seconds.observe(duration)

            category = classify_failure(e)

            history_manager.log_execution(
                WorkflowExecutionRecord(
                    workflow_name="analyze_sequence",
                    task_id=str(track_id),
                    status=WorkflowStatus.FAILED,
                    retry_count=retry_count,
                    duration_seconds=duration,
                    failure_category=category,
                    error_message=str(e),
                )
            )

            logger.error(f"Error in analyze_sequence task: {e}")

            raise