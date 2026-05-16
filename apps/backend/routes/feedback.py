from fastapi import APIRouter, HTTPException, status, Depends
from typing import Annotated
import logging
from libs.schemas.feedback import FeedbackRequest, FeedbackRecord
from apps.backend.services.feedback_collector import FeedbackCollector
from apps.backend.dependencies import get_redis_client
from redis import asyncio as aioredis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["feedback"])

async def get_feedback_collector(redis: Annotated[aioredis.Redis, Depends(get_redis_client)]) -> FeedbackCollector:
    """Inject FeedbackCollector with Redis client."""
    return FeedbackCollector(redis)



@router.post("/feedback", response_model=FeedbackRecord, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback: FeedbackRequest,
    collector: Annotated[FeedbackCollector, Depends(get_feedback_collector)]
):
    """Accept human corrections for alerts, validate, and store in Redis."""

    try:
        record = await collector.store_feedback(feedback)
        logger.info(f"Feedback stored: alert={feedback.alert_id}, label={feedback.human_label}")
        return record
    except ValueError as e:
        logger.warning(f"Invalid feedback: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Storage failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Storage error")



@router.get("/feedback/{alert_id}", response_model=FeedbackRecord)
async def get_feedback(
    alert_id: str,
    collector: Annotated[FeedbackCollector, Depends(get_feedback_collector)]
):
    """Retrieve feedback by alert ID."""
    try:
        record = await collector.get_feedback_by_alert_id(alert_id)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
        return record
    except HTTPException:
       raise
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval error"
        )



@router.get("/feedback/track/{track_id}")
async def get_track_feedback(
    track_id: int,
    collector: Annotated[FeedbackCollector, Depends(get_feedback_collector)]
):
    """Retrieve all feedback for a track."""
    try:
        records = await collector.get_feedback_by_track_id(track_id)
        return {"track_id": track_id, "count": len(records), "data": records}
    except HTTPException:
       raise
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval error"
        )