import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, AsyncIterator
from redis import asyncio as aioredis
from libs.schemas.feedback import FeedbackRecord, FeedbackRequest

logger = logging.getLogger(__name__)

class FeedbackCollector:
    def __init__(self, redis_client: aioredis.Redis):
        """Initialize collector with Redis client and key prefixes."""
        self.redis = redis_client
        self.feedback_prefix = "feedback:alert"
        self.index_prefix = "feedback:index:track"

    async def store_feedback(self, feedback: FeedbackRequest) -> FeedbackRecord:
        """Store feedback with atomic deduplication. Updates only human_label/note if alert exists."""
        key = f"{self.feedback_prefix}:{feedback.alert_id}"
        index_key = f"{self.index_prefix}:{feedback.track_id}"
        
        record = FeedbackRecord(
            alert_id=feedback.alert_id,
            track_id=feedback.track_id,
            caption_sequence=feedback.caption_sequence,
            original_label=feedback.original_label,
            human_label=feedback.human_label,
            human_note=feedback.human_note,
            frame_b64=feedback.frame_b64,
            timestamp=datetime.now(timezone.utc)
        )
        
        try:
            existing = await self.redis.get(key)
            if existing:
                existing_dict = json.loads(existing if isinstance(existing, str) else existing.decode())
                existing_dict.update({'human_label': record.human_label, 'human_note': record.human_note, 'timestamp': record.timestamp.isoformat()})
                record_json = json.dumps(existing_dict)
            else:
                record_json = record.model_dump_json()
            
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.set(key, record_json)
                pipe.sadd(index_key, feedback.alert_id)
                await pipe.execute()
            return record
        except Exception as e:
            logger.error(f"Store feedback failed {feedback.alert_id}: {e}", exc_info=True)
            raise

    async def get_feedback_by_alert_id(self, alert_id: str) -> Optional[FeedbackRecord]:
        """Retrieve feedback record by alert ID from Redis."""
        try:
            data = await self.redis.get(f"{self.feedback_prefix}:{alert_id}")
            if not data:
                return None
            return FeedbackRecord(**json.loads(data if isinstance(data, str) else data.decode()))
        except Exception as e:
            logger.error(f"Get alert {alert_id} failed: {e}", exc_info=True)
            raise

    async def get_feedback_by_track_id(self, track_id: int) -> List[FeedbackRecord]:
        """Retrieve all feedback for a track using batch pipeline fetch (no N+1 queries)."""
        try:
            alert_ids = await self.redis.smembers(f"{self.index_prefix}:{track_id}")
            if not alert_ids:
                return []
            
            async with self.redis.pipeline(transaction=False) as pipe:
                for aid in alert_ids:
                    pipe.get(f"{self.feedback_prefix}:{aid.decode() if isinstance(aid, bytes) else aid}")
                results = await pipe.execute()
            
            return [FeedbackRecord(**json.loads(r if isinstance(r, str) else r.decode())) for r in results if r]
        except Exception as e:
            logger.error(f"Get track {track_id} failed: {e}", exc_info=True)
            raise

    async def get_all_feedback(self, batch_size: int = 1000) -> AsyncIterator[FeedbackRecord]:
        """Stream all feedback records using SCAN cursor with async iteration (memory efficient)."""
        try:
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match=f"{self.feedback_prefix}:*", count=batch_size)
                if keys:
                    async with self.redis.pipeline(transaction=False) as pipe:
                        for key in keys:
                            pipe.get(key)
                        results = await pipe.execute()
                    for r in results:
                        if r:
                            yield FeedbackRecord(**json.loads(r if isinstance(r, str) else r.decode()))
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Stream feedback failed: {e}", exc_info=True)
            raise

    async def delete_feedback(self, alert_id: str) -> bool:
        """Delete feedback and index entry atomically via transaction."""
        try:
            data = await self.redis.get(f"{self.feedback_prefix}:{alert_id}")
            if not data:
                return False
            track_id = json.loads(data if isinstance(data, str) else data.decode())['track_id']
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.delete(f"{self.feedback_prefix}:{alert_id}")
                pipe.srem(f"{self.index_prefix}:{track_id}", alert_id)
                await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Delete {alert_id} failed: {e}", exc_info=True)
            raise