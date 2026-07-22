"""
POST /ingest — accept a detection event, store in ring buffer,
              conditionally schedule reasoning via BackgroundTasks.
"""
from __future__ import annotations
import base64
import logging
import time

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends

from apps.backend.deps            import get_store, get_pipeline
from apps.backend.schemas         import IngestRequest, IngestResponse
from libs.schemas.memory          import TrackEvent, ActionHint
from libs.observability.metrics   import INGEST_COUNTER, REASONING_TRIGGER_COUNTER
from services.memory.ring_buffer  import MemoryStore
from services.memory.trigger      import should_trigger_reasoning
from services.reasoning.pipeline  import ReasoningPipeline

logger = logging.getLogger("eagle.ingest")

router = APIRouter()

def _decode_frame(b64: str | None) -> np.ndarray:
    """Decode base64 JPEG to BGR numpy array.  Returns blank frame if None."""
    if not b64:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    import cv2
    jpg_bytes = base64.b64decode(b64)
    buf       = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame     = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

def _run_reasoning(
    pipeline: ReasoningPipeline,
    track_id: int,
    frame_b64: str | None,
    detections: list[str],
) -> None:
    """Background task — called off the request thread."""
    try:
        frame  = _decode_frame(frame_b64)
        result = pipeline.run(track_id=track_id, frame=frame,
                              detections=detections)
        if result:
            REASONING_TRIGGER_COUNTER.inc()
            logger.info("Reasoning fired  track=%d  label=%s  conf=%.2f",
                        track_id, result.label, result.confidence)
    except Exception as exc:
        logger.error("Background reasoning failed  track=%d: %s", track_id, exc)

@router.post("", response_model=IngestResponse)
async def ingest_event(
    body:        IngestRequest,
    bg:          BackgroundTasks,
    store:       MemoryStore        = Depends(get_store),
    pipeline:    ReasoningPipeline  = Depends(get_pipeline),
) -> IngestResponse:
    """
    Ingest a detection event from Phase 1/2.

    - Stores a TrackEvent in the Redis ring buffer.
    - If `should_trigger_reasoning()` returns True, schedules reasoning
      in the background (non-blocking).
    """
    INGEST_COUNTER.inc()

    event = TrackEvent(
        track_id           = body.track_id,
        frame_id           = 0,
        timestamp_ms       = body.timestamp_ms,
        zone               = body.zones[0] if body.zones else None,
        action_hint        = ActionHint.UNKNOWN,
        bbox               = body.bbox or [0, 0, 0, 0],
        center             = (
            (body.bbox[0] + body.bbox[2]) / 2 if len(body.bbox) == 4 else 0.0,
            (body.bbox[1] + body.bbox[3]) / 2 if len(body.bbox) == 4 else 0.0,
        ),
        dwell_time_seconds = 0.0,
        confidence         = body.confidence,
    )
    store.store_event(event)

    seq     = store.get_sequence(body.track_id)
    trigger = should_trigger_reasoning(seq)

    if trigger:
        bg.add_task(
            _run_reasoning,
            pipeline,
            body.track_id,
            body.frame_b64,
            [body.label],
        )

    return IngestResponse(
        accepted = True,
        track_id = body.track_id,
        queued   = trigger,
        message  = "reasoning scheduled" if trigger else "stored",
    )
