"""
ReasoningPipeline — end-to-end orchestration of Phase 4.

Flow per call:
  1. Fetch TrackSequence from ring buffer
  2. Check dedup gate
  3. Sample up to MAX_CAPTIONS trigger events
  4. For each: VLM.caption(frame, action_hint)
  5. Grounding check (reject hallucinated objects)
  6. Build LLM prompt → LLM.reason(seq, captions)
  7. Compute severity score
  8. Store alert in Redis sorted set
  9. Push to alert_queue for SSE streaming
"""
from __future__ import annotations

import json

import asyncio
import datetime
import logging
import time
import uuid
from typing import Optional

import numpy as np

from libs.schemas.memory   import ActionHint, TrackSequence
from libs.schemas.reasoning import ReasoningResult, GroundingResult
from services.memory.ring_buffer import MemoryStore
from services.reasoning.dedup         import AlertDeduplicator
from services.reasoning.vlm           import BaseCaptioner, get_captioner
from services.reasoning.llm           import BaseLLMReasoner, get_reasoner
from services.reasoning.risk_scoring  import AdaptiveRiskScorer

logger = logging.getLogger(__name__)

MAX_CAPTIONS      = int(__import__("os").getenv("MAX_CAPTIONS",      "3"))
GROUNDING_ENABLED = __import__("os").getenv("GROUNDING_ENABLED", "true").lower() == "true"

# Global asyncio queue consumed by FastAPI SSE endpoint
alert_queue: asyncio.Queue = asyncio.Queue(maxsize=200)




class ReasoningPipeline:
    """
    Orchestrates VLM + grounding + LLM for one track/frame pair.

    Usage:
        pipeline = ReasoningPipeline()
        result   = pipeline.run(track_id=5, frame=bgr_frame)
    """

    def __init__(
        self,
        captioner:    Optional[BaseCaptioner]   = None,
        reasoner:     Optional[BaseLLMReasoner] = None,
        store:        Optional[MemoryStore]     = None,
        deduplicator: Optional[AlertDeduplicator] = None,
        risk_scorer:  Optional[AdaptiveRiskScorer] = None,
    ) -> None:
        self._captioner    = captioner    or get_captioner()
        self._reasoner     = reasoner     or get_reasoner()
        self._store        = store        or MemoryStore()
        self._deduplicator = deduplicator or AlertDeduplicator(
            redis_client=self._store._r
        )
        self._risk_scorer  = risk_scorer  or AdaptiveRiskScorer()

    # ── Public ───────────────────────────────────────────────────────────────

    def run(
        self,
        track_id:   int,
        frame:      np.ndarray,
        detections: Optional[list[str]] = None,
    ) -> Optional[ReasoningResult]:
        """
        Run the full reasoning pipeline for `track_id`.

        Args:
            track_id:   ID from Phase 2 tracker.
            frame:      Current BGR frame (used for VLM captioning).
            detections: List of YOLO-detected labels in this frame
                        (used for grounding check).  Pass None to skip check.

        Returns:
            ReasoningResult if reasoning ran, None if deduplicated / no events.
        """
        seq = self._store.get_sequence(track_id)
        if not seq.events:
            logger.debug("No events for track %d — skipping reasoning", track_id)
            return None

        zone = seq.zones_visited[0] if seq.zones_visited else "unknown"

        if self._deduplicator.is_duplicate(track_id, zone):
            logger.debug("Duplicate suppressed  track=%d zone=%s", track_id, zone)
            return None

        captions = self._collect_captions(frame, seq, detections)
        result   = self._reasoner.reason(seq, captions)
        result   = self._attach_severity(result, seq)
        result.alert_id   = str(uuid.uuid4())
        result.timestamp_ms = time.time() * 1000

        self._store_alert(result)
        self._deduplicator.mark_alerted(track_id, zone)

        # Back-link the trigger event to the reasoning result (update in place)
        if seq.events:
            trigger_event = seq.events[-1]
            trigger_event.reasoning_result_id = result.alert_id
            key = self._store._events_key(trigger_event.track_id)
            payload = trigger_event.model_dump() if hasattr(trigger_event, "model_dump") else trigger_event.dict()
            self._store._r.lset(key, -1, json.dumps(payload))

        # Non-blocking push to SSE queue
        try:
            alert_queue.put_nowait(result)
        except asyncio.QueueFull:
            logger.warning("alert_queue full — SSE push dropped for alert %s",
                           result.alert_id)

        logger.info(
            "Reasoning complete  track=%d  label=%s  conf=%.2f  "
            "severity=%.2f  alert_id=%s",
            track_id, result.label, result.confidence,
            result.severity_score, result.alert_id,
        )
        return result

    # ── VLM captioning ────────────────────────────────────────────────────────

    def _collect_captions(
        self,
        frame:      np.ndarray,
        seq:        TrackSequence,
        detections: Optional[list[str]],
    ) -> list[str]:
        """
        Sample up to MAX_CAPTIONS trigger events and caption each.
        Falls back to action_summary text on VLM error.
        """
        trigger_events = [
            e for e in seq.events
            if e.action_hint in (
                ActionHint.LINGERING,
                ActionHint.NEAR_KEYPAD,
                ActionHint.REPEATED_APPROACH,
                ActionHint.ZONE_ENTRY,
            )
        ]
        if not trigger_events:
            trigger_events = seq.events[-MAX_CAPTIONS:]

        # Sample evenly
        sampled = trigger_events
        if len(trigger_events) > MAX_CAPTIONS:
            step    = len(trigger_events) // MAX_CAPTIONS
            sampled = trigger_events[::step][:MAX_CAPTIONS]

        captions: list[str] = []
        for event in sampled:
            try:
                raw_caption = self._captioner.caption(frame, event.action_hint)
                if GROUNDING_ENABLED and detections:
                    gr = self._ground(raw_caption, detections)
                    if not gr.grounded:
                        logger.warning(
                            "Hallucination detected for track %d: %s",
                            seq.track_id, gr.invented_label,
                        )
                        # Retry with strict prompt
                        raw_caption = self._captioner.caption(
                            frame, event.action_hint, detections
                        )
                captions.append(raw_caption)
            except Exception as exc:
                logger.error("VLM caption failed: %s — using hint text", exc)
                captions.append(
                    f"Person performing {event.action_hint.value.replace('_', ' ')}."
                )
        return captions

    # ── Grounding check ───────────────────────────────────────────────────────

    def _ground(
        self,
        caption:    str,
        detections: list[str],
    ) -> GroundingResult:
        """
        Lightweight heuristic: reject captions mentioning objects
        not in the YOLO detection list.

        For production, replace with a second LLM call using GROUNDING_PROMPT.
        """
        caption_lower = caption.lower()
        # Objects commonly hallucinated by LLaVA
        watch_list = ["gun", "knife", "weapon", "phone", "laptop",
                      "bag", "suitcase", "bicycle", "car"]
        detected_lower = {d.lower() for d in detections}

        for obj in watch_list:
            if obj in caption_lower and obj not in detected_lower:
                return GroundingResult(
                    grounded        = False,
                    invented_label  = obj,
                    checked_caption = caption,
                )
        return GroundingResult(grounded=True, checked_caption=caption)

    # ── Severity scoring (powered by AdaptiveRiskScorer) ─────────────────────

    def _attach_severity(
        self,
        result: ReasoningResult,
        seq:    TrackSequence,
    ) -> ReasoningResult:
        """Compute severity using the context-aware adaptive risk scorer.

        Builds contextual signals from the reasoning result and track
        sequence, delegates to :class:`AdaptiveRiskScorer`, and maps
        the 0–100 risk score back to the 0.0–1.0 ``severity_score`` field.
        """
        # Determine whether any visited zone is restricted
        restricted_keywords = ("restricted", "danger")
        in_restricted = any(
            any(kw in z.lower() for kw in restricted_keywords)
            for z in seq.zones_visited
        )

        # Count repeated approaches from structured event data
        approach_count = sum(
            1 for event in seq.events
            if event.action_hint == ActionHint.REPEATED_APPROACH
        )

        # Determine after-hours status from current wall-clock hour
        current_hour = datetime.datetime.now().hour
        is_after_hours = current_hour >= 20 or current_hour < 6

        signals = {
            "restricted_zone": in_restricted,
            "repeated_approach": approach_count,
            "loitering": seq.total_dwell,
            "after_hours": is_after_hours,
            "reasoning_confidence": result.confidence,
        }

        risk_result = self._risk_scorer.score(signals)
        result.severity_score = round(risk_result["risk_score"] / 100.0, 3)
        return result

    # ── Storage ───────────────────────────────────────────────────────────────

    def _store_alert(self, result: ReasoningResult) -> None:
        self._store.store_alert(
            alert_json   = result.model_dump_json(),
            timestamp_ms = result.timestamp_ms,
            camera_id    = result.camera_id,
        )
