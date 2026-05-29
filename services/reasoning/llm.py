"""
Minimal LLM reasoner implementations for testing.
Provides a MockLLMReasoner used by unit tests and a small JSON parser.
"""
from __future__ import annotations
import json
import logging
from typing import Optional

from libs.schemas.reasoning import ReasoningResult
from libs.schemas.memory import TrackSequence

logger = logging.getLogger(__name__)


class ReasoningParseError(RuntimeError):
    pass


class BaseLLMReasoner:
    def reason(self, seq: TrackSequence, captions: list[str]) -> ReasoningResult:
        raise NotImplementedError()


class MockLLMReasoner(BaseLLMReasoner):
    """Deterministic mock reasoner for tests."""

    def reason(self, seq: TrackSequence, captions: list[str]) -> ReasoningResult:
        # Simple heuristics: if lingering or repeated_approach present -> Suspicious
        has_suspicious = any(
            e.action_hint.name in ("LINGERING", "REPEATED_APPROACH", "NEAR_KEYPAD")
            for e in seq.events
        )
        if has_suspicious or seq.total_dwell > 10:
            label = "Suspicious"
            confidence = 0.85
            reason = "Loitering near access point."
            key_signal = "lingering"
        else:
            label = "Normal"
            confidence = 0.6
            reason = "Normal movement."
            key_signal = "walking"

        return ReasoningResult(
            track_id = seq.track_id,
            camera_id = seq.camera_id,
            label = label,
            confidence = confidence,
            reason = reason,
            key_signal = key_signal,
            vlm_captions = captions,
        )

    def _parse_json(self, raw: str, track_id: int, camera_id: str) -> ReasoningResult:
        try:
            obj = json.loads(raw)
        except Exception as e:
            raise ReasoningParseError("Failed to parse LLM JSON") from e

        label = obj.get("label", "Normal")
        if label not in ("Suspicious", "Normal"):
            label = "Normal"

        try:
            confidence = float(obj.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        return ReasoningResult(
            track_id = track_id,
            camera_id = camera_id,
            label = label,
            confidence = confidence,
            reason = obj.get("reason", ""),
            key_signal = obj.get("key_signal", ""),
        )


def get_reasoner(provider: Optional[str] = None) -> BaseLLMReasoner:
    p = (provider or "mock").lower()
    if p == "mock":
        return MockLLMReasoner()
    raise ValueError("Unknown LLM reasoner provider")
