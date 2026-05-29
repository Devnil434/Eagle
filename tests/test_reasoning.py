"""
Full test suite for Phase 4.
All tests run without Ollama, GPU, or real API keys.
Uses MockVLMCaptioner + MockLLMReasoner + fakeredis.
"""
from __future__ import annotations
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import fakeredis
import numpy as np

from libs.schemas.memory   import TrackEvent, TrackSequence, ActionHint
from libs.schemas.reasoning import ReasoningResult
from services.memory.ring_buffer   import MemoryStore
from services.reasoning.vlm        import MockVLMCaptioner, get_captioner
from services.reasoning.llm        import MockLLMReasoner, get_reasoner, ReasoningParseError
from services.reasoning.dedup      import AlertDeduplicator
from services.reasoning.pipeline   import ReasoningPipeline
from services.reasoning.formatters import sequence_to_text, captions_to_text
from services.reasoning.prompts    import build_reasoning_prompt


from tests.fixtures.reasoning import (
    SUSPICIOUS_RESULT,
    NORMAL_RESULT,
    LOW_CONF_RESULT,
    EMPTY_CAPTIONS_RESULT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)

@pytest.fixture
def store(fake_redis):
    return MemoryStore(redis_client=fake_redis)

@pytest.fixture
def dedup(fake_redis):
    return AlertDeduplicator(redis_client=fake_redis, window_seconds=300)

@pytest.fixture
def pipeline(store, dedup):
    return ReasoningPipeline(
        captioner    = MockVLMCaptioner(),
        reasoner     = MockLLMReasoner(),
        store        = store,
        deduplicator = dedup,
    )

def make_event(
    track_id: int = 1,
    hint: ActionHint = ActionHint.WALKING,
    dwell: float = 0.0,
    zone: str | None = None,
    frame_id: int = 0,
) -> TrackEvent:
    return TrackEvent(
        track_id           = track_id,
        frame_id           = frame_id,
        timestamp_ms       = time.time() * 1000 + frame_id * 33,
        zone               = zone,
        action_hint        = hint,
        bbox               = [100.0, 80.0, 200.0, 300.0],
        center             = (150.0, 190.0),
        dwell_time_seconds = dwell,
        confidence         = 0.9,
    )

def make_suspicious_seq(track_id: int = 1) -> TrackSequence:
    events = [
        make_event(track_id, ActionHint.WALKING,     0.0,   "safe_corridor", 0),
        make_event(track_id, ActionHint.ZONE_ENTRY,  1.0,   "restricted_door", 10),
        make_event(track_id, ActionHint.LINGERING,   8.0,   "restricted_door", 20),
        make_event(track_id, ActionHint.NEAR_KEYPAD, 12.0,  "restricted_door", 30),
        make_event(track_id, ActionHint.NEAR_KEYPAD, 16.0,  "restricted_door", 40),
        make_event(track_id, ActionHint.REPEATED_APPROACH, 22.0, "restricted_door", 50),
    ]
    return TrackSequence(
        track_id      = track_id,
        camera_id     = "cam_01",
        events        = events,
        total_dwell   = 22.0,
        zones_visited = ["safe_corridor", "restricted_door"],
    )

def make_normal_seq(track_id: int = 2) -> TrackSequence:
    events = [
        make_event(track_id, ActionHint.WALKING,  0.0, None, 0),
        make_event(track_id, ActionHint.WALKING,  0.5, None, 5),
        make_event(track_id, ActionHint.STANDING, 1.0, None, 10),
    ]
    return TrackSequence(
        track_id      = track_id,
        camera_id     = "cam_01",
        events        = events,
        total_dwell   = 1.0,
        zones_visited = [],
    )


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_reasoning_result_confidence_tier_high():
    assert SUSPICIOUS_RESULT.confidence_tier == "high"


def test_reasoning_result_confidence_tier_medium():
    assert LOW_CONF_RESULT.confidence_tier == "medium"


def test_reasoning_result_is_actionable_true():
    assert SUSPICIOUS_RESULT.is_actionable is True


def test_reasoning_result_is_actionable_false_low_conf():
    assert LOW_CONF_RESULT.is_actionable is False


def test_reasoning_result_is_actionable_false_normal():
    assert NORMAL_RESULT.is_actionable is False


# ── Formatter tests ───────────────────────────────────────────────────────────

def test_sequence_to_text_non_empty():
    seq  = make_suspicious_seq()
    text = sequence_to_text(seq)
    assert "Track #1" in text
    assert "zone_entry" in text or "lingering" in text

def test_sequence_to_text_empty():
    seq  = TrackSequence(track_id=9)
    text = sequence_to_text(seq)
    assert "no events" in text

def test_captions_to_text_empty():
    assert "no visual" in captions_to_text([])

def test_captions_to_text_numbered():
    result = captions_to_text(["Person walking.", "Person at keypad."])
    assert "1." in result
    assert "2." in result

def test_build_reasoning_prompt_substitution():
    prompt = build_reasoning_prompt("summary", "captions", "cam_01", "restricted", 22.0)
    assert "cam_01" in prompt
    assert "restricted" in prompt
    assert "22" in prompt


# ── VLM tests ─────────────────────────────────────────────────────────────────

def test_mock_captioner_returns_string():
    cap   = MockVLMCaptioner()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = cap.caption(frame, ActionHint.LINGERING)
    assert isinstance(result, str)
    assert len(result) > 0

def test_mock_captioner_all_hints_covered():
    cap   = MockVLMCaptioner()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for hint in ActionHint:
        r = cap.caption(frame, hint)
        assert isinstance(r, str) and len(r) > 0

def test_get_captioner_mock():
    cap = get_captioner("mock")
    assert isinstance(cap, MockVLMCaptioner)

def test_get_captioner_invalid():
    with pytest.raises(ValueError, match="Unknown VLM_PROVIDER"):
        get_captioner("nonexistent_provider")


# ── LLM tests ─────────────────────────────────────────────────────────────────

def test_mock_reasoner_suspicious_on_lingering():
    reasoner = MockLLMReasoner()
    seq      = make_suspicious_seq()
    result   = reasoner.reason(seq, ["Person at keypad."])
    assert result.label      == "Suspicious"
    assert result.confidence >= 0.80
    assert result.track_id   == seq.track_id

def test_mock_reasoner_normal_on_walking():
    reasoner = MockLLMReasoner()
    seq      = make_normal_seq()
    result   = reasoner.reason(seq, ["Person walking."])
    assert result.label     == "Normal"
    assert result.confidence > 0.5

def test_mock_reasoner_deterministic():
    """Same input → same output every time."""
    reasoner = MockLLMReasoner()
    seq      = make_suspicious_seq()
    r1 = reasoner.reason(seq, ["Person at keypad."])
    r2 = reasoner.reason(seq, ["Person at keypad."])
    assert r1.label      == r2.label
    assert r1.confidence == r2.confidence
    assert r1.reason     == r2.reason

def test_llm_parse_json_valid():
    import json
    raw = json.dumps({
        "label": "Suspicious",
        "confidence": 0.85,
        "reason": "Person loitering.",
        "key_signal": "lingering",
    })
    result = MockLLMReasoner()._parse_json(raw, 1, "cam_01")
    assert result.label      == "Suspicious"
    assert result.confidence == 0.85

def test_llm_parse_json_invalid_raises():
    with pytest.raises(ReasoningParseError):
        MockLLMReasoner()._parse_json("not json at all %%%", 1, "cam_01")

def test_llm_parse_json_unknown_label_defaults_to_normal():
    import json
    raw = json.dumps({"label": "UNKNOWN", "confidence": 0.5,
                       "reason": "x", "key_signal": "x"})
    result = MockLLMReasoner()._parse_json(raw, 1, "cam_01")
    assert result.label == "Normal"

def test_get_reasoner_mock():
    r = get_reasoner("mock")
    assert isinstance(r, MockLLMReasoner)

def test_get_reasoner_invalid():
    with pytest.raises(ValueError):
        get_reasoner("invalid_provider")


# ── Deduplication tests ───────────────────────────────────────────────────────

def test_dedup_first_alert_not_duplicate(dedup):
    assert dedup.is_duplicate(1, "restricted_door") is False

def test_dedup_after_mark_is_duplicate(dedup):
    dedup.mark_alerted(1, "restricted_door")
    assert dedup.is_duplicate(1, "restricted_door") is True

def test_dedup_different_zone_not_blocked(dedup):
    dedup.mark_alerted(1, "restricted_door")
    assert dedup.is_duplicate(1, "safe_corridor") is False

def test_dedup_different_track_not_blocked(dedup):
    dedup.mark_alerted(1, "restricted_door")
    assert dedup.is_duplicate(2, "restricted_door") is False

def test_dedup_reset_allows_retrigger(dedup):
    dedup.mark_alerted(1, "restricted_door")
    dedup.reset(1, "restricted_door")
    assert dedup.is_duplicate(1, "restricted_door") is False


# ── Ring buffer tests ─────────────────────────────────────────────────────────

def test_store_and_retrieve_event(store):
    evt = make_event(5, ActionHint.ZONE_ENTRY, zone="restricted_door")
    store.store_event(evt)
    seq = store.get_sequence(5)
    assert len(seq.events)                  == 1
    assert seq.events[0].action_hint       == ActionHint.ZONE_ENTRY

def test_ring_buffer_caps_at_50(store):
    for i in range(60):
        store.store_event(make_event(7, frame_id=i))
    seq = store.get_sequence(7)
    assert len(seq.events) == 50

def test_empty_sequence_for_unknown_track(store):
    seq = store.get_sequence(9999)
    assert seq.events == []


# ── Full pipeline integration tests ───────────────────────────────────────────

def test_pipeline_run_returns_result(pipeline, store):
    seq = make_suspicious_seq(track_id=10)
    for e in seq.events:
        store.store_event(e)
    frame  = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.run(track_id=10, frame=frame)
    assert result is not None
    assert result.label      in ("Suspicious", "Normal")
    assert result.alert_id   is not None
    assert result.confidence >= 0.0

def test_pipeline_run_deduplicates_second_call(pipeline, store):
    seq = make_suspicious_seq(track_id=11)
    for e in seq.events:
        store.store_event(e)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r1 = pipeline.run(track_id=11, frame=frame)
    r2 = pipeline.run(track_id=11, frame=frame)   # should be deduplicated
    assert r1 is not None
    assert r2 is None

def test_pipeline_run_returns_none_for_empty_track(pipeline):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.run(track_id=9999, frame=frame)
    assert result is None

def test_pipeline_severity_score_range(pipeline, store):
    seq = make_suspicious_seq(track_id=12)
    for e in seq.events:
        store.store_event(e)
    frame  = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.run(track_id=12, frame=frame)
    assert result is not None
    assert 0.0 <= result.severity_score <= 1.0

def test_pipeline_alert_stored_in_redis(pipeline, store):
    seq = make_suspicious_seq(track_id=13)
    for e in seq.events:
        store.store_event(e)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pipeline.run(track_id=13, frame=frame)
    alerts = store.get_alerts("cam_01", limit=10)
    assert len(alerts) >= 1

def test_pipeline_vlm_captions_in_result(pipeline, store):
    seq = make_suspicious_seq(track_id=14)
    for e in seq.events:
        store.store_event(e)
    frame  = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.run(track_id=14, frame=frame)
    assert result is not None
    assert isinstance(result.vlm_captions, list)
    assert len(result.vlm_captions) > 0

def test_grounding_check_rejects_hallucination(pipeline):
    caption    = "Person is holding a gun and reaching for a knife."
    detections = ["person", "backpack"]
    gr = pipeline._ground(caption, detections)
    assert gr.grounded       is False
    assert gr.invented_label in ("gun", "knife")

def test_grounding_check_passes_clean_caption(pipeline):
    caption    = "Person is standing near the door."
    detections = ["person"]
    gr = pipeline._ground(caption, detections)
    assert gr.grounded is True
