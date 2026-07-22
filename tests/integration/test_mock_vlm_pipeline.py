"""
tests/integration/test_mock_vlm_pipeline.py

Integration tests for the deterministic mock VLM + LLM reasoning layer
(Phase 4).  These tests require no GPU, no Ollama installation, and no
external API keys — they run entirely offline and are fully CI-compatible.

Coverage:
  - MockVLMCaptioner.caption() for all known ActionHint values
  - MockVLMCaptioner fallback for unknown hints
  - MockLLMReasoner.reason() for suspicious and normal captions
  - MockLLMReasoner.reason() with action_hints kwarg
  - MockLLMReasoner fallback for unrecognised captions
  - ReasoningOutput.to_dict() serialisation
  - Factory helpers get_vlm_captioner() / get_llm_reasoner()
  - Unsupported provider raises ValueError
  - Full P1→P4 offline pipeline (detection → memory → caption → reasoning)
  - VLM_PROVIDER=mock setting is respected by Settings
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path when running pytest from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from libs.schemas.memory import ActionHint, TrackEvent, TrackSequence
from services.reasoning.mock_vlm import (
    MockLLMReasoner,
    MockVLMCaptioner,
    ReasoningOutput,
    get_llm_reasoner,
    get_vlm_captioner,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_track_event(
    track_id: int = 1,
    frame_id: int = 0,
    hint: ActionHint = ActionHint.WALKING,
    zone: str | None = None,
    dwell: float = 0.0,
) -> TrackEvent:
    return TrackEvent(
        track_id=track_id,
        frame_id=frame_id,
        timestamp_ms=time.time() * 1000 + frame_id * 33,
        zone=zone,
        action_hint=hint,
        bbox=[100.0, 80.0, 150.0, 200.0],
        center=(125.0, 140.0),
        dwell_time_seconds=dwell,
        confidence=0.91,
    )


# ── MockVLMCaptioner unit tests ───────────────────────────────────────────────

class TestMockVLMCaptioner:
    """Unit tests for MockVLMCaptioner."""

    def setup_method(self):
        self.captioner = MockVLMCaptioner()

    # --- Known action hints ---------------------------------------------------

    def test_zone_entry_caption(self):
        result = self.captioner.caption(frame=None, action_hint="ZONE_ENTRY")
        assert result == "Person enters restricted area."

    def test_lingering_caption(self):
        result = self.captioner.caption(frame=None, action_hint="LINGERING")
        assert result == "Person remains near access point."

    def test_near_keypad_caption(self):
        result = self.captioner.caption(frame=None, action_hint="NEAR_KEYPAD")
        assert result == "Person interacts with keypad."

    def test_repeated_approach_caption(self):
        result = self.captioner.caption(frame=None, action_hint="REPEATED_APPROACH")
        assert result == "Person repeatedly approaches restricted zone."

    def test_walking_caption(self):
        result = self.captioner.caption(frame=None, action_hint="WALKING")
        assert result == "Person is walking through the scene."

    def test_standing_caption(self):
        result = self.captioner.caption(frame=None, action_hint="STANDING")
        assert result == "Person is standing still."

    def test_zone_exit_caption(self):
        result = self.captioner.caption(frame=None, action_hint="ZONE_EXIT")
        assert result == "Person exits the monitored zone."

    def test_unknown_caption(self):
        result = self.captioner.caption(frame=None, action_hint="UNKNOWN")
        assert result == "Person visible in frame."

    # --- Case insensitivity ---------------------------------------------------

    def test_lowercase_hint_is_normalised(self):
        result = self.captioner.caption(frame=None, action_hint="zone_entry")
        assert result == "Person enters restricted area."

    def test_mixed_case_hint_is_normalised(self):
        result = self.captioner.caption(frame=None, action_hint="Near_Keypad")
        assert result == "Person interacts with keypad."

    # --- Fallback for unrecognised hints --------------------------------------

    def test_unrecognised_hint_returns_default(self):
        result = self.captioner.caption(frame=None, action_hint="FLYING")
        assert result == MockVLMCaptioner.DEFAULT_CAPTION

    def test_empty_string_hint_returns_default(self):
        result = self.captioner.caption(frame=None, action_hint="")
        assert result == MockVLMCaptioner.DEFAULT_CAPTION

    # --- Frame argument is ignored (API compatibility) -----------------------

    def test_frame_argument_is_ignored(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_with_frame = self.captioner.caption(
            frame=dummy_frame, action_hint="ZONE_ENTRY"
        )
        result_without_frame = self.captioner.caption(
            frame=None, action_hint="ZONE_ENTRY"
        )
        assert result_with_frame == result_without_frame

    # --- Determinism ----------------------------------------------------------

    def test_same_hint_always_returns_same_caption(self):
        results = {
            self.captioner.caption(frame=None, action_hint="LINGERING")
            for _ in range(10)
        }
        assert len(results) == 1, "Caption must be deterministic"

    # --- All ActionHint enum values are covered --------------------------------

    def test_all_action_hints_return_strings(self):
        for hint in ActionHint:
            result = self.captioner.caption(frame=None, action_hint=hint.value.upper())
            assert isinstance(result, str)
            assert len(result) > 0


# ── MockLLMReasoner unit tests ────────────────────────────────────────────────

class TestMockLLMReasoner:
    """Unit tests for MockLLMReasoner."""

    def setup_method(self):
        self.reasoner = MockLLMReasoner()

    # --- Suspicious outputs ---------------------------------------------------

    def test_keypad_caption_is_suspicious(self):
        result = self.reasoner.reason(["Person interacts with keypad."])
        assert result.label == "Suspicious"
        assert result.confidence == pytest.approx(0.88)
        assert "keypad" in result.reason.lower()

    def test_restricted_area_caption_is_suspicious(self):
        result = self.reasoner.reason(["Person enters restricted area."])
        assert result.label == "Suspicious"
        assert result.confidence == pytest.approx(0.85)

    def test_lingering_caption_is_suspicious(self):
        result = self.reasoner.reason(["Person remains near access point."])
        assert result.label == "Suspicious"
        assert result.confidence == pytest.approx(0.75)

    def test_repeated_approach_caption_is_suspicious(self):
        result = self.reasoner.reason(
            ["Person repeatedly approaches restricted zone."]
        )
        assert result.label == "Suspicious"
        assert result.confidence == pytest.approx(0.82)

    # --- Normal outputs -------------------------------------------------------

    def test_walking_caption_is_normal(self):
        result = self.reasoner.reason(["Person is walking through the scene."])
        assert result.label == "Normal"
        assert result.confidence >= 0.9

    def test_standing_caption_is_normal(self):
        result = self.reasoner.reason(["Person is standing still."])
        assert result.label == "Normal"

    def test_exit_caption_is_normal(self):
        result = self.reasoner.reason(["Person exits the monitored zone."])
        assert result.label == "Normal"

    # --- Sequence of captions -------------------------------------------------

    def test_sequence_with_suspicious_caption_triggers_alert(self):
        captions = [
            "Person is walking through the scene.",
            "Person enters restricted area.",
            "Person interacts with keypad.",
        ]
        result = self.reasoner.reason(captions)
        assert result.label == "Suspicious"

    def test_single_string_caption_accepted(self):
        result = self.reasoner.reason("Person interacts with keypad.")
        assert result.label == "Suspicious"

    # --- action_hints kwarg ---------------------------------------------------

    def test_action_hints_influence_reasoning(self):
        # Caption alone is neutral, but hint pushes it to suspicious.
        result = self.reasoner.reason(
            captions=["Person visible in frame."],
            action_hints=["NEAR_KEYPAD"],
        )
        assert result.label == "Suspicious"

    def test_action_hints_none_does_not_crash(self):
        result = self.reasoner.reason(
            captions=["Person is walking through the scene."],
            action_hints=None,
        )
        assert result.label == "Normal"

    # --- Fallback -------------------------------------------------------------

    def test_unrecognised_caption_returns_default(self):
        result = self.reasoner.reason(["The sky is blue."])
        assert result.label == "Normal"
        assert result.confidence == pytest.approx(0.70)

    def test_empty_caption_list_returns_default(self):
        result = self.reasoner.reason([])
        assert result.label == "Normal"

    # --- Determinism ----------------------------------------------------------

    def test_same_captions_always_return_same_output(self):
        captions = ["Person interacts with keypad."]
        results = [self.reasoner.reason(captions) for _ in range(10)]
        labels = {r.label for r in results}
        confidences = {r.confidence for r in results}
        assert len(labels) == 1
        assert len(confidences) == 1


# ── ReasoningOutput tests ─────────────────────────────────────────────────────

class TestReasoningOutput:
    """Unit tests for ReasoningOutput serialisation."""

    def test_to_dict_contains_required_keys(self):
        output = ReasoningOutput(
            label="Suspicious",
            confidence=0.88,
            reason="Repeated keypad interaction detected.",
        )
        d = output.to_dict()
        assert set(d.keys()) == {"label", "confidence", "reason"}

    def test_to_dict_values_match(self):
        output = ReasoningOutput(
            label="Suspicious",
            confidence=0.88,
            reason="Repeated keypad interaction detected.",
        )
        d = output.to_dict()
        assert d["label"] == "Suspicious"
        assert d["confidence"] == pytest.approx(0.88)
        assert d["reason"] == "Repeated keypad interaction detected."

    def test_to_dict_is_json_serialisable(self):
        import json

        output = ReasoningOutput(label="Normal", confidence=0.95, reason="All clear.")
        # Should not raise
        serialised = json.dumps(output.to_dict())
        parsed = json.loads(serialised)
        assert parsed["label"] == "Normal"


# ── Factory helper tests ──────────────────────────────────────────────────────

class TestFactoryHelpers:
    """Tests for get_vlm_captioner() and get_llm_reasoner()."""

    def test_get_vlm_captioner_mock(self):
        captioner = get_vlm_captioner("mock")
        assert isinstance(captioner, MockVLMCaptioner)

    def test_get_vlm_captioner_case_insensitive(self):
        captioner = get_vlm_captioner("MOCK")
        assert isinstance(captioner, MockVLMCaptioner)

    def test_get_vlm_captioner_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported VLM_PROVIDER"):
            get_vlm_captioner("ollama")

    def test_get_llm_reasoner_mock(self):
        reasoner = get_llm_reasoner("mock")
        assert isinstance(reasoner, MockLLMReasoner)

    def test_get_llm_reasoner_case_insensitive(self):
        reasoner = get_llm_reasoner("Mock")
        assert isinstance(reasoner, MockLLMReasoner)

    def test_get_llm_reasoner_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported VLM_PROVIDER"):
            get_llm_reasoner("openai")


# ── Settings integration ──────────────────────────────────────────────────────

class TestSettingsVlmProvider:
    """Verify that VLM_PROVIDER is read from settings correctly."""

    def test_default_vlm_provider_is_mock(self):
        from libs.config.settings import settings

        assert settings.vlm_provider == "mock"

    def test_vlm_provider_env_override(self, monkeypatch):
        monkeypatch.setenv("VLM_PROVIDER", "mock")
        # Re-instantiate to pick up the env var.
        from libs.config.settings import Settings

        s = Settings()
        assert s.vlm_provider == "mock"

    def test_factory_uses_settings_provider(self):
        from libs.config.settings import settings

        captioner = get_vlm_captioner(settings.vlm_provider)
        assert isinstance(captioner, MockVLMCaptioner)

        reasoner = get_llm_reasoner(settings.vlm_provider)
        assert isinstance(reasoner, MockLLMReasoner)


# ── Full offline P1→P4 pipeline integration test ─────────────────────────────

class TestFullOfflinePipeline:
    """
    End-to-end test: synthetic detection → memory → VLM caption → LLM reasoning.

    Uses in-memory TrackSequence objects (no Redis required) for the Phase 4
    layer.  No GPU, no Ollama, no external services required.
    """

    def test_p1_to_p4_pipeline_offline(self):
        """
        Simulate a suspicious track sequence and verify that the mock
        reasoning layer produces a Suspicious label with a reason.
        """
        captioner = MockVLMCaptioner()
        reasoner = MockLLMReasoner()

        # Simulate a track that enters a restricted zone and lingers.
        events = [
            make_track_event(1, 0, ActionHint.ZONE_ENTRY, zone="restricted_door"),
            make_track_event(1, 30, ActionHint.LINGERING, zone="restricted_door", dwell=6.0),
            make_track_event(1, 60, ActionHint.NEAR_KEYPAD, zone="restricted_door", dwell=8.0),
        ]
        seq = TrackSequence(track_id=1, events=events)
        assert len(seq.events) == 3

        # Generate captions for each event in the sequence.
        captions = [
            captioner.caption(frame=None, action_hint=e.action_hint.value.upper())
            for e in seq.events
        ]
        assert len(captions) == 3
        assert captions[0] == "Person enters restricted area."
        assert captions[1] == "Person remains near access point."
        assert captions[2] == "Person interacts with keypad."

        # Run the reasoning layer.
        result = reasoner.reason(captions)

        assert result.label == "Suspicious"
        assert result.confidence >= 0.75
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_normal_track_produces_normal_label(self):
        """A track that only walks should be classified as Normal."""
        captioner = MockVLMCaptioner()
        reasoner = MockLLMReasoner()

        events = [
            make_track_event(2, 0, ActionHint.WALKING),
            make_track_event(2, 10, ActionHint.WALKING),
            make_track_event(2, 20, ActionHint.STANDING),
        ]
        seq = TrackSequence(track_id=2, events=events)
        captions = [
            captioner.caption(frame=None, action_hint=e.action_hint.value.upper())
            for e in seq.events
        ]

        result = reasoner.reason(captions)
        assert result.label == "Normal"

    def test_reasoning_output_is_json_serialisable(self):
        """ReasoningOutput.to_dict() must be JSON-serialisable for API responses."""
        import json

        captioner = MockVLMCaptioner()
        reasoner = MockLLMReasoner()

        caption = captioner.caption(frame=None, action_hint="NEAR_KEYPAD")
        result = reasoner.reason([caption])

        payload = json.dumps(result.to_dict())
        parsed = json.loads(payload)

        assert parsed["label"] in ("Suspicious", "Normal")
        assert 0.0 <= parsed["confidence"] <= 1.0
        assert isinstance(parsed["reason"], str)

    def test_no_ollama_dependency(self):
        """
        Importing the mock module must not import or require ollama.
        This guards against accidental hard dependencies being introduced.
        """
        import importlib
        import sys

        # Remove any cached ollama module to get a clean check.
        sys.modules.pop("ollama", None)

        # Re-import the mock module — should succeed even without ollama.
        mod = importlib.import_module("services.reasoning.mock_vlm")
        assert hasattr(mod, "MockVLMCaptioner")
        assert "ollama" not in sys.modules, (
            "mock_vlm.py must not import ollama"
        )
