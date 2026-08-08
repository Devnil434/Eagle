"""
Unit tests for ReasoningResult computed properties:
  - confidence_tier  (high / medium / low)
  - is_actionable    (Suspicious AND confidence >= 0.65)

Covers every boundary listed in the acceptance criteria for issue #117.
"""
from __future__ import annotations

import json
import pytest

from libs.schemas.reasoning import ReasoningResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _make(label: str = "Suspicious", confidence: float = 0.80) -> ReasoningResult:
    """Return a minimal valid ReasoningResult with the given label/confidence."""
    return ReasoningResult(
        track_id=1,
        label=label,
        confidence=confidence,
        reason="test",
    )


# ── confidence_tier tests (6 boundary values) ────────────────────────────────

class TestConfidenceTier:

    def test_confidence_below_050_returns_low(self):
        """0.49 → low"""
        assert _make(confidence=0.49).confidence_tier == "low"

    def test_confidence_at_050_returns_medium(self):
        """0.50 → medium (inclusive lower bound)"""
        assert _make(confidence=0.50).confidence_tier == "medium"

    def test_confidence_below_075_returns_medium(self):
        """0.74 → medium (just below high threshold)"""
        assert _make(confidence=0.74).confidence_tier == "medium"

    def test_confidence_at_075_returns_high(self):
        """0.75 → high (inclusive lower bound)"""
        assert _make(confidence=0.75).confidence_tier == "high"

    def test_confidence_at_100_returns_high(self):
        """1.0 → high (maximum)"""
        assert _make(confidence=1.0).confidence_tier == "high"

    def test_confidence_at_000_returns_low(self):
        """0.0 → low (minimum)"""
        assert _make(confidence=0.0).confidence_tier == "low"


# ── is_actionable tests ──────────────────────────────────────────────────────

class TestIsActionable:

    def test_suspicious_high_conf_is_actionable(self):
        """Suspicious + confidence 0.65 → True"""
        assert _make(label="Suspicious", confidence=0.65).is_actionable is True

    def test_suspicious_above_threshold_is_actionable(self):
        """Suspicious + confidence 0.90 → True"""
        assert _make(label="Suspicious", confidence=0.90).is_actionable is True

    def test_suspicious_below_threshold_not_actionable(self):
        """Suspicious + confidence 0.64 → False"""
        assert _make(label="Suspicious", confidence=0.64).is_actionable is False

    def test_normal_high_conf_not_actionable(self):
        """Normal + high confidence → False (wrong label)"""
        assert _make(label="Normal", confidence=0.95).is_actionable is False

    def test_normal_low_conf_not_actionable(self):
        """Normal + low confidence → False"""
        assert _make(label="Normal", confidence=0.30).is_actionable is False


# ── serialisation tests ──────────────────────────────────────────────────────

class TestSerialization:

    def test_model_dump_mode_json_includes_confidence_tier(self):
        r = _make(confidence=0.80)
        d = r.model_dump(mode="json")
        assert d["confidence_tier"] == "high"

    def test_model_dump_mode_json_includes_is_actionable(self):
        r = _make(label="Suspicious", confidence=0.80)
        d = r.model_dump(mode="json")
        assert d["is_actionable"] is True

    def test_model_dump_json_includes_both_properties(self):
        r = _make(label="Suspicious", confidence=0.51)
        parsed = json.loads(r.model_dump_json())
        assert parsed["confidence_tier"] == "medium"
        assert parsed["is_actionable"] is False

    def test_model_dump_plain_includes_computed_fields(self):
        r = _make(label="Normal", confidence=0.49)
        d = r.model_dump()
        assert d["confidence_tier"] == "low"
        assert d["is_actionable"] is False
