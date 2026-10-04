"""
Unit tests for the Adaptive Risk Scoring Engine.

All tests are deterministic — no network, no Redis, no randomness.
Tests use a temporary YAML policy file to isolate from the real config.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.reasoning.risk_scoring import AdaptiveRiskScorer


# ── Helpers ───────────────────────────────────────────────────────────────────

_VALID_POLICY = """\
risk_scoring:
  restricted_zone:
    weight: 0.30
    description: "Restricted zone"
  repeated_approach:
    weight: 0.25
    description: "Repeated approach"
  loitering:
    weight: 0.15
    description: "Loitering"
  after_hours:
    weight: 0.20
    description: "After hours"
  reasoning_confidence:
    weight: 0.10
    description: "Confidence"

risk_levels:
  low_max: 40
  medium_max: 70

normalization:
  loitering_max_seconds: 120.0
  repeated_approach_max_count: 5
"""


@pytest.fixture
def policy_file(tmp_path):
    """Write a valid policy YAML to a temp directory and return its path."""
    path = tmp_path / "risk_policy.yaml"
    path.write_text(_VALID_POLICY, encoding="utf-8")
    return path


@pytest.fixture
def scorer(policy_file):
    """Return an AdaptiveRiskScorer loaded from the temp policy file."""
    return AdaptiveRiskScorer(policy_path=policy_file)


# ── 1. YAML loading ──────────────────────────────────────────────────────────

def test_yaml_loading(policy_file):
    """Scorer loads a valid YAML policy without errors."""
    scorer = AdaptiveRiskScorer(policy_path=policy_file)
    # Weights should be accessible and sum to 1.0
    total_weight = sum(scorer._weights.values())
    assert abs(total_weight - 1.0) < 1e-9


def test_yaml_missing_raises(tmp_path):
    """Missing YAML file raises FileNotFoundError."""
    missing = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        AdaptiveRiskScorer(policy_path=missing)


# ── 2. Normalization ─────────────────────────────────────────────────────────

def test_normalization_logic(scorer):
    """Verify normalization for each signal type."""
    normalized = scorer._normalize_signals({
        "restricted_zone": True,
        "repeated_approach": 3,
        "loitering": 60.0,
        "after_hours": False,
        "reasoning_confidence": 0.85,
    })
    assert normalized["restricted_zone"] == 1.0
    assert normalized["repeated_approach"] == pytest.approx(0.6)   # 3/5
    assert normalized["loitering"] == pytest.approx(0.5)            # 60/120
    assert normalized["after_hours"] == 0.0
    assert normalized["reasoning_confidence"] == pytest.approx(0.85)

    # Clamping: loitering > max should cap at 1.0
    clamped = scorer._normalize_signals({"loitering": 999.0})
    assert clamped["loitering"] == 1.0


# ── 3. Weighted scoring ──────────────────────────────────────────────────────

def test_weighted_score_all_zeros(scorer):
    """All signals inactive → risk_score == 0."""
    result = scorer.score({
        "restricted_zone": False,
        "repeated_approach": 0,
        "loitering": 0.0,
        "after_hours": False,
        "reasoning_confidence": 0.0,
    })
    assert result["risk_score"] == 0


def test_weighted_score_restricted_zone_only(scorer):
    """Only restricted_zone active → risk_score == 30 (weight 0.30 × 100)."""
    result = scorer.score({
        "restricted_zone": True,
        "repeated_approach": 0,
        "loitering": 0.0,
        "after_hours": False,
        "reasoning_confidence": 0.0,
    })
    assert result["risk_score"] == 30


# ── 4. Risk level classification ─────────────────────────────────────────────

def test_risk_level_low(scorer):
    """Score ≤ 40 → 'Low'."""
    result = scorer.score({
        "restricted_zone": True,    # 0.30 → score 30
        "repeated_approach": 0,
        "loitering": 0.0,
        "after_hours": False,
        "reasoning_confidence": 0.0,
    })
    assert result["risk_level"] == "Low"


def test_risk_level_high(scorer):
    """Score > 70 → 'High'."""
    result = scorer.score({
        "restricted_zone": True,     # 0.30
        "repeated_approach": 5,      # 0.25 (5/5 = 1.0)
        "loitering": 120.0,         # 0.15 (120/120 = 1.0)
        "after_hours": True,         # 0.20
        "reasoning_confidence": 1.0, # 0.10
    })
    # All max → 100
    assert result["risk_score"] == 100
    assert result["risk_level"] == "High"


# ── 5. Output structure ──────────────────────────────────────────────────────

def test_output_structure(scorer):
    """Result dict has the expected keys with correct types."""
    result = scorer.score({
        "restricted_zone": True,
        "repeated_approach": 2,
        "loitering": 30.0,
        "after_hours": True,
        "reasoning_confidence": 0.7,
    })
    assert isinstance(result["risk_score"], int)
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in ("Low", "Medium", "High")
    assert isinstance(result["risk_factors"], list)
    assert all(isinstance(f, str) for f in result["risk_factors"])
    # Non-zero signals should produce factors
    assert len(result["risk_factors"]) > 0
    # Confidence at threshold (0.7) should appear as a factor
    assert "High reasoning confidence" in result["risk_factors"]


# ── 6. Confidence factor threshold ───────────────────────────────────────────

def test_confidence_factor_below_threshold(scorer):
    """Low confidence (< 0.7) should NOT produce 'High reasoning confidence' factor."""
    result = scorer.score({
        "restricted_zone": False,
        "repeated_approach": 0,
        "loitering": 0.0,
        "after_hours": False,
        "reasoning_confidence": 0.5,
    })
    assert "High reasoning confidence" not in result["risk_factors"]
    # Score should still reflect the confidence weight contribution
    assert result["risk_score"] == round(0.10 * 0.5 * 100)


# ── 7. Normalization clamping ─────────────────────────────────────────────────

def test_negative_signal_clamped(scorer):
    """Negative raw values are clamped to 0.0 during normalization."""
    normalized = scorer._normalize_signals({
        "loitering": -10.0,
        "repeated_approach": -3,
        "reasoning_confidence": -0.5,
    })
    assert normalized["loitering"] == 0.0
    assert normalized["repeated_approach"] == 0.0
    assert normalized["reasoning_confidence"] == 0.0


# ── 8. Policy validation ─────────────────────────────────────────────────────

def test_negative_weight_raises(tmp_path):
    """Negative weight in policy raises ValueError."""
    bad_policy = tmp_path / "bad.yaml"
    bad_policy.write_text("""\
risk_scoring:
  restricted_zone:
    weight: -0.30
  repeated_approach:
    weight: 0.25
  loitering:
    weight: 0.15
  after_hours:
    weight: 0.20
  reasoning_confidence:
    weight: 0.10
risk_levels:
  low_max: 40
  medium_max: 70
normalization:
  loitering_max_seconds: 120.0
  repeated_approach_max_count: 5
""", encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        AdaptiveRiskScorer(policy_path=bad_policy)


def test_inverted_risk_levels_raises(tmp_path):
    """low_max >= medium_max in policy raises ValueError."""
    bad_policy = tmp_path / "bad.yaml"
    bad_policy.write_text("""\
risk_scoring:
  restricted_zone:
    weight: 0.30
  repeated_approach:
    weight: 0.25
  loitering:
    weight: 0.15
  after_hours:
    weight: 0.20
  reasoning_confidence:
    weight: 0.10
risk_levels:
  low_max: 70
  medium_max: 40
normalization:
  loitering_max_seconds: 120.0
  repeated_approach_max_count: 5
""", encoding="utf-8")
    with pytest.raises(ValueError, match="less than"):
        AdaptiveRiskScorer(policy_path=bad_policy)
