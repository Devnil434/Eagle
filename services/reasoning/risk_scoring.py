"""
Adaptive Risk Scoring Engine for Eagle Surveillance.

Loads configurable weights from a YAML policy file and computes a normalised
0–100 risk score from multiple contextual signals.  Designed as a drop-in
replacement for the hardcoded severity weights previously defined in
``services.reasoning.pipeline._W``.

Usage
-----
    from services.reasoning.risk_scoring import AdaptiveRiskScorer

    scorer = AdaptiveRiskScorer("configs/risk_policy.yaml")
    result = scorer.score({
        "restricted_zone": True,
        "repeated_approach": 3,
        "loitering": 45.0,
        "after_hours": False,
        "reasoning_confidence": 0.85,
    })
    # result == {"risk_score": 62, "risk_level": "Medium", "risk_factors": [...]}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

import yaml

logger = logging.getLogger(__name__)

# Default policy path relative to the project root
_DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "risk_policy.yaml"

# Human-readable descriptions for each signal
_FACTOR_LABELS: dict[str, str] = {
    "restricted_zone": "Restricted zone detected",
    "repeated_approach": "Repeated approach behavior",
    "loitering": "Loitering / extended dwell time",
    "after_hours": "After-hours activity",
    "reasoning_confidence": "High reasoning confidence",
}


class RiskScoringResult(TypedDict):
    """Typed result returned by :meth:`AdaptiveRiskScorer.score`."""

    risk_score: int
    risk_level: str
    risk_factors: list[str]


class AdaptiveRiskScorer:
    """Context-aware risk scoring engine.

    Loads configurable weights from a YAML policy file and computes a
    normalised 0–100 risk score from multiple contextual signals.

    Parameters
    ----------
    policy_path:
        Path to the YAML policy file.  If ``None``, uses the default
        ``configs/risk_policy.yaml`` relative to the project root.

    Raises
    ------
    FileNotFoundError
        If the policy YAML file does not exist.
    ValueError
        If the YAML file is malformed or has an invalid structure.
    """

    def __init__(self, policy_path: str | Path | None = None) -> None:
        resolved = Path(policy_path) if policy_path else _DEFAULT_POLICY_PATH
        self._policy = self._load_policy(resolved)
        self._weights: dict[str, float] = {
            key: cfg["weight"]
            for key, cfg in self._policy["risk_scoring"].items()
        }
        self._risk_levels: dict[str, int] = self._policy["risk_levels"]
        self._normalization: dict[str, float] = self._policy["normalization"]

    # ── Public API ────────────────────────────────────────────────────────

    def score(self, signals: dict[str, Any]) -> RiskScoringResult:
        """Compute a context-aware risk score from raw contextual signals.

        Parameters
        ----------
        signals:
            Dictionary of raw signal values.  Expected keys:

            - ``restricted_zone``      – ``bool``
            - ``repeated_approach``    – ``int``  (entry count)
            - ``loitering``            – ``float`` (dwell seconds)
            - ``after_hours``          – ``bool``
            - ``reasoning_confidence`` – ``float`` in [0, 1]

            Missing keys are treated as zero / inactive.

        Returns
        -------
        RiskScoringResult
            A dict with ``risk_score`` (0–100), ``risk_level``
            (``"Low"`` / ``"Medium"`` / ``"High"``), and
            ``risk_factors`` (list of human-readable strings).
        """
        normalized = self._normalize_signals(signals)
        raw_score = self._compute_weighted_score(normalized)
        score_100 = round(raw_score * 100)
        score_100 = max(0, min(score_100, 100))

        return RiskScoringResult(
            risk_score=score_100,
            risk_level=self._classify_risk_level(score_100),
            risk_factors=self._identify_contributing_factors(normalized),
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _normalize_signals(self, signals: dict[str, Any]) -> dict[str, float]:
        """Normalise each raw signal to a value in [0.0, 1.0].

        Boolean signals map to 1.0 / 0.0.  Continuous signals are clamped
        against the normalization parameters defined in the policy YAML.
        """
        loitering_max = self._normalization["loitering_max_seconds"]
        approach_max = self._normalization["repeated_approach_max_count"]

        restricted = signals.get("restricted_zone", False)
        approach_count = signals.get("repeated_approach", 0)
        dwell = signals.get("loitering", 0.0)
        after_hours = signals.get("after_hours", False)
        confidence = signals.get("reasoning_confidence", 0.0)

        return {
            "restricted_zone": 1.0 if restricted else 0.0,
            "repeated_approach": max(0.0, min(approach_count / approach_max, 1.0)) if approach_max > 0 else 0.0,
            "loitering": max(0.0, min(dwell / loitering_max, 1.0)) if loitering_max > 0 else 0.0,
            "after_hours": 1.0 if after_hours else 0.0,
            "reasoning_confidence": max(0.0, min(float(confidence), 1.0)),
        }

    def _compute_weighted_score(self, normalized: dict[str, float]) -> float:
        """Apply YAML weights to normalised signals and return a 0–1 float."""
        total = 0.0
        for key, weight in self._weights.items():
            total += weight * normalized.get(key, 0.0)
        return min(total, 1.0)

    def _classify_risk_level(self, score_100: int) -> str:
        """Classify a 0–100 score into Low / Medium / High."""
        if score_100 <= self._risk_levels["low_max"]:
            return "Low"
        if score_100 <= self._risk_levels["medium_max"]:
            return "Medium"
        return "High"

    def _identify_contributing_factors(
        self, normalized: dict[str, float]
    ) -> list[str]:
        """Return human-readable labels for signals that contributed."""
        # Confidence threshold for reporting — only flag genuinely high values
        _CONFIDENCE_HIGH_THRESHOLD = 0.7

        factors: list[str] = []
        for key, value in normalized.items():
            if value > 0.0 and key in _FACTOR_LABELS:
                # "High reasoning confidence" should only appear when
                # confidence is genuinely high, not for any nonzero value.
                if key == "reasoning_confidence" and value < _CONFIDENCE_HIGH_THRESHOLD:
                    continue
                factors.append(_FACTOR_LABELS[key])
        return factors

    # ── YAML loading ──────────────────────────────────────────────────────

    @staticmethod
    def _load_policy(path: Path) -> dict[str, Any]:
        """Load and validate the YAML policy file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the YAML is empty, unparseable, missing required sections,
            or contains invalid weight / threshold values.
        """
        if not path.exists():
            raise FileNotFoundError(f"Risk policy file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in risk policy: {exc}") from exc

        if data is None:
            raise ValueError(f"Risk policy file is empty: {path}")

        for section in ("risk_scoring", "risk_levels", "normalization"):
            if section not in data:
                raise ValueError(
                    f"Risk policy missing required section: '{section}'"
                )

        # Lightweight weight validation
        for signal, cfg in data["risk_scoring"].items():
            w = cfg.get("weight", 0)
            if not isinstance(w, (int, float)) or w < 0:
                raise ValueError(
                    f"Weight for '{signal}' must be a non-negative number, got {w!r}"
                )
            if w > 1.0:
                raise ValueError(
                    f"Weight for '{signal}' must not exceed 1.0, got {w}"
                )

        # Risk-level threshold ordering
        levels = data["risk_levels"]
        if levels["low_max"] >= levels["medium_max"]:
            raise ValueError(
                f"risk_levels.low_max ({levels['low_max']}) must be less than "
                f"medium_max ({levels['medium_max']})"
            )

        return data
