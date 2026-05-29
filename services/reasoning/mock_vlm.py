"""
mock_vlm.py — Deterministic mock VLM captioner and LLM reasoner.

Designed for CI environments and contributors who cannot run GPU-heavy
models (Ollama / LLaVA-Next) locally.  Activated by setting:

    VLM_PROVIDER=mock

Both classes are fully deterministic: the same action_hint always
produces the same caption / reasoning output, making them safe for
automated integration tests.
"""
from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# MockVLMCaptioner
# ---------------------------------------------------------------------------

class MockVLMCaptioner:
    """
    Simulates a Vision-Language Model (VLM) caption generator.

    Instead of sending a frame to Ollama / LLaVA-Next, it maps the
    ``action_hint`` string produced by the action classifier to a
    pre-defined natural-language caption.

    Usage::

        captioner = MockVLMCaptioner()
        caption = captioner.caption(frame=None, action_hint="ZONE_ENTRY")
        # → "Person enters restricted area."

    The ``frame`` argument is accepted for API compatibility with the real
    VLM client but is intentionally ignored.
    """

    # Maps ActionHint values (upper-case strings) → caption text.
    # Keys mirror ``libs.schemas.memory.ActionHint`` enum values.
    CAPTION_MAP: dict[str, str] = {
        "ZONE_ENTRY":         "Person enters restricted area.",
        "LINGERING":          "Person remains near access point.",
        "NEAR_KEYPAD":        "Person interacts with keypad.",
        "REPEATED_APPROACH":  "Person repeatedly approaches restricted zone.",
        "WALKING":            "Person is walking through the scene.",
        "STANDING":           "Person is standing still.",
        "ZONE_EXIT":          "Person exits the monitored zone.",
        "UNKNOWN":            "Person visible in frame.",
    }

    #: Fallback caption when the hint is not in CAPTION_MAP.
    DEFAULT_CAPTION: str = "Person visible in frame."

    def caption(
        self,
        frame: Any,
        action_hint: str,
    ) -> str:
        """
        Return a deterministic caption for the given action hint.

        Args:
            frame:        Video frame (numpy array or None).  Ignored by
                          this mock — present for API compatibility only.
            action_hint:  Coarse action label from the action classifier,
                          e.g. ``"ZONE_ENTRY"``.  Case-insensitive.

        Returns:
            A natural-language description of the observed behaviour.
        """
        normalised = action_hint.upper() if action_hint else "UNKNOWN"
        return self.CAPTION_MAP.get(normalised, self.DEFAULT_CAPTION)


# ---------------------------------------------------------------------------
# ReasoningOutput
# ---------------------------------------------------------------------------

class ReasoningOutput:
    """
    Structured output from the LLM reasoning layer.

    Attributes:
        label:      ``"Suspicious"`` or ``"Normal"``.
        confidence: Float in [0, 1].
        reason:     Human-readable explanation.
    """

    __slots__ = ("label", "confidence", "reason")

    def __init__(self, label: str, confidence: float, reason: str) -> None:
        self.label = label
        self.confidence = confidence
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "label":      self.label,
            "confidence": self.confidence,
            "reason":     self.reason,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ReasoningOutput(label={self.label!r}, "
            f"confidence={self.confidence}, reason={self.reason!r})"
        )


# ---------------------------------------------------------------------------
# MockLLMReasoner
# ---------------------------------------------------------------------------

class MockLLMReasoner:
    """
    Simulates the LLM reasoning layer that classifies intent from a
    sequence of VLM captions.

    Returns deterministic :class:`ReasoningOutput` objects based on
    keywords found in the caption text.  No external API calls are made.

    Usage::

        reasoner = MockLLMReasoner()
        result = reasoner.reason(
            captions=["Person interacts with keypad.",
                      "Person remains near access point."]
        )
        # result.label      → "Suspicious"
        # result.confidence → 0.88
        # result.reason     → "Repeated keypad interaction detected."

    The ``captions`` argument may also be a single string for convenience.
    """

    # Ordered list of (keyword_substring, ReasoningOutput) pairs.
    # The first matching rule wins.
    _RULES: list[tuple[str, str, float, str]] = [
        # (keyword,              label,        confidence, reason)
        (
            "keypad",
            "Suspicious",
            0.88,
            "Repeated keypad interaction detected.",
        ),
        (
            "restricted area",
            "Suspicious",
            0.85,
            "Unauthorised entry into restricted area detected.",
        ),
        (
            "repeatedly approaches",
            "Suspicious",
            0.82,
            "Subject repeatedly approaches restricted zone.",
        ),
        (
            "remains near",
            "Suspicious",
            0.75,
            "Prolonged loitering near access point detected.",
        ),
        (
            "walking",
            "Normal",
            0.95,
            "Subject is moving normally through the scene.",
        ),
        (
            "standing",
            "Normal",
            0.90,
            "Subject is stationary; no suspicious behaviour observed.",
        ),
        (
            "exits",
            "Normal",
            0.92,
            "Subject has left the monitored zone.",
        ),
    ]

    #: Fallback output when no rule matches.
    _DEFAULT = ReasoningOutput(
        label="Normal",
        confidence=0.70,
        reason="No suspicious behaviour detected.",
    )

    def reason(
        self,
        captions: list[str] | str,
        action_hints: Optional[list[str]] = None,
    ) -> ReasoningOutput:
        """
        Classify intent from one or more VLM captions.

        Args:
            captions:     A single caption string or a list of captions
                          representing a temporal sequence.
            action_hints: Optional list of raw action hint strings (e.g.
                          ``["ZONE_ENTRY", "LINGERING"]``).  When provided,
                          they are appended to the combined caption text
                          before rule matching, allowing hint-based
                          overrides.

        Returns:
            A :class:`ReasoningOutput` with ``label``, ``confidence``,
            and ``reason`` fields.
        """
        if isinstance(captions, str):
            captions = [captions]

        # Combine all captions into one lower-case blob for matching.
        combined = " ".join(captions).lower()

        # Optionally fold in raw action hints.
        if action_hints:
            combined += " " + " ".join(action_hints).lower()

        for keyword, label, confidence, reason in self._RULES:
            if keyword.lower() in combined:
                return ReasoningOutput(
                    label=label,
                    confidence=confidence,
                    reason=reason,
                )

        return self._DEFAULT


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_vlm_captioner(provider: str = "mock") -> MockVLMCaptioner:
    """
    Return a VLM captioner instance for the given provider name.

    Currently only ``"mock"`` is supported.  Future providers (e.g.
    ``"ollama"``) will be added here without changing call sites.

    Args:
        provider: Value of the ``VLM_PROVIDER`` environment variable.

    Returns:
        A captioner with a ``caption(frame, action_hint)`` method.

    Raises:
        ValueError: If an unsupported provider is requested.
    """
    if provider.lower() == "mock":
        return MockVLMCaptioner()
    raise ValueError(
        f"Unsupported VLM_PROVIDER={provider!r}. "
        "Supported values: 'mock'."
    )


def get_llm_reasoner(provider: str = "mock") -> MockLLMReasoner:
    """
    Return an LLM reasoner instance for the given provider name.

    Args:
        provider: Value of the ``VLM_PROVIDER`` environment variable.

    Returns:
        A reasoner with a ``reason(captions, action_hints)`` method.

    Raises:
        ValueError: If an unsupported provider is requested.
    """
    if provider.lower() == "mock":
        return MockLLMReasoner()
    raise ValueError(
        f"Unsupported VLM_PROVIDER={provider!r}. "
        "Supported values: 'mock'."
    )
