"""
Alert rule evaluation.

Pure matching logic: the rules, the activity, and the current time all arrive as
arguments, so a decision is reproducible and testable without a config file or a
frozen clock.

Semantics
---------
Rules are an allow-list. When at least one rule is enabled, activity must match
one of them to be alerted on. When no rule is enabled the engine *abstains*,
and the caller keeps whatever behaviour it had before rules existed — that is
what makes the feature safe to ship without a config file.

Within a rule every populated dimension must be satisfied (AND); an empty
dimension means "any". Rules are tested in file order and the first match wins,
so the most specific rules belong at the top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

from libs.schemas.memory import ActionHint, TrackSequence
from libs.schemas.rules import AlertRule, RuleSet


@dataclass(frozen=True)
class RuleContext:
    """The activity being judged, flattened to what rules match on."""

    object_labels: frozenset[str] = frozenset()
    zones:         frozenset[str] = frozenset()
    action_hints:  frozenset[ActionHint] = frozenset()
    confidence:    float = 0.0
    moment:        Optional[datetime] = None

    @classmethod
    def from_sequence(
        cls, seq: TrackSequence, moment: Optional[datetime] = None
    ) -> "RuleContext":
        """Build a context from a track sequence.

        Confidence is the highest seen in the sequence: a rule with a confidence
        floor should fire when the detector was ever that sure, not only if it
        was sure on the final frame.
        """
        return cls(
            object_labels = frozenset(e.label for e in seq.events if e.label),
            zones         = frozenset(e.zone for e in seq.events if e.zone),
            action_hints  = frozenset(e.action_hint for e in seq.events),
            confidence    = max((e.confidence for e in seq.events), default=0.0),
            moment        = moment,
        )


@dataclass(frozen=True)
class RuleDecision:
    """Outcome of evaluating a context against a rule set."""

    matched:   bool = False
    # True when no rule was enabled, meaning the engine expressed no opinion.
    abstained: bool = False
    rule_id:   Optional[str] = None
    cooldown_seconds: Optional[float] = None
    # Rules that were considered and rejected, for explaining a suppression.
    rejected:  tuple[str, ...] = field(default=())

    @property
    def suppressed(self) -> bool:
        """True when the engine actively blocked this activity."""
        return not self.matched and not self.abstained


class RuleEngine:
    """Evaluates activity against a set of alert rules."""

    def __init__(self, rules: Sequence[AlertRule] | RuleSet | None = None) -> None:
        if isinstance(rules, RuleSet):
            self._rules = list(rules.rules)
        else:
            self._rules = list(rules or [])

    @property
    def rules(self) -> list[AlertRule]:
        return list(self._rules)

    def evaluate(self, context: RuleContext) -> RuleDecision:
        """Decide whether `context` should raise an alert."""
        enabled = [rule for rule in self._rules if rule.enabled]
        if not enabled:
            return RuleDecision(matched=False, abstained=True)

        rejected: list[str] = []
        for rule in enabled:
            if self._matches(rule, context):
                return RuleDecision(
                    matched          = True,
                    rule_id          = rule.id,
                    cooldown_seconds = rule.cooldown_seconds,
                    rejected         = tuple(rejected),
                )
            rejected.append(rule.id)

        return RuleDecision(matched=False, rejected=tuple(rejected))

    @staticmethod
    def _matches(rule: AlertRule, context: RuleContext) -> bool:
        if context.confidence < rule.min_confidence:
            return False

        accepted_labels = rule.resolved_object_types
        if accepted_labels and not (accepted_labels & context.object_labels):
            return False

        if rule.zones and not (set(rule.zones) & context.zones):
            return False

        if rule.action_hints and not (set(rule.action_hints) & context.action_hints):
            return False

        if rule.time_windows:
            # A time-scoped rule cannot be judged without a timestamp, so treat
            # a missing one as a non-match rather than silently always firing.
            if context.moment is None:
                return False
            if not any(window.contains(context.moment) for window in rule.time_windows):
                return False

        return True
