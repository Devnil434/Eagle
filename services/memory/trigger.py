from __future__ import annotations

import logging
from datetime import datetime
from time import monotonic
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from libs.config.settings import settings
from libs.schemas.memory import (
    TrackSequence,
    ActionHint,
)
from services.rules.engine import RuleContext, RuleDecision, RuleEngine

logger = logging.getLogger(__name__)

_reasoning_cooldowns: dict[int, float] = {}

SUSPICIOUS_ACTIONS = {
    ActionHint.LINGERING,
    ActionHint.NEAR_KEYPAD,
    ActionHint.REPEATED_APPROACH,
}


def reset_cooldown(track_id: int) -> None:
    """
    Clear cooldown state after reasoning completes.
    """
    _reasoning_cooldowns.pop(track_id, None)


def _rules_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(settings.rules_timezone)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(
            "Unknown rules_timezone %r — evaluating time windows in UTC.",
            settings.rules_timezone,
        )
        return ZoneInfo("UTC")


def _evaluate_rules(seq: TrackSequence, engine: Optional[RuleEngine]) -> RuleDecision:
    """Ask the configured rules whether this activity is worth alerting on."""
    if engine is None:
        from services.rules.provider import get_rule_engine

        engine = get_rule_engine()

    moment = datetime.now(tz=_rules_timezone())
    if seq.events:
        # Judge the activity at the time it happened, not at evaluation time.
        moment = datetime.fromtimestamp(
            seq.events[-1].timestamp_ms / 1000, tz=_rules_timezone()
        )

    return engine.evaluate(RuleContext.from_sequence(seq, moment=moment))


def should_trigger_reasoning(
    seq: TrackSequence,
    engine: Optional[RuleEngine] = None,
) -> bool:
    """
    Determine whether VLM/LLM reasoning should be triggered.

    Conditions:
    - Track is inside a restricted zone
    - Dwell time exceeds configured threshold
    - At least one suspicious action exists
    - Activity matches an enabled alert rule, when any rule is configured
    - Track is not inside cooldown window

    Configured rules narrow this gate; they never widen it. With no rules
    configured the engine abstains and behaviour is unchanged.

    Args:
        seq:    The track's recent event sequence.
        engine: Rule engine to consult. Defaults to the configured rule file.
    """

    if not seq.events:
        return False

    # Zone check
    if not seq.zones_visited:
        return False

    # Dwell threshold
    if seq.total_dwell < settings.reasoning_dwell_threshold_seconds:
        return False

    # Suspicious action check
    has_suspicious_action = any(event.action_hint in SUSPICIOUS_ACTIONS for event in seq.events)

    if not has_suspicious_action:
        return False

    # Configurable alert rules
    decision = _evaluate_rules(seq, engine)
    if decision.suppressed:
        logger.debug(
            "Rules suppressed track %d (considered: %s)",
            seq.track_id, ", ".join(decision.rejected) or "none",
        )
        return False

    now = monotonic()

    # Cooldown check — a matched rule may throttle harder than the global default
    cooldown = (
        decision.cooldown_seconds
        if decision.cooldown_seconds is not None
        else settings.reasoning_cooldown_seconds
    )
    last_trigger = _reasoning_cooldowns.get(seq.track_id)

    if last_trigger is not None and (now - last_trigger < cooldown):
        return False

    # Start cooldown
    _reasoning_cooldowns[seq.track_id] = now

    if decision.matched:
        logger.info(
            "Track %d matched alert rule %r", seq.track_id, decision.rule_id
        )

    return True
