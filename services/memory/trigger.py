from __future__ import annotations

from time import monotonic

from libs.config.settings import settings
from libs.schemas.memory import (
    TrackSequence,
    ActionHint,
)

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


def should_trigger_reasoning(
    seq: TrackSequence,
) -> bool:
    """
    Determine whether VLM/LLM reasoning should be triggered.

    Conditions:
    - Track is inside a restricted zone
    - Dwell time exceeds configured threshold
    - At least one suspicious action exists
    - Track is not inside cooldown window
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

    now = monotonic()

    # Cooldown check
    last_trigger = _reasoning_cooldowns.get(seq.track_id)

    if last_trigger is not None and (now - last_trigger < settings.reasoning_cooldown_seconds):
        return False

    # Start cooldown
    _reasoning_cooldowns[seq.track_id] = now

    return True
