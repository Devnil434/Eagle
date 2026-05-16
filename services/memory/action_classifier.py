"""
action_classifier.py — Deterministic rules to infer ActionHint from tracker state.

No ML here — pure geometric/temporal logic:
  - Is the centroid moving?  → WALKING vs STANDING
  - Dwell time threshold?    → LINGERING
  - First frame in zone?     → ZONE_ENTRY
  - Near keypad bounding box? → NEAR_KEYPAD
"""
from __future__ import annotations

import math
from libs.schemas.tracking import TrackedObject
from libs.schemas.memory   import ActionHint


LINGERING_THRESHOLD_SEC  = 5.0    # seconds in zone before LINGERING
MOVEMENT_THRESHOLD_PX    = 8.0    # pixels/frame centroid must move to be WALKING
NEAR_KEYPAD_DIST_PX      = 80.0   # pixels from keypad centre to count as NEAR_KEYPAD

# Approximate keypad centre (pixels) — configurable via env in production
KEYPAD_CENTER = (600.0, 280.0)


def classify_action(
    obj:      TrackedObject,
    prev_obj: TrackedObject | None,
    known_zone_entries: dict[int, set[str]],   # track_id → set of zones already entered
) -> ActionHint:
    """
    Infer an ActionHint for this frame based on tracker state and history.

    Args:
        obj:                Current TrackedObject.
        prev_obj:           Same track in previous frame (None if BORN this frame).
        known_zone_entries: Dict tracking which zones each track has entered before.

    Returns:
        ActionHint enum value.
    """
    # ── Zone entry detection ───────────────────────────────────────────────
    if obj.zones_present:
        zone = obj.zones_present[0]
        entered = known_zone_entries.setdefault(obj.track_id, set())
        if zone not in entered:
            entered.add(zone)
            return ActionHint.ZONE_ENTRY

    # ── Lingering ─────────────────────────────────────────────────────────
    if obj.zones_present and obj.dwell_time_seconds > LINGERING_THRESHOLD_SEC:
        return ActionHint.LINGERING

    # ── Proximity to keypad ────────────────────────────────────────────────
    cx, cy = obj.center
    dist   = math.hypot(cx - KEYPAD_CENTER[0], cy - KEYPAD_CENTER[1])
    if dist < NEAR_KEYPAD_DIST_PX:
        return ActionHint.NEAR_KEYPAD

    # ── Movement detection ────────────────────────────────────────────────
    if prev_obj is not None:
        dx = obj.center[0] - prev_obj.center[0]
        dy = obj.center[1] - prev_obj.center[1]
        if math.hypot(dx, dy) > MOVEMENT_THRESHOLD_PX:
            return ActionHint.WALKING
        return ActionHint.STANDING

    return ActionHint.UNKNOWN