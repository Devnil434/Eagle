"""
Convert a TrackSequence into a numbered plain-text log for the LLM prompt.
No ML — pure string formatting.
"""
from __future__ import annotations
from libs.schemas.memory import TrackSequence


def sequence_to_text(seq: TrackSequence, max_events: int = 20) -> str:
    """
    Example output:
        Track #3 — 22.4s in restricted_door
        [00:00] zone_entry       → Person enters restricted_door
        [00:05] lingering        → Person standing near access point
        [00:12] near_keypad      → Person near keypad
    """
    if not seq.events:
        return f"Track #{seq.track_id} — no events recorded."

    first_ts = seq.events[0].timestamp_ms
    lines = [f"Track #{seq.track_id} — {seq.total_dwell:.1f}s in "
             f"{', '.join(seq.zones_visited) or 'unknown zone'}"]

    # Sample evenly if sequence is long
    events = seq.events
    if len(events) > max_events:
        step = len(events) // max_events
        events = events[::step][:max_events]

    for e in events:
        elapsed = (e.timestamp_ms - first_ts) / 1000
        mm = int(elapsed // 60)
        ss = int(elapsed % 60)
        hint = e.action_hint.value.replace("_", " ")
        zone_tag = f"in {e.zone}" if e.zone else "in corridor"
        lines.append(f"  [{mm:02d}:{ss:02d}] {hint:<20} → Person {hint} {zone_tag}")

    return "\n".join(lines)


def captions_to_text(captions: list[str]) -> str:
    if not captions:
        return "(no visual descriptions available)"
    return "\n".join(f"  {i+1}. {c}" for i, c in enumerate(captions))
