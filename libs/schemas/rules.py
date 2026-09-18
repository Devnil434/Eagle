"""
Schemas for configurable alert rules.

A rule describes which detected activity is worth notifying an operator about.
Every matching dimension is optional and an empty value means "any", so the
narrowest useful rule is two lines of YAML:

    - id: any_person
      object_types: [person]

Field names deliberately mirror the runtime data they are matched against —
`TrackEvent.label`, `TrackEvent.zone`, `TrackEvent.confidence`, and
`ActionHint` — so the config reads like the events it filters.
"""
from __future__ import annotations

from datetime import datetime, time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from libs.schemas.memory import ActionHint

# Group aliases usable in `object_types` alongside raw COCO classes, so a rule
# can say `vehicle` instead of enumerating every vehicle class YOLO emits.
OBJECT_GROUPS: dict[str, frozenset[str]] = {
    "person": frozenset({"person"}),
    "vehicle": frozenset({"car", "truck", "bus", "motorcycle", "bicycle"}),
    "bag": frozenset({"backpack", "handbag", "suitcase"}),
    "device": frozenset({"cell phone", "laptop"}),
}


def resolve_object_types(object_types: list[str]) -> frozenset[str]:
    """Expand group aliases to the concrete labels a detector emits.

    Unknown names pass through unchanged so any COCO class can be named
    directly without the taxonomy needing to know about it.
    """
    resolved: set[str] = set()
    for entry in object_types:
        key = entry.strip().lower()
        resolved |= OBJECT_GROUPS.get(key, frozenset({key}))
    return frozenset(resolved)


class Weekday(str, Enum):
    MON = "mon"
    TUE = "tue"
    WED = "wed"
    THU = "thu"
    FRI = "fri"
    SAT = "sat"
    SUN = "sun"


_WEEKDAY_BY_INDEX = (
    Weekday.MON, Weekday.TUE, Weekday.WED,
    Weekday.THU, Weekday.FRI, Weekday.SAT, Weekday.SUN,
)


class RuleTimeWindow(BaseModel):
    """A recurring daily window, optionally limited to certain weekdays.

    Windows may cross midnight (`22:00`–`06:00`).  When `days` is set it is
    checked against the weekday of the moment being tested, so a crossing
    window applies to both the late-evening and early-morning portions of each
    listed day.
    """

    start: time
    end:   time
    days:  list[Weekday] = Field(default_factory=list)

    def contains(self, moment: datetime) -> bool:
        if self.days and _WEEKDAY_BY_INDEX[moment.weekday()] not in self.days:
            return False

        current = moment.time()
        if self.start <= self.end:
            return self.start <= current <= self.end
        # Crosses midnight: inside if after the start or before the end.
        return current >= self.start or current <= self.end


class AlertRule(BaseModel):
    """One operator-defined condition for firing an alert.

    All dimensions are ANDed together; a rule matches when every populated
    dimension is satisfied.
    """

    id:      str  = Field(..., min_length=1, max_length=64)
    enabled: bool = True

    object_types:  list[str]        = Field(default_factory=list)
    zones:         list[str]        = Field(default_factory=list)
    action_hints:  list[ActionHint] = Field(default_factory=list)
    time_windows:  list[RuleTimeWindow] = Field(default_factory=list)
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)

    # Overrides settings.reasoning_cooldown_seconds for activity this rule matches,
    # letting noisy rules be throttled harder than quiet ones.
    cooldown_seconds: Optional[float] = Field(None, ge=0.0)
    description: str = ""

    @field_validator("object_types", "zones", mode="after")
    @classmethod
    def _strip_entries(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if v and v.strip()]

    @property
    def resolved_object_types(self) -> frozenset[str]:
        """Concrete labels this rule accepts; empty means any."""
        return resolve_object_types(self.object_types)


class RuleSet(BaseModel):
    """The parsed contents of an alert rules config file."""

    rules: list[AlertRule] = Field(default_factory=list)

    @property
    def enabled_rules(self) -> list[AlertRule]:
        return [rule for rule in self.rules if rule.enabled]

    @field_validator("rules", mode="after")
    @classmethod
    def _reject_duplicate_ids(cls, rules: list[AlertRule]) -> list[AlertRule]:
        seen: set[str] = set()
        for rule in rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id {rule.id!r}")
            seen.add(rule.id)
        return rules
