"""
GET /rules            — list configured alert rules and their enabled state
GET /rules/{rule_id}  — fetch one rule

Rules are owned by `config/alert_rules.yaml` and hot-reloaded, so these
endpoints are read-only: they report what the pipeline is currently enforcing.
Toggling a rule means editing the file, which takes effect without a restart.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from libs.schemas.rules import AlertRule
from services.rules.provider import get_rule_engine

logger = logging.getLogger("eagle.rules")

router = APIRouter(prefix="/rules", tags=["rules"])


@router.get("", response_model=list[AlertRule])
def list_rules(
    enabled_only: bool = Query(
        False, description="Return only the rules currently in force."
    ),
) -> list[AlertRule]:
    """Return the configured alert rules, in evaluation order."""
    rules = get_rule_engine().rules
    if enabled_only:
        rules = [rule for rule in rules if rule.enabled]
    return rules


@router.get("/{rule_id}", response_model=AlertRule)
def get_rule(rule_id: str) -> AlertRule:
    """Return a single alert rule by id."""
    for rule in get_rule_engine().rules:
        if rule.id == rule_id:
            return rule
    raise HTTPException(status_code=404, detail=f"Alert rule {rule_id!r} not found")
