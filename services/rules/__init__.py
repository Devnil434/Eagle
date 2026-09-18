"""
Configurable alert rules.

Operators describe which activity deserves a notification in
`config/alert_rules.yaml`; those rules are evaluated before an alert is raised.

    engine   — pure matching (no I/O, no clock)
    provider — the process-wide engine, rebuilt when the config file changes

Only the pure engine is re-exported here. `provider` reads YAML, so importing it
is left to the caller that actually needs the configured rules — that keeps the
matcher importable in environments without a YAML parser.

With no rules configured the engine abstains and the pipeline keeps its default
behaviour, so the feature is inert until an operator opts in.
"""
from services.rules.engine import RuleContext, RuleDecision, RuleEngine

__all__ = [
    "RuleContext",
    "RuleDecision",
    "RuleEngine",
]
