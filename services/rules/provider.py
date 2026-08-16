"""
Process-wide access to the configured rule engine.

Keeping the loader here rather than in `engine.py` leaves the engine pure and
gives the trigger path a single place to fetch rules from. The loader is created
lazily so that importing the pipeline never touches the filesystem.
"""
from __future__ import annotations

import threading
from typing import Optional

from libs.config.rule_loader import RuleConfigLoader
from services.rules.engine import RuleEngine

_lock = threading.Lock()
_loader: Optional[RuleConfigLoader] = None


def get_rule_engine() -> RuleEngine:
    """Return an engine over the current rule file contents.

    The loader reloads when the file changes, so enabling or disabling a rule
    takes effect without a restart.
    """
    global _loader
    if _loader is None:
        with _lock:
            if _loader is None:
                _loader = RuleConfigLoader()
    return RuleEngine(_loader.get_rules())


def reset_rule_engine() -> None:
    """Drop the cached loader. Used by tests and after a config path change."""
    global _loader
    with _lock:
        _loader = None
