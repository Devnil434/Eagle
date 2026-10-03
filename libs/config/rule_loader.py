"""
libs/config/rule_loader.py

Loads configurable alert rules from a YAML file.  Follows the conventions of
`zone_loader.py`:
  - ALERT_RULES_PATH environment variable override
  - Validation on load, with the offending rule named in the error
  - Hot reload, so operators can enable or disable rules without a restart

Reload is driven by the file's mtime rather than a background thread, because
the rules are read on the request path: there is no thread to supervise, and a
freshness check costs one `stat` at most every `rules_reload_seconds`.

A missing rules file is not an error.  It yields an empty rule set, which the
engine treats as "no opinion" so the pipeline keeps its default behaviour.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from libs.config.settings import settings
from libs.schemas.rules import RuleSet

logger = logging.getLogger(__name__)


def _resolve_config_path() -> Path:
    """Resolve the rules path from env var or settings."""
    env_path = os.environ.get("ALERT_RULES_PATH")
    return Path(env_path) if env_path else Path(settings.alert_rules_path)


def load_rules(config_path: Optional[Path] = None) -> RuleSet:
    """Load and validate alert rules from a YAML file.

    Args:
        config_path: Optional override. Falls back to the ALERT_RULES_PATH env
                     var, then `settings.alert_rules_path`.

    Returns:
        The parsed RuleSet; empty when the file does not exist.

    Raises:
        ValueError: If the YAML is malformed or any rule fails validation.
    """
    path = config_path or _resolve_config_path()

    if not path.exists():
        logger.info(
            "No alert rules file at '%s' — every event keeps the default "
            "trigger behaviour.", path,
        )
        return RuleSet()

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in alert rules file '{path}': {exc}") from exc

    if raw is None:
        return RuleSet()
    if not isinstance(raw, dict) or "rules" not in raw:
        raise ValueError(f"Alert rules file '{path}' must contain a 'rules' key.")
    if raw["rules"] is None:
        return RuleSet()

    try:
        rule_set = RuleSet(rules=raw["rules"])
    except ValidationError as exc:
        raise ValueError(f"Invalid alert rule in '{path}': {exc}") from exc

    logger.info(
        "Loaded %d alert rule(s) from %s (%d enabled).",
        len(rule_set.rules), path, len(rule_set.enabled_rules),
    )
    return rule_set


class RuleConfigLoader:
    """Thread-safe rule set with mtime-based hot reload.

    Usage:
        loader = RuleConfigLoader()
        rules = loader.get_rules()      # reloads if the file changed
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        reload_interval: Optional[float] = None,
    ) -> None:
        self._config_path = config_path or _resolve_config_path()
        self._reload_interval = (
            settings.rules_reload_seconds if reload_interval is None else reload_interval
        )
        self._lock = threading.RLock()
        self._rule_set = RuleSet()
        self._last_mtime: Optional[float] = None
        self._last_checked: float = 0.0

        # Initial load. A broken file must not stop the process from starting,
        # so it degrades to an empty rule set with the reason logged.
        try:
            self._reload()
        except ValueError as exc:
            logger.error("Alert rules disabled — %s", exc)

    def get_rules(self) -> RuleSet:
        """Return the current rule set, reloading first if the file changed."""
        self._maybe_reload()
        with self._lock:
            return self._rule_set

    def force_reload(self) -> RuleSet:
        """Reload immediately, bypassing the freshness interval."""
        try:
            self._reload()
        except ValueError as exc:
            logger.error("Keeping previous alert rules — %s", exc)
        with self._lock:
            return self._rule_set

    def _maybe_reload(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_checked < self._reload_interval:
                return
            self._last_checked = now

        try:
            mtime = self._config_path.stat().st_mtime
        except OSError:
            # File removed since the last load; keep serving what we have.
            return

        if mtime == self._last_mtime:
            return

        try:
            self._reload()
        except ValueError as exc:
            # Keep the last good rule set: a typo mid-edit must not silently
            # widen or narrow what gets alerted on.
            logger.error("Keeping previous alert rules — %s", exc)

    def _reload(self) -> None:
        rule_set = load_rules(self._config_path)
        try:
            mtime = self._config_path.stat().st_mtime
        except OSError:
            mtime = None
        with self._lock:
            self._rule_set = rule_set
            self._last_mtime = mtime
            self._last_checked = time.monotonic()
