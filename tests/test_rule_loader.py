"""
Tests for loading alert rules from YAML.

A rules file is operator-editable and read on the request path, so the loader
has to be forgiving about absence and unforgiving about ambiguity: a missing
file disables the feature, while a malformed one must never silently change what
gets alerted on.
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from libs.config.rule_loader import RuleConfigLoader, load_rules

VALID_RULES = """
rules:
  - id: restricted_door_person
    object_types: [person]
    zones: [restricted_door]
    min_confidence: 0.6
  - id: after_hours_vehicle
    object_types: [vehicle]
    time_windows:
      - start: "19:00"
        end: "07:00"
    enabled: false
"""


def write_rules(tmp_path: Path, content: str, name: str = "alert_rules.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


# ── Loading ───────────────────────────────────────────────────────────────────

def test_loads_and_orders_rules(tmp_path):
    rule_set = load_rules(write_rules(tmp_path, VALID_RULES))

    assert [r.id for r in rule_set.rules] == [
        "restricted_door_person",
        "after_hours_vehicle",
    ]


def test_enabled_flag_is_honoured(tmp_path):
    rule_set = load_rules(write_rules(tmp_path, VALID_RULES))

    assert [r.id for r in rule_set.enabled_rules] == ["restricted_door_person"]


def test_nested_fields_are_parsed(tmp_path):
    rule_set = load_rules(write_rules(tmp_path, VALID_RULES))
    vehicle_rule = rule_set.rules[1]

    assert vehicle_rule.time_windows[0].start.hour == 19
    assert "car" in vehicle_rule.resolved_object_types


def test_missing_file_yields_an_empty_rule_set(tmp_path):
    """Absence is the default state and must not raise."""
    rule_set = load_rules(tmp_path / "does_not_exist.yaml")

    assert rule_set.rules == []
    assert rule_set.enabled_rules == []


def test_empty_file_yields_an_empty_rule_set(tmp_path):
    assert load_rules(write_rules(tmp_path, "")).rules == []


def test_null_rules_key_yields_an_empty_rule_set(tmp_path):
    assert load_rules(write_rules(tmp_path, "rules:\n")).rules == []


def test_the_shipped_example_config_is_valid():
    """The example operators are told to copy must actually load."""
    example = Path(__file__).parents[1] / "config" / "alert_rules.example.yaml"

    rule_set = load_rules(example)

    assert len(rule_set.rules) >= 1
    assert any(r.enabled for r in rule_set.rules)


# ── Rejection ─────────────────────────────────────────────────────────────────

def test_malformed_yaml_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_rules(write_rules(tmp_path, "rules: [unclosed\n"))


def test_missing_rules_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must contain a 'rules' key"):
        load_rules(write_rules(tmp_path, "policies: []\n"))


def test_invalid_rule_names_the_file(tmp_path):
    content = """
    rules:
      - id: bad
        min_confidence: 5.0
    """
    with pytest.raises(ValueError, match="Invalid alert rule"):
        load_rules(write_rules(tmp_path, content))


def test_duplicate_ids_are_rejected(tmp_path):
    content = """
    rules:
      - id: same
      - id: same
    """
    with pytest.raises(ValueError, match="Invalid alert rule"):
        load_rules(write_rules(tmp_path, content))


# ── Env override ──────────────────────────────────────────────────────────────

def test_env_var_overrides_the_configured_path(tmp_path, monkeypatch):
    path = write_rules(tmp_path, VALID_RULES, name="custom.yaml")
    monkeypatch.setenv("ALERT_RULES_PATH", str(path))

    assert len(load_rules().rules) == 2


# ── Hot reload ────────────────────────────────────────────────────────────────

def test_loader_serves_rules_from_the_file(tmp_path):
    loader = RuleConfigLoader(write_rules(tmp_path, VALID_RULES), reload_interval=0)

    assert len(loader.get_rules().rules) == 2


def test_loader_picks_up_edits(tmp_path):
    """Enabling a rule must take effect without a restart."""
    path = write_rules(tmp_path, VALID_RULES)
    loader = RuleConfigLoader(path, reload_interval=0)
    assert len(loader.get_rules().enabled_rules) == 1

    path.write_text(
        textwrap.dedent(
            """
            rules:
              - id: everything
            """
        ),
        encoding="utf-8",
    )
    # mtime resolution can be coarse, so make the change unambiguous.
    os.utime(path, (0, 0))

    assert [r.id for r in loader.get_rules().rules] == ["everything"]


def test_loader_keeps_last_good_rules_when_an_edit_breaks(tmp_path):
    """A typo mid-edit must not widen or narrow what gets alerted on."""
    path = write_rules(tmp_path, VALID_RULES)
    loader = RuleConfigLoader(path, reload_interval=0)

    path.write_text("rules: [unclosed\n", encoding="utf-8")
    os.utime(path, (0, 0))

    assert len(loader.get_rules().rules) == 2


def test_loader_starts_empty_when_the_file_is_broken(tmp_path):
    """A broken file at startup must not stop the process from booting."""
    loader = RuleConfigLoader(write_rules(tmp_path, "rules: [unclosed\n"))

    assert loader.get_rules().rules == []


def test_loader_tolerates_a_deleted_file(tmp_path):
    path = write_rules(tmp_path, VALID_RULES)
    loader = RuleConfigLoader(path, reload_interval=0)
    path.unlink()

    assert len(loader.get_rules().rules) == 2


def test_force_reload_bypasses_the_freshness_interval(tmp_path):
    path = write_rules(tmp_path, VALID_RULES)
    loader = RuleConfigLoader(path, reload_interval=3600)

    path.write_text("rules:\n  - id: only_one\n", encoding="utf-8")
    os.utime(path, (0, 0))

    assert [r.id for r in loader.force_reload().rules] == ["only_one"]
