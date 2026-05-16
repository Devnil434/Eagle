from __future__ import annotations
import yaml
from pathlib import Path

from libs.config.settings import Settings

class PolicyLoader:
    def __init__(self):
        self.policy_path = Settings().policy_path
        self._cache = None
        self._last_mtime = 0

    def load_policy(self):
        path = Path(self.policy_path)

        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        mtime = path.stat().st_mtime

        if self._cache is None or mtime != self._last_mtime:
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)

                if data is None:
                    raise ValueError("Empty YAML file")

                # basic validation
                if "zones" not in data or "global" not in data:
                    raise ValueError("Invalid policy structure")

                self._cache = data
                self._last_mtime = mtime

            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format: {e}")

        return self._cache