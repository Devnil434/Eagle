"""
libs/config/zone_loader.py

Loads zone definitions from a YAML config file.
Supports:
  - ZONES_CONFIG_PATH environment variable override
  - Polygon integrity validation
  - Hot reload every 60 seconds
"""

import os
import threading
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "config" / "zones.yaml"
_RELOAD_INTERVAL_SECONDS = 60


def _resolve_config_path() -> Path:
    """Resolve config path from env var or default."""
    env_path = os.environ.get("ZONES_CONFIG_PATH")
    return Path(env_path) if env_path else DEFAULT_CONFIG_PATH


def _validate_polygon(polygon: Any, zone_name: str) -> None:
    """
    Validate that a polygon is a list of at least 3 [x, y] integer/float pairs.
    Raises ValueError on any violation.
    """
    if not isinstance(polygon, list):
        raise ValueError(
            f"Zone '{zone_name}': polygon must be a list, got {type(polygon).__name__}"
        )
    if len(polygon) < 3:
        raise ValueError(
            f"Zone '{zone_name}': polygon must have at least 3 points, got {len(polygon)}"
        )
    for i, point in enumerate(polygon):
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(isinstance(coord, (int, float)) for coord in point)
        ):
            raise ValueError(
                f"Zone '{zone_name}': point[{i}] must be a pair of numbers, got {point!r}"
            )


def load_zones(config_path: Path | None = None) -> dict:
    """
    Load and validate zones from a YAML file.

    Args:
        config_path: Optional override path. Falls back to ZONES_CONFIG_PATH
                     env var, then the default config/zones.yaml.

    Returns:
        Parsed and validated config dict with keys: camera_id, zones.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If any zone has an invalid polygon.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = config_path or _resolve_config_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Zone config file not found: {path}. "
            f"Set ZONES_CONFIG_PATH or create config/zones.yaml."
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "zones" not in config:
        raise ValueError(f"Zone config at '{path}' must contain a 'zones' key.")

    for zone in config["zones"]:
        if "name" not in zone:
            raise ValueError(f"Every zone must have a 'name' field. Got: {zone}")
        if "polygon" not in zone:
            raise ValueError(f"Zone '{zone['name']}' is missing 'polygon'.")
        _validate_polygon(zone["polygon"], zone["name"])

    logger.info("Loaded %d zone(s) from %s", len(config["zones"]), path)
    return config


class ZoneConfigLoader:
    """
    Thread-safe loader that hot-reloads zone config every 60 seconds.

    Usage:
        loader = ZoneConfigLoader()
        loader.start()
        zones = loader.get_zones()
    """

    def __init__(self, config_path: Path | None = None, reload_interval: int = _RELOAD_INTERVAL_SECONDS):
        self._config_path = config_path or _resolve_config_path()
        self._reload_interval = reload_interval
        self._lock = threading.RLock()
        self._config: dict = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Initial load (fail fast if config is broken)
        self._reload()

    def _reload(self) -> None:
        try:
            new_config = load_zones(self._config_path)
            with self._lock:
                self._config = new_config
            logger.debug("Zone config hot-reloaded successfully.")
        except FileNotFoundError:
            logger.warning(
                "Zone config file not found at '%s'. Keeping previous config.",
                self._config_path,
            )
        except Exception as exc:
            logger.error("Failed to reload zone config: %s", exc)

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self._reload_interval):
            self._reload()

    def start(self) -> None:
        """Start the background hot-reload thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ZoneConfigReloader")
        self._thread.start()
        logger.info("Zone config hot-reload started (interval: %ds).", self._reload_interval)

    def stop(self) -> None:
        """Stop the background hot-reload thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_zones(self) -> list[dict]:
        """Return the current list of zone definitions (thread-safe)."""
        with self._lock:
            return self._config.get("zones", [])

    def get_camera_id(self) -> str | None:
        """Return the camera_id from config (thread-safe)."""
        with self._lock:
            return self._config.get("camera_id")
