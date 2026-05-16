"""
services/detection/zones.py

Zone definitions are loaded dynamically from YAML using ZoneConfigLoader.
Static DEFAULT_ZONES have been removed in favor of runtime configuration.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np
from dataclasses import dataclass
from libs.config.zone_loader import ZoneConfigLoader

logger = logging.getLogger(__name__)


# ── Data Model (compatibility only) ─────────────────────────────
@dataclass
class Zone:
    name: str
    polygon: list[tuple[int, int]]
    alert_on_entry: bool = True
    color_bgr: tuple[int, int, int] = (0, 0, 255)

    def as_array(self) -> np.ndarray:
        return np.array(self.polygon, dtype=np.int32)


# ── Zone Loader (production system) ─────────────────────────────
_loader = ZoneConfigLoader()
_loader.start()


# ── Core API ────────────────────────────────────────────────────
def get_zones() -> list[dict]:
    """
    Return zones from YAML config (live reloaded).
    Each zone is a dict: name, polygon, alert_on_entry, color_hex
    """
    return _loader.get_zones()


def get_camera_id() -> str | None:
    """Return active camera ID from config."""
    return _loader.get_camera_id()


# ── Geometry Helpers ────────────────────────────────────────────
def point_in_zone(x: float, y: float, zone) -> bool:
    """
    Check if a point is inside a polygon zone.
    Supports both dict-based and Zone objects.
    """
    polygon = zone["polygon"] if isinstance(zone, dict) else zone.polygon
    polygon_np = np.array(polygon, dtype=np.int32)

    result = cv2.pointPolygonTest(polygon_np, (float(x), float(y)), False)
    return result >= 0


def get_zones_for_point(x: float, y: float, zones=None) -> list:
    """
    Return all zones containing the given point.
    """
    zones = zones or get_zones()
    return [z for z in zones if point_in_zone(x, y, z)]


# ── Backward compatibility ──────────────────────────────────────
DEFAULT_ZONES = get_zones()