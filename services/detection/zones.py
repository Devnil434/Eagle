"""
services/detection/zones.py

Zone definitions are now loaded from config/zones.yaml via ZoneConfigLoader.
Set ZONES_CONFIG_PATH env var to override the default config location.
"""

import logging
import cv2
import numpy as np
from libs.config.zone_loader import ZoneConfigLoader
from typing import List, Tuple


class Zone:
    """Lightweight wrapper around zone dicts from ZoneConfigLoader."""

    def __init__(self, data: dict) -> None:
        self.name: str = data.get("name")
        self.polygon: List[Tuple[float, float]] = data.get("polygon", [])
        self.alert_on_entry: bool = data.get("alert_on_entry", False)
        self.color_hex: str = data.get("color_hex", "#FF0000")

    def as_array(self) -> np.ndarray:
        return np.array(self.polygon, dtype=np.int32)

    @property
    def color_bgr(self) -> Tuple[int, int, int]:
        # hex #RRGGBB -> BGR tuple for OpenCV
        h = self.color_hex.lstrip("#")
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return (b, g, r)

    def contains_point(self, x: float, y: float) -> bool:
        # Ray casting algorithm for point-in-polygon
        pts = self.polygon
        inside = False
        n = len(pts)
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            intersect = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            )
            if intersect:
                inside = not inside
            j = i
        return inside
logger = logging.getLogger(__name__)

# Module-level singleton loader — starts hot-reload background thread
_loader = ZoneConfigLoader()
_loader.start()


def get_zones() -> list["Zone"]:
    """
    Return the current list of Zone objects loaded from YAML.
    """
    return [Zone(z) for z in _loader.get_zones()]


def get_camera_id() -> str | None:
    """Return the camera_id from the active zone config."""
    return _loader.get_camera_id()


# Alias for legacy support in detection.py
DEFAULT_ZONES = get_zones()
# Convenience alias for code that previously referenced DEFAULT_ZONES directly
DEFAULT_ZONES = get_zones()


def get_zones_for_point(x: float, y: float) -> List[Zone]:
    """Return list of Zone objects that contain the point (x, y).

    Coordinates are in image pixel space (x horizontal, y vertical).
    """
    zones = get_zones()
    return [z for z in zones if z.contains_point(x, y)]
