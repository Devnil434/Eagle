"""
zones.py — Define restricted polygon regions for the scene.

A zone is a named polygon defined as a list of (x, y) pixel coordinates.
Points must be in clockwise or counter-clockwise order.
"""
from dataclasses import dataclass, field
import numpy as np
from libs.config.settings import settings

@dataclass
class Zone:
    name: str
    polygon: list[tuple[int, int]]  # list of (x, y) corner points
    alert_on_entry: bool = True
    color_bgr: tuple[int, int, int] = (0, 0, 255)  # red by default

    def as_array(self) -> np.ndarray:
        """Return polygon as numpy array for cv2.pointPolygonTest."""
        return np.array(self.polygon, dtype=np.int32)


# ─── Default zones for a sample indoor corridor scene ───────────────────────
DEFAULT_ZONES: list[Zone] = [
    Zone(
        name="restricted_door",
        polygon=[(540, 200), (740, 200), (740, 480), (540, 480)],
        alert_on_entry=True,
        color_bgr=(0, 0, 255),      # red
    ),
    Zone(
        name="keypad_area",
        polygon=[(620, 280), (720, 280), (720, 420), (620, 420)],
        alert_on_entry=True,
        color_bgr=(0, 165, 255),    # orange
    ),
    Zone(
        name="safe_corridor",
        polygon=[(0, 0), (settings.track_ttl_seconds, 0), (settings.track_ttl_seconds, 480), (0, 480)],
        alert_on_entry=False,
        color_bgr=(0, 255, 0),      # green
    ),
]


def point_in_zone(x: float, y: float, zone: Zone) -> bool:
    """Return True if point (x, y) is inside the zone polygon."""
    import cv2
    result = cv2.pointPolygonTest(zone.as_array(), (float(x), float(y)), False)
    return result >= 0


def get_zones_for_point(x: float, y: float, zones: list[Zone] | None = None) -> list[Zone]:
    """Return all zones that contain point (x, y)."""
    zones = zones or DEFAULT_ZONES
    return [z for z in zones if point_in_zone(x, y, z)]