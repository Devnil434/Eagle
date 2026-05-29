"""
visualizer.py — Draw tracked bounding boxes, IDs, dwell times, and trajectories.
Used by the CLI demo and optionally the FastAPI /snapshot endpoint.
"""
from __future__ import annotations

import colorsys
import numpy as np
import cv2

from libs.schemas.tracking import TrackedFrame


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic per-track color using HSV wheel."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)   # BGR


def draw_tracks(frame: np.ndarray, tracked: TrackedFrame) -> np.ndarray:
    """
    Overlay bounding boxes, track IDs, dwell times, and trajectory paths.

    Args:
        frame:   BGR image.
        tracked: TrackedFrame from Tracker.update().

    Returns:
        Annotated BGR image.
    """
    out = frame.copy()

    for obj in tracked.tracks:
        color = _track_color(obj.track_id)
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]

        # Bounding box — thicker if in restricted zone
        thickness = 3 if obj.zones_present else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Track ID + dwell label
        header = f"#{obj.track_id}  {obj.dwell_time_seconds:.1f}s"
        if obj.zones_present:
            header += f"  [{obj.zones_present[0]}]"

        (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, header, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Centroid
        cx, cy = int(obj.center[0]), int(obj.center[1])
        cv2.circle(out, (cx, cy), 4, color, -1)

        # Trajectory path
        pts = [(int(p.x), int(p.y)) for p in obj.trajectory[-40:]]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(out, pts[i - 1], pts[i], faded, 1)

    # HUD
    cv2.putText(out,
                f"Frame: {tracked.frame_id}  |  Active tracks: {len(tracked.tracks)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out