"""
services.action_recognition.temporal_buffer
============================================
Track-wise rolling frame buffer for temporal action recognition.

Each active track maintains a deque of cropped person images (numpy arrays).
When a track disappears, its buffer is pruned automatically.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np


CROP_SIZE: tuple[int, int] = (112, 112)   # (W, H) fed to the model


class TemporalBuffer:
    """
    Maintains a fixed-length sliding window of person crops per track.

    Args:
        seq_len:    Number of frames to keep per track (rolling window).
        crop_size:  (W, H) to resize each person crop before storing.
    """

    def __init__(self, seq_len: int = 32, crop_size: tuple[int, int] = CROP_SIZE) -> None:
        self.seq_len   = seq_len
        self.crop_size = crop_size
        self._buffers: dict[int, deque[np.ndarray]] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def add_frame(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: list[float],
    ) -> None:
        """
        Crop the person region from *frame* using *bbox* and append to the buffer.

        Args:
            track_id: Unique integer ID for the tracked person.
            frame:    Full BGR frame (H, W, 3) from the camera.
            bbox:     [x1, y1, x2, y2] bounding box in absolute pixels.
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.seq_len)

        crop = self._safe_crop(frame, bbox)
        self._buffers[track_id].append(crop)

    def get_sequence(self, track_id: int) -> Optional[list[np.ndarray]]:
        """
        Return the full sliding-window sequence for a track, or None if
        fewer than *seq_len* frames have been accumulated.

        Returns:
            List of seq_len BGR crops, or None.
        """
        buf = self._buffers.get(track_id)
        if buf is None or len(buf) < self.seq_len:
            return None
        return list(buf)

    def cleanup(self, active_ids: set[int]) -> None:
        """Remove buffers for tracks that are no longer active."""
        dead = [tid for tid in self._buffers if tid not in active_ids]
        for tid in dead:
            del self._buffers[tid]

    def track_ids(self) -> list[int]:
        return list(self._buffers.keys())

    # ── Internal ─────────────────────────────────────────────────────────────

    def _safe_crop(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return np.zeros((self.crop_size[1], self.crop_size[0], 3), dtype=np.uint8)

        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, self.crop_size)
