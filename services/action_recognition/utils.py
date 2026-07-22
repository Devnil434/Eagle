"""
services.action_recognition.utils
====================================
Preprocessing helpers, ImageNet normalisation, trajectory kinematics,
and alert formatting utilities.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from libs.schemas.action_recognition import ActionLabel, AlertSeverity
    from libs.schemas.tracking import TrackedObject

# ImageNet statistics (RGB order)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Frame preprocessing ───────────────────────────────────────────────────────

def preprocess_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Convert a BGR uint8 crop to a float32 CHW tensor normalised with
    ImageNet mean/std. Returns shape (3, H, W).
    """
    import cv2  # local import so the module is importable without cv2 in tests

    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return normalised.transpose(2, 0, 1)   # (H, W, 3) → (3, H, W)


def frames_to_tensor(frames: list[np.ndarray]):
    """
    Stack a sequence of BGR crops into a float32 tensor of shape
    (1, T, 3, H, W) ready for the model.

    Requires PyTorch.
    """
    import torch

    processed = [preprocess_frame(f) for f in frames]
    stacked = np.stack(processed, axis=0)        # (T, 3, H, W)
    tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, T, 3, H, W)
    return tensor


# ── Trajectory kinematics ─────────────────────────────────────────────────────

def compute_trajectory_speed(obj: "TrackedObject") -> float:
    """
    Return the mean per-frame pixel speed over the last N trajectory points.
    Returns 0.0 if fewer than 2 points are available.
    """
    traj = obj.trajectory
    if len(traj) < 2:
        return 0.0

    deltas = [
        np.hypot(traj[i].x - traj[i - 1].x, traj[i].y - traj[i - 1].y)
        for i in range(1, len(traj))
    ]
    return float(np.mean(deltas))


def compute_vertical_displacement(obj: "TrackedObject") -> float:
    """
    Signed vertical displacement from first to last trajectory point.
    Positive = downward (increasing y in image coordinates).
    """
    traj = obj.trajectory
    if len(traj) < 2:
        return 0.0
    return traj[-1].y - traj[0].y


# ── Alert formatting ──────────────────────────────────────────────────────────

def format_alert_message(track_id: int, action: "ActionLabel") -> str:
    templates = {
        "fighting":              f"⚠ ALERT — Track #{track_id}: Fighting detected!",
        "running":               f"⚠ ALERT — Track #{track_id}: Person running.",
        "loitering":             f"⚠ ALERT — Track #{track_id}: Loitering detected.",
        "falling":               f"⚠ ALERT — Track #{track_id}: Person may have fallen!",
        "suspicious_stationary": f"⚠ ALERT — Track #{track_id}: Suspicious stationary behaviour.",
    }
    return templates.get(action.value, f"⚠ ALERT — Track #{track_id}: {action.value}")


def get_severity(action: "ActionLabel") -> "AlertSeverity":
    from libs.schemas.action_recognition import SEVERITY_MAP
    return SEVERITY_MAP[action]
