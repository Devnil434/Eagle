"""
services.action_recognition.inference
=======================================
Main inference engine for temporal action recognition.

Provides two classifiers:
  - HeuristicClassifier: rule-based fallback using trajectory kinematics.
                         Works out-of-the-box with no training required.
  - ActionRecognizer:    orchestrates the temporal buffer, heuristic,
                         and (optionally) the neural model, and generates
                         ActionAlert objects for suspicious activities.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from libs.schemas.action_recognition import (
    ActionLabel,
    ActionPrediction,
    ActionAlert,
    ActionFrameResult,
    ALERTABLE_ACTIONS,
    SEVERITY_MAP,
)
from libs.schemas.tracking import TrackedFrame, TrackedObject
from services.action_recognition.temporal_buffer import TemporalBuffer
from services.action_recognition.utils import (
    compute_trajectory_speed,
    compute_vertical_displacement,
    format_alert_message,
    frames_to_tensor,
)

logger = logging.getLogger(__name__)

# ── Heuristic thresholds (tune per camera / pixel scale) ─────────────────────
SPEED_RUNNING  = 15.0   # px/frame mean speed → running
SPEED_LOITER   = 3.0    # px/frame mean speed below this → possible loitering
DWELL_LOITER   = 10.0   # seconds dwell before loitering is declared
DWELL_STATIONARY = 5.0  # seconds stationary before suspicious_stationary
FALL_DISPLACEMENT = 40  # px sudden downward displacement → falling
FIGHT_PROXIMITY   = 100 # px between two persons → potential fight zone
FIGHT_MIN_SPEED   = 8.0 # px/frame — both persons must be moving fast


# ═════════════════════════════════════════════════════════════════════════════
# Heuristic Classifier
# ═════════════════════════════════════════════════════════════════════════════

class HeuristicClassifier:
    """
    Rule-based action classifier using trajectory kinematics.

    No training required. Works immediately using data already produced
    by the existing Tracker (trajectory, dwell_time_seconds).
    """

    def classify(self, obj: TrackedObject) -> ActionPrediction:
        speed      = compute_trajectory_speed(obj)
        vert_disp  = compute_vertical_displacement(obj)
        dwell      = obj.dwell_time_seconds

        if vert_disp > FALL_DISPLACEMENT and speed > SPEED_LOITER:
            action = ActionLabel.FALLING
            conf   = 0.80

        elif speed >= SPEED_RUNNING:
            action = ActionLabel.RUNNING
            conf   = min(0.95, 0.65 + (speed - SPEED_RUNNING) / 30.0)

        elif speed < SPEED_LOITER and dwell >= DWELL_LOITER:
            action = ActionLabel.LOITERING
            conf   = min(0.90, 0.60 + (dwell - DWELL_LOITER) / 30.0)

        elif speed < 0.5 and dwell >= DWELL_STATIONARY:
            action = ActionLabel.SUSPICIOUS_STATIONARY
            conf   = 0.70

        else:
            action = ActionLabel.WALKING
            conf   = 0.75

        return ActionPrediction(
            track_id=obj.track_id,
            action=action,
            confidence=round(conf, 3),
            source="heuristic",
        )

    def detect_fighting(self, objects: list[TrackedObject]) -> list[tuple[int, int]]:
        """
        Return pairs of (track_id_a, track_id_b) that appear to be fighting.
        Both must be moving fast and within FIGHT_PROXIMITY pixels.
        """
        fast = [o for o in objects if compute_trajectory_speed(o) >= FIGHT_MIN_SPEED]
        pairs: list[tuple[int, int]] = []
        for i, a in enumerate(fast):
            for b in fast[i + 1:]:
                dist = np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
                if dist < FIGHT_PROXIMITY:
                    pairs.append((a.track_id, b.track_id))
        return pairs


# ═════════════════════════════════════════════════════════════════════════════
# Main ActionRecognizer
# ═════════════════════════════════════════════════════════════════════════════

class ActionRecognizer:
    """
    Orchestrates temporal buffering, heuristic classification, and optional
    neural-model inference to produce per-frame action predictions and alerts.

    Args:
        seq_len:       Number of frames per temporal sequence.
        model_path:    Optional path to trained .pt weights.
                       If None or file missing, heuristic mode is used.
        infer_every_n: Run neural inference every N frames (performance knob).
    """

    # Default search paths — ONNX is preferred for real-time inference
    _DEFAULT_MODEL_PATHS = [
        Path(__file__).resolve().parents[2] / "action_model.onnx",
        Path(__file__).resolve().parents[2] / "weights" / "action_model.onnx",
        Path(__file__).resolve().parents[2] / "apps" / "dashboard" / "action_model.onnx",
        Path(__file__).resolve().parents[2] / "action_model.pt",
        Path(__file__).resolve().parents[2] / "weights" / "action_model.pt",
    ]

    def __init__(
        self,
        seq_len:       int              = 32,  # matches feature spec and TemporalBuffer default
        model_path:    Optional[str]    = None,
        infer_every_n: int              = 8,
    ) -> None:
        self._buffer     = TemporalBuffer(seq_len=seq_len)
        self._heuristic  = HeuristicClassifier()
        self._model      = None
        self._frame_ctr  = 0
        self._infer_n    = infer_every_n

        # ACTION_MODEL_PATH env overrides; otherwise auto-detect (ONNX first)
        resolved_path = None
        env_path = os.getenv("ACTION_MODEL_PATH", "").strip()
        if env_path and Path(env_path).is_file():
            resolved_path = env_path
        elif model_path and Path(model_path).is_file():
            resolved_path = model_path
        else:
            for default in self._DEFAULT_MODEL_PATHS:
                if default.is_file():
                    resolved_path = str(default)
                    break

        if resolved_path:
            self._load_model(resolved_path)
        else:
            logger.info(
                "No action_model.onnx found — running in heuristic-only mode. "
                "Place action_model.onnx at the repo root or set ACTION_MODEL_PATH."
            )

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        tracked_frame: TrackedFrame,
        raw_frame: np.ndarray,
    ) -> ActionFrameResult:
        """
        Process one frame of tracking results.

        Args:
            tracked_frame: Output of Tracker.update().
            raw_frame:     The original BGR frame from the camera.

        Returns:
            ActionFrameResult with predictions and any triggered alerts.
        """
        self._frame_ctr += 1
        tracks = tracked_frame.tracks
        active_ids = {t.track_id for t in tracks}

        predictions: list[ActionPrediction] = []
        fight_overrides: set[int] = set()

        # ── 1. Feed buffer ────────────────────────────────────────────────
        for obj in tracks:
            self._buffer.add_frame(obj.track_id, raw_frame, obj.bbox)

        # ── 2. Fighting detection (heuristic, multi-track) ────────────────
        fight_pairs = self._heuristic.detect_fighting(tracks)
        for tid_a, tid_b in fight_pairs:
            fight_overrides.add(tid_a)
            fight_overrides.add(tid_b)

        # ── 3. Per-track classification ───────────────────────────────────
        for obj in tracks:
            if obj.track_id in fight_overrides:
                pred = ActionPrediction(
                    track_id=obj.track_id,
                    action=ActionLabel.FIGHTING,
                    confidence=0.85,
                    source="heuristic",
                )
            elif self._model is not None and self._frame_ctr % self._infer_n == 0:
                pred = self._neural_predict(obj)
            else:
                pred = self._heuristic.classify(obj)

            predictions.append(pred)

        # ── 4. Generate alerts ────────────────────────────────────────────
        alerts = self._make_alerts(predictions)

        # ── 5. Cleanup dead tracks ────────────────────────────────────────
        self._buffer.cleanup(active_ids)

        return ActionFrameResult(
            frame_id=tracked_frame.frame_id,
            predictions=predictions,
            alerts=alerts,
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _neural_predict(self, obj: TrackedObject) -> ActionPrediction:
        seq = self._buffer.get_sequence(obj.track_id)
        if seq is None:
            return self._heuristic.classify(obj)
        try:
            tensor = frames_to_tensor(seq)
            action_str, conf = self._model.predict(tensor)
            return ActionPrediction(
                track_id=obj.track_id,
                action=ActionLabel(action_str),
                confidence=round(conf, 3),
                source="model",
            )
        except Exception as exc:
            logger.warning("Neural inference failed for track %d: %s", obj.track_id, exc)
            return self._heuristic.classify(obj)

    def _make_alerts(self, predictions: list[ActionPrediction]) -> list[ActionAlert]:
        alerts = []
        for pred in predictions:
            if pred.action in ALERTABLE_ACTIONS:
                alerts.append(ActionAlert(
                    track_id=pred.track_id,
                    action=pred.action,
                    severity=SEVERITY_MAP[pred.action],
                    message=format_alert_message(pred.track_id, pred.action),
                ))
        return alerts

    def _load_model(self, path: str) -> None:
        try:
            if path.endswith(".onnx"):
                from services.action_recognition.model import ONNXActionRecognitionModel
                self._model = ONNXActionRecognitionModel(path)
                logger.info("ONNX Action recognition model loaded from %s", path)
            else:
                import torch
                from services.action_recognition.model import ActionRecognitionModel
                model = ActionRecognitionModel()
                # Suppress security warnings for weights_only
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                model.eval()
                self._model = model
                logger.info("PyTorch Action recognition model loaded from %s", path)
        except Exception as exc:
            logger.warning("Could not load model from %s: %s — using heuristic.", path, exc)
