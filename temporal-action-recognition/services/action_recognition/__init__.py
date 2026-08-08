"""
services.action_recognition — Temporal action recognition for surveillance.

Provides track-wise temporal buffering, CNN+LSTM action classification,
heuristic fallback, and alert generation for suspicious activities.
"""

from services.action_recognition.temporal_buffer import TemporalBuffer
from services.action_recognition.inference import ActionRecognizer, HeuristicClassifier

__all__ = [
    "TemporalBuffer",
    "ActionRecognizer",
    "HeuristicClassifier",
]
