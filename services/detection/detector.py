"""Compatibility shim: re-export Detector from detection.py.

Some code/tests import `services.detection.detector.Detector` — keep a
small shim for backwards compatibility.
"""
from .detection import Detector

__all__ = ["Detector"]
