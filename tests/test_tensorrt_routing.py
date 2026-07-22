"""
test_tensorrt_routing.py — Unit tests verifying model loader routing and auto-fallback behavior.
"""

import sys
import types
from pathlib import Path

# Dynamically alias the 'Eagle' namespace to the project root at runtime
if "Eagle" not in sys.modules:
    eagle_mod = types.ModuleType("Eagle")
    eagle_mod.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules["Eagle"] = eagle_mod

import pytest
from unittest.mock import MagicMock, patch
from services.detection.detection import Detector


@pytest.fixture
def mock_yolo():
    """Mocks the ultralytics YOLO class to prevent loading real model weights during tests."""
    with patch("services.detection.detection.YOLO") as mock:
        yield mock


def test_routing_pytorch(mock_yolo):
    """Verifies that .pt model paths correctly route to load_pytorch_model."""
    detector = Detector(model_name="yolov8n.pt", device="cpu")
    assert detector.model_path == "yolov8n.pt"
    mock_yolo.assert_called_with("yolov8n.pt", task="detect")
    

def test_routing_onnx(mock_yolo):
    """Verifies that .onnx model paths correctly route to load_onnx_model."""
    detector = Detector(model_name="yolov8n.onnx", device="cpu")
    assert detector.model_path == "yolov8n.onnx"
    mock_yolo.assert_called_with("yolov8n.onnx", task="detect")


def test_routing_engine_success(mock_yolo):
    """Verifies that .engine model paths route to load_tensorrt_model when device is CUDA."""
    with patch("services.detection.detection.Path.exists") as mock_exists:
        mock_exists.return_value = True
        detector = Detector(model_name="yolov8n.engine", device="cuda:0")
        assert detector.model_path == "yolov8n.engine"
        mock_yolo.assert_called_with("yolov8n.engine", task="detect")


def test_routing_engine_cpu_fallback(mock_yolo):
    """Verifies that .engine model path on CPU triggers auto-fallback to available formats."""
    # Define a plain method to bypass bound descriptor mock complexities
    def mock_exists(self_obj):
        return str(self_obj).endswith(".pt")
        
    with patch("services.detection.detection.Path.exists", mock_exists):
        detector = Detector(model_name="yolov8n.engine", device="cpu")
        # Should fallback to yolov8n.pt
        mock_yolo.assert_called_with("yolov8n.pt", task="detect")


def test_routing_engine_load_failure_fallback(mock_yolo):
    """Verifies that .engine loading failure on CUDA triggers automatic fallback to .pt."""
    def mock_exists(self_obj):
        return str(self_obj).endswith(".pt") or str(self_obj).endswith(".engine")
        
    with patch("services.detection.detection.Path.exists", mock_exists):
        # YOLO fails to load the engine file (simulating driver mismatch or corrupt engine)
        def side_effect(path, task=None):
            if path.endswith(".engine"):
                raise RuntimeError("Cuda driver mismatch")
            return MagicMock()
        mock_yolo.side_effect = side_effect
        
        detector = Detector(model_name="yolov8n.engine", device="cuda:0")
        # Should fallback to yolov8n.pt
        mock_yolo.assert_called_with("yolov8n.pt", task="detect")

