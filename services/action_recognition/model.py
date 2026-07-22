"""
services.action_recognition.model
====================================
MobileNetV2 + 2-layer LSTM temporal action recognition model.

Architecture:
    frame → MobileNetV2 backbone → (B, T, 1280) features
          → LSTM (512 hidden, 2 layers)
          → Linear classifier → num_classes logits

The backbone is frozen by default; only the LSTM head is trained.
Requires: torch, torchvision
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torchvision import models

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not found. Neural model is unavailable; "
        "heuristic fallback will be used."
    )


# 6 trained classes (the model was trained without "unknown")
TRAINED_CLASSES = [
    "walking",
    "running",
    "fighting",
    "loitering",
    "falling",
    "suspicious_stationary",
]

# Full list including fallback
ACTION_CLASSES = TRAINED_CLASSES + ["unknown"]

NUM_TRAINED_CLASSES = len(TRAINED_CLASSES)  # 6
NUM_CLASSES = len(ACTION_CLASSES)           # 7


class ActionRecognitionModel(nn.Module if TORCH_AVAILABLE else object):
    """
    MobileNetV2 backbone + LSTM classifier for temporal action recognition.

    Architecture matches the Kaggle training script:
        - self.backbone = mobilenet_v2().features  (NOT wrapped in Sequential)
        - self.pool     = AdaptiveAvgPool2d
        - self.flatten  = Flatten
        - self.lstm     = 2-layer LSTM
        - self.dropout  = Dropout
        - self.classifier = Linear(512, num_classes)

    Args:
        num_classes:     Number of action classes (default: 6, matching trained weights).
        lstm_hidden:     LSTM hidden size.
        lstm_layers:     Number of stacked LSTM layers.
        dropout:         Dropout probability.
        freeze_backbone: If True, backbone weights are frozen during training.
    """

    def __init__(
        self,
        num_classes:     int   = NUM_TRAINED_CLASSES,
        lstm_hidden:     int   = 512,
        lstm_layers:     int   = 2,
        dropout:         float = 0.3,
        freeze_backbone: bool  = True,
        use_mock:        bool  = False,
    ) -> None:
        if not TORCH_AVAILABLE:
            logger.warning("Torch unavailable — model is a no-op stub.")
            return

        super().__init__()

        # Backbone: MobileNetV2 features (NOT wrapped in Sequential)
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.feature_dim = 1280

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Temporal head
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):  # x: (B, T, 3, H, W)
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for neural inference.")
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        features = self.flatten(self.pool(self.backbone(x_flat)))  # (B*T, 1280)
        features = features.view(B, T, self.feature_dim)           # (B, T, 1280)
        _, (h_n, _) = self.lstm(features)
        final_hidden = h_n[-1]                                     # (B, hidden)
        out = self.dropout(final_hidden)
        return self.classifier(out)                                # (B, num_classes)

    def predict(self, tensor) -> tuple[str, float]:
        """Run a single (1, T, 3, H, W) tensor through the model."""
        if not TORCH_AVAILABLE:
            return "unknown", 0.0
        import torch
        self.eval()
        with torch.no_grad():
            logits = self.forward(tensor)
            probs = torch.softmax(logits, dim=-1)
            idx = int(probs.argmax(dim=-1).item())
            conf = float(probs[0, idx].item())
        # Map to trained class name (6 classes)
        if idx < len(TRAINED_CLASSES):
            return TRAINED_CLASSES[idx], conf
        return "unknown", conf


class ONNXActionRecognitionModel:
    """Wrapper for ONNX runtime inference."""

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
            # Provider priority: CUDA (NVIDIA GPU) → OpenVINO (Intel) → CPU.
            # onnxruntime silently skips any provider that is not installed,
            # so listing all three is safe on CPU-only machines.
            providers = [
                'CUDAExecutionProvider',
                'OpenVINOExecutionProvider',
                'CPUExecutionProvider',
            ]
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        except ImportError:
            raise RuntimeError("onnxruntime is required for ONNX inference.")

    def predict(self, tensor) -> tuple[str, float]:
        """Run inference using ONNX Runtime. tensor is (1, T, 3, H, W) PyTorch tensor or numpy array."""
        try:
            import numpy as np
            if hasattr(tensor, "numpy"):
                tensor = tensor.numpy()
            
            # Ensure it's float32
            tensor = tensor.astype(np.float32)
            
            outputs = self.session.run(None, {self.input_name: tensor})
            logits = outputs[0]  # (1, num_classes)
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            idx = int(np.argmax(probs[0]))
            conf = float(probs[0, idx])
            
            if idx < len(TRAINED_CLASSES):
                return TRAINED_CLASSES[idx], conf
            return "unknown", conf
        except Exception as exc:
            logger.warning("ONNX inference failed: %s", exc)
            return "unknown", 0.0
