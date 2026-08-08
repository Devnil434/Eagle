# Benchmark Report: Drone-Optimized YOLOv8n

## 📊 Performance Results
- **Baseline (FP32):** 18.50 FPS
- **Optimized (INT8):** 42.92 FPS
- **Speed-up Factor:** 2.32x 🚀

## 🛠️ Optimization Strategy
- **Quantization:** OpenVINO INT8
- **Resolution:** 320x320 (Drone-optimized)
- **Calibration:** Specialized drone dataset via Roboflow (`drone-detection-lzvig-sa0py`)