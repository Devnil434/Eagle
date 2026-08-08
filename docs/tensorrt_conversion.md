# TensorRT Compilation & Optimization Guide

This guide covers installing dependencies, converting models into high-performance TensorRT `.engine` formats, and running optimized inference using Eagle’s smart automatic fallback protocol.

---

## 🚀 Why TensorRT?

NVIDIA TensorRT is a high-performance deep learning inference library that optimizes neural network models for deployment on NVIDIA GPUs and Jetson hardware. Utilizing `.engine` formats provides:

* **Up to 5x Faster Inference**: Highly optimized CUDA kernels tailored directly to your GPU.
* **Low Latency & High FPS**: Crucial for real-time surveillance and anomaly detection.
* **FP16 Half-Precision Optimization**: Reduces memory footprint and doubles processing speed with negligible accuracy loss.
* **Dynamic Batching & Memory Efficiency**: Saves critical GPU memory (VRAM) bounds.

---

## 🛠️ 1. Installation & Setup

To compile and execute `.engine` models, your host machine requires the CUDA Toolkit, cuDNN, TensorRT, and PyCUDA python APIs.

### Step A: Install NVIDIA Drivers & CUDA Toolkit
1. Download and install compatible **NVIDIA GPU Drivers** from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx).
2. Download and install **CUDA Toolkit 11.8 or 12.x** from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
3. Ensure CUDA is added to your environment `PATH` variables. Verify by running:
   ```bash
   nvcc --version
   ```

### Step B: Install cuDNN
1. Download **cuDNN** (matching your CUDA version) from the [cuDNN Download Portal](https://developer.nvidia.com/cudnn).
2. Copy cuDNN headers and libraries into your local CUDA Toolkit directory.

### Step C: Install TensorRT
1. Download **NVIDIA TensorRT** matching your CUDA version from [TensorRT Portal](https://developer.nvidia.com/tensorrt).
2. Follow the installation guide to unzip and add TensorRT binaries to your system library path.
3. Install the TensorRT Python wheel matching your Python version (found in the `python/` directory of the TensorRT package):
   ```bash
   pip install tensorrt
   ```

### Step D: Install PyCUDA
PyCUDA is required for low-level memory copies (DMA transfer coordination) on NVIDIA GPUs.
```bash
pip install pycuda
```

---

## 📦 2. Model Conversion Using `export_tensorrt.py`

We have provided a streamlined conversion script in `scripts/export_tensorrt.py` to automate the compilation of `.pt` or `.onnx` models into `.engine` format.

### Basic Compilation (Recommended FP16)
To compile a PyTorch YOLOv8 baseline model using optimized **FP16 half-precision**, run:
```bash
python scripts/export_tensorrt.py --model yolov8n.pt --fp16
```
This automatically compiles the model and saves a newly optimized `yolov8n.engine` file in the same directory!

### Command Parameters
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | `yolov8n.pt` | Path to the source `.pt` or `.onnx` model file to compile. |
| `--fp16` | `bool` | `True` | Enables FP16 half-precision optimization (highly recommended). |
| `--int8` | `bool` | `False` | Enables INT8 quantization (requires calibrating dataset). |
| `--imgsz` | `int` | `640` | Resolution width/height of input frames (default: 640). |
| `--device` | `str` | `cuda:0` | GPU device ID to execute compiling (default: `cuda:0`). |

---

## 🧠 3. Smart Automatic Fallback Execution

You **do not need** to modify your application logic or worry about crashing on non-GPU/non-TensorRT machines. The system implements a **smart fallback routing layer**:

1. **Auto-Search**: The `Detector` class checks if a matching `.engine` file exists in the directory of your configured model (e.g. if `yolov8n.pt` is requested, it looks for `yolov8n.engine`).
2. **Auto-Promote**: If the `.engine` model is present and CUDA/TensorRT drivers are available, the detector automatically loads the optimized TensorRT engine for accelerated performance.
3. **Resilient Fallback**: If the `.engine` file is missing, corrupted, compiled on a different GPU, or if TensorRT is not supported on the host system, the code:
   - Prints a non-blocking warning log: `Failed to load TensorRT engine. Triggering automatic fallback...`
   - Automatically loads the baseline `.pt` or `.onnx` file and continues normal execution without interruption.

---

## 📊 4. Running the Performance Benchmarks

To measure the latency and FPS throughput improvements, we have upgraded `benchmark.py` to test and compare multiple formats.

### Run Multi-Format Comparative Benchmark:
```bash
python benchmark.py --compare
```

This runs a simulated video pipeline processing frames across `.pt`, `.onnx`, and `.engine` files, and generates a unified report under:
`docs/benchmarks/comparison_report.md`

### Benchmark a Single Specific Model:
```bash
python benchmark.py --model yolov8n.engine
```
Report generated under:
`docs/benchmarks/pipeline_benchmark.md`
