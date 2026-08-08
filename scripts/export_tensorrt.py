"""
export_tensorrt.py — CLI tool to compile YOLO models to high-performance TensorRT (.engine) format.

This script manages hardware validation (CUDA, GPU capability) and uses the Ultralytics 
export engine wrapper to convert standard PyTorch (.pt) or ONNX (.onnx) files into 
accelerated TensorRT engines tailored specifically to the host GPU.

Usage:
    python scripts/export_tensorrt.py --model yolov8n.pt --fp16
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    logger.error("Required libraries (torch, ultralytics) are missing. Please run: pip install ultralytics torch")
    sys.exit(1)


def export_model(model_path: str, fp16: bool, int8: bool, imgsz: int, device: str) -> None:
    """
    Compiles a PyTorch (.pt) or ONNX (.onnx) model into a TensorRT (.engine) model.
    """
    logger.info("Checking environment status for TensorRT export...")
    
    # 1. Hardware verification
    if "cpu" in device.lower():
        logger.error("TensorRT compilation is NOT supported on CPU. Please specify a CUDA device (e.g., --device 0 or cuda).")
        sys.exit(1)
        
    if not torch.cuda.is_available():
        logger.error("CUDA is not available on this machine. TensorRT requires an NVIDIA GPU with CUDA drivers.")
        sys.exit(1)
        
    # Check if GPU device is valid
    device_id = 0
    if ":" in device:
        device_id = int(device.split(":")[1])
    try:
        device_name = torch.cuda.get_device_name(device_id)
        logger.info(f"Using NVIDIA GPU: {device_name} (Device ID: {device_id})")
    except Exception as e:
        logger.error(f"Invalid CUDA device specified: {device}. Error: {e}")
        sys.exit(1)

    # 2. File verification
    model_file = Path(model_path)
    if not model_file.exists():
        logger.error(f"Source model file '{model_path}' not found!")
        sys.exit(1)
        
    if not (model_path.endswith(".pt") or model_path.endswith(".onnx")):
        logger.error("Unsupported source format. Model must end with '.pt' or '.onnx'")
        sys.exit(1)

    logger.info(f"Loading source model: {model_path}...")
    model = YOLO(model_path)

    logger.info("Starting compilation to TensorRT (.engine) format...")
    logger.info(f"Configuration: FP16={fp16}, INT8={int8}, Image Size={imgsz}, Target Device={device}")

    try:
        # Ultralytics natively wraps the ONNX -> TensorRT conversion process
        exported_path = model.export(
            format="engine",
            half=fp16,
            int8=int8,
            imgsz=imgsz,
            device=device,
            dynamic=True  # Enables dynamic batching support
        )
        logger.info("========================================= SUCCESS =========================================")
        logger.info("TensorRT Engine compiled and optimized successfully!")
        logger.info(f"Saved optimized model to: {os.path.abspath(exported_path)}")
        logger.info("===========================================================================================")
        
    except Exception as e:
        logger.error(f"An error occurred during TensorRT compilation: {e}")
        logger.error(
            "Please ensure you have the TensorRT Python API and CUDA toolkit properly installed. "
            "Refer to docs/tensorrt_conversion.md for assistance."
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compile YOLOv8/v9 models (.pt/.onnx) to highly-optimized TensorRT (.engine) format."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to the source model (.pt or .onnx file) to compile."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 (half-precision) float operations for faster inference (recommended)."
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        default=False,
        help="Enable INT8 quantization (requires calibrating dataset)."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Standard resolution (width/height) of input frames (default: 640)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to use for compilation (default: cuda:0)."
    )
    
    args = parser.parse_args()
    export_model(
        model_path=args.model,
        fp16=args.fp16,
        int8=args.int8,
        imgsz=args.imgsz,
        device=args.device
    )


if __name__ == "__main__":
    main()
