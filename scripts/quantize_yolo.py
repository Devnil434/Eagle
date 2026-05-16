import os
from ultralytics import YOLO

def drone_optimized_quantization():
    """
    Automates the INT8 quantization pipeline for YOLOv8n models targeted at drone telemetry.
    Reads secure credentials dynamically from system environment variables.
    """
    # AI FIX: Moving raw credentials to environment properties
    API_KEY = os.getenv("ROBOFLOW_API_KEY")
    WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "goyalpreeti")
    PROJECT = os.getenv("ROBOFLOW_PROJECT", "drone-detection-lzvig-sa0py")
    VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))
    
    if not API_KEY:
        print("⚠️ ROBOFLOW_API_KEY environment variable missing! Programmatic dataset pull skipped.")
        drone_data_yaml = "Drone-detection-1/data.yaml"
    else:
        from roboflow import Roboflow
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        dataset = project.version(VERSION).download("yolov8")
        drone_data_yaml = f"{dataset.location}/data.yaml"

    print("⚡ Initiating OpenVINO INT8 Calibration Export process...")
    model = YOLO("yolov8n.pt")
    
    path = model.export(
        format="openvino", 
        int8=True, 
        data=drone_data_yaml, 
        imgsz=320
    )
    
    # AI FIX: Using the returned true absolute path variable instead of a hardcoded string literal
    print(f"\n✅ SUCCESS: Drone-Optimized model saved to: {os.path.abspath(str(path))}")

if __name__ == "__main__":
    drone_optimized_quantization()