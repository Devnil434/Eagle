"""
Webcam / RTSP demo — detection (YOLO) + tracking + temporal actions (ONNX).

Usage:
    python gssoc/temporal_action_recognition/demos/demo_cctv.py
    python gssoc/temporal_action_recognition/demos/demo_cctv.py --source rtsp://user:pass@host/stream
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
  source = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else "0"
  cmd = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--source", source, "--camera-id", "cam_01"]
  raise SystemExit(subprocess.call(cmd))
