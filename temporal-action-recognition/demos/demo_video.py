"""
Video file demo — same pipeline as live CCTV.

Usage:
    python gssoc/temporal_action_recognition/demos/demo_video.py --source path/to/video.mp4
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
  if "--source" not in sys.argv:
    raise SystemExit("Usage: demo_video.py --source <video.mp4>")
  source = sys.argv[sys.argv.index("--source") + 1]
  cmd = [
    sys.executable,
    str(ROOT / "scripts" / "run_pipeline.py"),
    "--source", source,
    "--camera-id", "cam_01",
  ]
  raise SystemExit(subprocess.call(cmd))
