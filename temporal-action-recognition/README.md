# GSSoC 2026 — Temporal Action Recognition

This folder is the **review entry point** for the temporal action recognition feature.
Production code lives in the main repo paths below (not duplicated here).

## What this feature does

- Maintains per-track rolling frame buffers (`TemporalBuffer`)
- Classifies activities with **MobileNetV2 + LSTM**, exported as **`action_model.onnx`**
- Falls back to kinematic heuristics when ONNX is missing
- Publishes labels to Redis for the dashboard API and memory/reasoning events

## Repo layout (PR scope)

| Path | Purpose |
|------|---------|
| `services/action_recognition/` | Model, inference, temporal buffer, utils |
| `libs/schemas/action_recognition.py` | Pydantic schemas + alert severity |
| `services/memory/action_bridge.py` | Redis + memory event integration |
| `services/memory/pipeline.py` | Wired into Phase 3 pipeline |
| `scripts/run_pipeline.py` | Live/video pipeline with actions |
| `apps/backend/routes/cameras.py` | `current_action` on track API |
| `apps/dashboard/` | Displays actions from API |

## Model file (required for neural inference)

Place your trained ONNX file at the **repository root**:

```text
action_model.onnx
```

Or set:

```bash
export ACTION_MODEL_PATH=/path/to/action_model.onnx
```

**Note:** Object detection still uses YOLO (`yolov8n.pt`).  
`action_model.onnx` is only for **temporal action classification** on person crops.

## Quick start

```bash
# Install action recognition extras
pip install -r services/action_recognition/requirements.txt

# Terminal 1 — Redis + API
docker compose up redis backend

# Terminal 2 — CCTV / webcam pipeline (writes actions to Redis)
python scripts/run_pipeline.py --source 0 --camera-id cam_01

# Terminal 3 — Dashboard
cd apps/dashboard && npm install && npm run dev
```

## Demos in this folder

- `demos/demo_cctv.py` — webcam / RTSP stream
- `demos/demo_video.py` — file-based video

## Tests

```bash
pytest tests/test_action_recognition.py -q
pytest tests/integration/test_pipeline.py -q
```

## Action classes

`walking`, `running`, `fighting`, `loitering`, `falling`, `suspicious_stationary`

Alertable: fighting, running, loitering, falling, suspicious_stationary.
