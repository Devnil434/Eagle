# Install this package into Eagle repo

## 1. Copy folders into your Eagle clone

Merge these paths into the repo root (overwrite if asked):

```text
services/action_recognition/   →  services/action_recognition/
libs/schemas/action_recognition.py
services/memory/action_bridge.py
services/memory/pipeline.py
scripts/run_pipeline.py
tests/test_action_recognition.py
apps/backend/main.py
apps/backend/routes/cameras.py
apps/dashboard/src/pages/Dashboard.jsx
apps/dashboard/src/components/CameraCard.jsx
```

## 2. Model file (ONNX)

Copy your trained model to the **repo root** (or use `weights/`):

```text
action_model.onnx
```

Set in `.env`:

```bash
ACTION_MODEL_PATH=action_model.onnx
```

## 3. Dependencies

```bash
pip install -r services/action_recognition/requirements.txt
```

## 4. Run

```bash
docker compose up redis
uvicorn apps.backend.main:app --reload --port 8000
python scripts/run_pipeline.py --source 0 --camera-id cam_01
cd apps/dashboard && npm run dev
```

## 5. Demos

```bash
python demos/demo_cctv.py
python demos/demo_video.py --source your_video.mp4
```
