# Eagle Architecture Reference

Eagle is an event-driven surveillance reasoning pipeline that converts raw video frames into natural-language risk assessments. Frames enter the detection layer (`services/detection/detector.py`), tracked entities are persisted across time (`services/tracking/tracker.py`), recent events are stored in Redis (`services/memory/memory.py`), and only meaningful behavioral changes trigger multimodal reasoning (`services/reasoning/vlm.py` + `services/reasoning/llm.py`). The final output is a structured alert served through the FastAPI backend (`apps/backend/main.py`) and visualized in the React dashboard (`apps/dashboard/`).

---

## Component Overview

| Service | Tech | Input Schema | Output Schema |
|---|---|---|---|
| Detection | YOLOv8/v9 | `FrameInput(frame, camera_id)` | `Detection(track_boxes, classes, confidence)` |
| Tracking | ByteTrack / DeepSORT | Detection results | `TrackedObject(track_id, trajectory, dwell_time)` |
| Temporal Memory | Redis Ring Buffer | `track_id + event payload` | Sliding event history (`last_n_events`) |
| VLM Captioning | LLaVA-Next / Qwen-VL | Triggered frame sequence | Natural language captions |
| LLM Reasoning | Mixtral / GPT-4o / Gemini | Caption sequence + policies | `Alert(label, confidence, reason)` |
| Incident Reporting | Jinja2 + fpdf2 | Stored alerts for a time window | `IncidentSummary` as Markdown / JSON / PDF |
| Backend API | FastAPI + Celery | REST requests | JSON API responses |
| Frontend | React 19 + Vite | SSE / REST payloads | Live dashboard + alert timeline |

---

## Data Flow

```mermaid
flowchart TD

A[Camera Stream / Video File]
--> B[Detection Service<br/>services/detection/detector.py]

B --> C[Tracking Service<br/>services/tracking/tracker.py]

C --> D[Temporal Memory<br/>services/memory/memory.py]

D --> E{Event Trigger}

E -->|Zone Entry / Dwell / Interaction| F[VLM Captioning<br/>services/reasoning/vlm.py]

F --> G[LLM Reasoning<br/>services/reasoning/llm.py]

G --> H[FastAPI Backend<br/>apps/backend/main.py]

H --> I[React Dashboard<br/>apps/dashboard]

H --> J[Incident Reporting<br/>services/reporting]

J -->|Markdown / JSON / PDF| K[Operator Review]
