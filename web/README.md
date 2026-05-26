# TrafficAI Realtime Vehicle Detection

TrafficAI is a realtime traffic analytics web system built on YOLOv8 with a FastAPI backend
and a React + Vite frontend. The backend runs inference, renders annotated frames, streams
MJPEG previews, and tracks analytics. The frontend provides image and video workflows with
live metrics and charts.

## Project Structure

```
code/web/
	backend/                 FastAPI + YOLOv8 inference
		app/
			api/                 API route handlers
			core/                config + logging
			services/            inference, ROI, analytics, video jobs
		models/                YOLO weights (best.pt)
		storage/               uploaded + processed video artifacts
	frontend/                React + Vite + Tailwind UI
		src/
			api/                 API client helpers
			components/          shared UI widgets
			pages/               Image + Video modes
	docs/                    API documentation
```

## Core Features

- Image detection with bounding boxes, confidence, and export
- Video upload, background processing, and result download
- MJPEG realtime annotated stream with FPS throttling
- ByteTrack multi-object tracking via boxmot (video + stream)
- Adaptive ROI (optional) to focus inference on motion regions
- Live analytics: FPS, average objects, and time-series charts

## Supported File Types

Images: JPG, PNG, WEBP

Videos: MP4, MOV, AVI, MKV, WEBM

## Backend Setup

```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Place your model at:

```
backend/models/best.pt
```

Or configure path:

```
set APP_MODEL_PATH=path\\to\\best.pt
```

Run API:

```bash
APP_ROI_DRAW_ALPHA=0.25
```

Run API (force GPU):

```bash
set APP_DEVICE=cuda:0
set APP_USE_HALF=true
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open docs:

- http://localhost:8000/docs

## ByteTrack Notes

- Tracking is applied in offline video processing and MJPEG streaming.
- Metrics in `/api/video/{job_id}` are based on tracked objects (not raw detections).
- Each tracked box is rendered with a `track_id` label when labels are enabled.

Config (environment variables):

```
APP_BYTETRACK_ENABLED=true
APP_BYTETRACK_CONF_HIGH=0.5
APP_BYTETRACK_IOU_HIGH=0.3
APP_BYTETRACK_TRACK_BUFFER=30
APP_BYTETRACK_MIN_BOX_AREA=400
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Optional environment:

```
VITE_API_BASE=http://localhost:8000
```

## CUDA GPU Setup

1. Install NVIDIA driver and CUDA Toolkit.
2. Install PyTorch with CUDA support (example for CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Verify in Python:

```python
import torch
print(torch.cuda.is_available())
```

4. Force GPU usage (optional):

```bash
set APP_DEVICE=cuda:0
set APP_USE_HALF=true
```


## Realtime Notes

- Upload video realtime preview uses MJPEG stream.
- Frame skip for offline processing is configurable via `APP_FRAME_SKIP`.
- JPEG quality via `APP_JPEG_QUALITY`.
- Stream resize for realtime preview via `APP_STREAM_MAX_DIM` (set 0 to disable).

## Key Modules and Functions

Backend

- Inference: `InferenceEngine.detect()` in [code/web/backend/app/services/inference.py](code/web/backend/app/services/inference.py)
- Video jobs: `VideoJobStore.create()` and `process_video()` in [code/web/backend/app/services/video_jobs.py](code/web/backend/app/services/video_jobs.py)
- Adaptive ROI: `compute_adaptive_roi()` and `ROIUpdater` in [code/web/backend/app/services/adaptive_roi.py](code/web/backend/app/services/adaptive_roi.py)
- Rendering: `render_boxes()` in [code/web/backend/app/services/renderer.py](code/web/backend/app/services/renderer.py)
- Streaming: `/api/video/{job_id}/stream` handler in [code/web/backend/app/api/routes.py](code/web/backend/app/api/routes.py)

Frontend

- API client: `detectImage()`, `uploadVideo()`, `getVideoStatus()` in [code/web/frontend/src/api/client.js](code/web/frontend/src/api/client.js)
- Image workflow: `ImageMode` in [code/web/frontend/src/pages/ImageMode.jsx](code/web/frontend/src/pages/ImageMode.jsx)
- Video workflow: `VideoMode` in [code/web/frontend/src/pages/VideoMode.jsx](code/web/frontend/src/pages/VideoMode.jsx)
- Charts: `ChartPanel` in [code/web/frontend/src/components/ChartPanel.jsx](code/web/frontend/src/components/ChartPanel.jsx)

## API Docs

See [docs/api.md](docs/api.md).
