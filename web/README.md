# TrafficAI Realtime Vehicle Detection

TrafficAI is a realtime traffic analytics web application. It combines a
FastAPI backend, YOLOv8 vehicle detection, optional ByteTrack object tracking,
optional adaptive ROI filtering, and a React + Vite dashboard for image and
video workflows.

## What This Project Does

- Detects vehicles in uploaded images and returns annotated results.
- Uploads videos, processes them in the background, and produces annotated MP4
  output.
- Streams annotated MJPEG previews for uploaded videos.
- Tracks vehicles across frames with ByteTrack when enabled.
- Computes live traffic metrics: FPS, active objects, average objects,
  occupancy, PCE count, and congestion alert level.
- Displays detection results, live metrics, and time-series charts in the
  frontend.

## Repository Structure

```text
web/
  README.md                         Combined project, setup, flow, and API docs
  requirements.txt                  Root pointer to backend requirements

  backend/
    .env                            Active backend runtime configuration
    .env.example                    Backend environment sample
    requirements.txt                Python dependencies
    app/
      main.py                       FastAPI app bootstrap
      __main__.py                   Short backend runner: python -m app
      schemas.py                    Shared response schemas
      api/
        routes.py                   HTTP endpoints
      core/
        config.py                   App settings loaded from APP_* env vars
        logging.py                  Logging setup
      services/
        adaptive_roi.py             Motion-based ROI estimation and filtering
        analytics.py                Rolling FPS/object count metrics
        frame_utils.py              Frame resize helpers
        inference.py                YOLOv8 runtime wrapper
        renderer.py                 Bounding-box rendering
        tracking.py                 ByteTrack adapter
        traffic_metrics.py          Occupancy, PCE, and alert logic
        video_jobs.py               In-memory video jobs and offline processing
    models/
      best.pt                       PyTorch YOLO weights
      best.onnx                     Optional exported ONNX model
      best_fp16.engine              Optional TensorRT FP16 engine
    storage/                        Uploaded videos and processed artifacts
    tools/
      benchmark_models.py           Compare .pt, .onnx, and .engine speed

  frontend/
    .env.example                    Frontend environment sample
    package.json                    Node scripts and dependencies
    src/
      api/client.js                 Backend API client
      components/                   Reusable UI components
      pages/ImageMode.jsx           Image upload and detection workflow
      pages/VideoMode.jsx           Video upload, stream, metrics workflow
      styles/index.css              Tailwind entry and global styles
      utils/fileTypes.js            Shared upload type validation
```

## Runtime Flow

```text
React UI
  -> Upload image/video
  -> FastAPI /api endpoints
  -> OpenCV file/frame decoding
  -> YOLOv8 inference
  -> Optional adaptive ROI filtering
  -> Optional ByteTrack tracking
  -> Bounding-box rendering
  -> Analytics and traffic metrics
  -> JSON, processed MP4, or MJPEG stream back to React
```

### Image Flow

```text
User selects image
  -> frontend validates JPG/PNG/WEBP
  -> POST /api/image
  -> backend decodes image with OpenCV
  -> InferenceEngine.detect()
  -> render_boxes()
  -> backend returns base64 annotated JPEG + detections + timings
  -> frontend displays result, stats, detection list, and export action
```

### Video Upload and Offline Processing Flow

```text
User selects video
  -> frontend validates MP4/MOV/AVI/MKV/WEBM
  -> POST /api/video/upload
  -> backend stores source video in storage/
  -> backend creates in-memory VideoJob
  -> process_video() runs only when APP_AUTO_PROCESS_VIDEO=true
  -> each processed frame is detected, optionally tracked, rendered, and written
  -> GET /api/video/{job_id} reports progress and analytics
  -> GET /api/video/{job_id}/result downloads the processed MP4
```

### Realtime Preview Flow

```text
User clicks Run AI
  -> frontend creates /api/video/{job_id}/stream URL
  -> backend reads source video frame-by-frame
  -> target_fps controls frame skipping; default is 30 to keep ByteTrack stable
  -> initial frames are detected full-frame while ROI calibration frames are collected
  -> ROI filtering and overlay are applied only after enough calibration frames
  -> each streamed frame is optionally tracked, rendered, and JPEG encoded
  -> backend updates live_metrics and live_series on the video job
  -> frontend displays MJPEG stream, live cards, alert badge, and chart
```

## Supported Files

- Images: `.jpg`, `.jpeg`, `.png`, `.webp`
- Videos: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

## Quick Start

Run the backend:

```bash
cd web/backend
python -m app
```

Run the frontend in another terminal:

```bash
cd web/frontend
npm run dev
```

Default URLs:

```text
Backend API: http://localhost:8000
FastAPI docs: http://localhost:8000/docs
Frontend: shown by Vite after npm run dev
```

## Backend Setup

```bash
cd web/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes `onnxruntime-gpu` so the exported ONNX model can run
through Ultralytics/ONNX Runtime. If your machine does not have a compatible
NVIDIA GPU/CUDA setup, replace it with `onnxruntime` for CPU-only ONNX
inference.

Place the YOLO PyTorch model at:

```text
web/backend/models/best.pt
```

The active model is configured by `APP_MODEL_PATH` in `web/backend/.env`. The
current configuration uses the exported ONNX model:

```text
APP_MODEL_PATH=models/best.onnx
```

To switch back to PyTorch weights:

```bash
set APP_MODEL_PATH=models/best.pt
```

Run the API with the short project runner:

```bash
python -m app
```

The runner reads `APP_HOST`, `APP_PORT`, and `APP_RELOAD` from
`web/backend/.env`. The current default is:

```text
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=true
```

The equivalent explicit command is:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open generated FastAPI docs:

```text
http://localhost:8000/docs
```

## Frontend Setup

```bash
cd web/frontend
npm install
npm run dev
```

Optional frontend environment:

```text
VITE_API_BASE=http://localhost:8000
```

## GPU Setup

Install NVIDIA driver and a CUDA-compatible PyTorch build. Example for CUDA
12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:

```python
import torch
print(torch.cuda.is_available())
```

Force GPU usage:

```bash
set APP_DEVICE=cuda:0
set APP_USE_HALF=true
```

## Model Export

The backend can load PyTorch weights (`.pt`) and Ultralytics exported models
such as ONNX (`.onnx`) and TensorRT (`.engine`) through the same
`APP_MODEL_PATH` setting.

Recommended optimization path:

```text
best.pt -> best.onnx -> benchmark -> best_fp16.engine
```

Export ONNX:

```bash
cd web/backend
yolo export model=models/best.pt format=onnx imgsz=640 simplify=True opset=12
```

The exported file is typically written to:

```text
web/backend/models/best.onnx
```

Then update `web/backend/.env`:

```text
APP_MODEL_PATH=models/best.onnx
```

If ONNX inference fails because the runtime is missing, install the matching
ONNX Runtime package for your machine, for example `onnxruntime` for CPU or
`onnxruntime-gpu` for NVIDIA GPU.

Export TensorRT FP16 on the deployment GPU:

```bash
cd web/backend
yolo export model=models/best.pt format=engine imgsz=640 half=True device=0
```

Ultralytics typically writes `models/best.engine`. If you rename it to
`models/best_fp16.engine`, select it with:

```text
APP_MODEL_PATH=models/best_fp16.engine
```

TensorRT engine input size must match `APP_IMG_SIZE`. For example, an engine
exported with `imgsz=640` must run with:

```text
APP_IMG_SIZE=640
```

If you want to run `APP_IMG_SIZE=960`, export a matching engine:

```bash
yolo export model=models/best.pt format=engine imgsz=960 half=True device=0
```

The backend currently selects ONNX in `web/backend/.env`:

```text
APP_MODEL_PATH=models/best.onnx
```

Benchmark available models on the same image or video:

```bash
cd web/backend
python tools/benchmark_models.py --source path\to\sample.mp4 --device 0
```

To choose exactly which models to compare:

```bash
python tools/benchmark_models.py ^
  --source path\to\sample.mp4 ^
  --models models/best.pt models/best.onnx models/best_fp16.engine ^
  --device 0 ^
  --max-frames 120
```

Use `--device cpu` when benchmarking CPU-compatible models only. TensorRT engine
files are GPU and TensorRT-version specific, so build `best_fp16.engine` on the
same machine that will run the backend.

## Backend Configuration

Settings are loaded from `APP_*` environment variables in `web/backend/.env`.
Use `web/backend/.env.example` as the template if the active `.env` file is
missing.

Common options:

```text
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=true
APP_MODEL_PATH=models/best.onnx
APP_CONF=0.25
APP_IOU=0.45
APP_IMG_SIZE=640
APP_DEVICE=auto
APP_USE_HALF=true
APP_FRAME_SKIP=1
APP_JPEG_QUALITY=85
APP_STREAM_MAX_DIM=640
APP_AUTO_PROCESS_VIDEO=false
```

`APP_AUTO_PROCESS_VIDEO=false` avoids running offline video processing at the
same time as the realtime MJPEG stream. Enable it only when you explicitly need
processed MP4 files to be generated immediately after upload.

ByteTrack options:

```text
APP_BYTETRACK_ENABLED=true
APP_BYTETRACK_CONF_HIGH=0.5
APP_BYTETRACK_IOU_HIGH=0.3
APP_BYTETRACK_TRACK_BUFFER=30
APP_BYTETRACK_MIN_BOX_AREA=400
APP_BYTETRACK_FRAME_SKIP=1
```

Adaptive ROI options:

```text
APP_ROI_ENABLED=true
APP_ROI_MODE=mask
APP_ROI_ANCHOR=bottom_center
APP_ROI_CALIB_FRAMES=100
APP_ROI_ROLLING_WINDOW_SIZE=100
APP_ROI_UPDATE_INTERVAL_SEC=600
APP_ROI_DRAW=true
APP_ROI_DRAW_ALPHA=0.25
```

For realtime stream, ROI calibration is non-blocking. The backend still detects
and streams the initial frames normally without ROI; once `APP_ROI_CALIB_FRAMES`
frames have been collected, ROI filtering and drawing are applied to subsequent
frames.

Set `APP_ROI_ENABLED=false` and `APP_ROI_DRAW=false` if you want to run full
frame detection without ROI calibration or overlay drawing.

## API Reference

Base URL:

```text
http://localhost:8000
```

### Health

```http
GET /api/health
```

Response:

```json
{ "status": "ok" }
```

### Image Detection

```http
POST /api/image?labels=true&conf=true
```

Form data:

```text
file: image file
```

Response:

```json
{
  "image_b64": "...",
  "detections": [
    {
      "object_id": 0,
      "class_id": 0,
      "class_name": "car",
      "confidence": 0.92,
      "x1": 12,
      "y1": 20,
      "x2": 180,
      "y2": 240
    }
  ],
  "inference_ms": 12.3,
  "processing_ms": 18.6
}
```

### Video Upload

```http
POST /api/video/upload?labels=true&conf=true
```

Form data:

```text
file: video file
```

Response:

```json
{
  "job_id": "uuid",
  "fps": 29.97,
  "total_frames": 932
}
```

### Video Status

```http
GET /api/video/{job_id}
```

Response:

```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": 0.6,
  "result_url": "/api/video/{job_id}/result",
  "fps": 29.97,
  "total_frames": 932,
  "live_metrics": {
    "fps": 12.4,
    "avg_objects": 3.1,
    "objects_in_frame": 4,
    "pce_count": 9.2,
    "occupancy_pct": 17.3,
    "alert_level": 1,
    "alert_label": "BUSY",
    "alert_message": "Traffic is increasing"
  },
  "live_series": [{ "t": "12:00:01", "count": 4 }],
  "analytics": {
    "avg_objects": 3.2,
    "series": [{ "frame": 1, "count": 2 }]
  }
}
```

### Start Offline Video Processing

```http
POST /api/video/{job_id}/process?labels=true&conf=true
```

Starts MP4 generation for an uploaded video. This is optional when
`APP_AUTO_PROCESS_VIDEO=false`.

Response:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "progress": 0.0,
  "result_url": null
}
```

### Download Processed Video

```http
GET /api/video/{job_id}/result
```

Returns the processed MP4 file.

### MJPEG Stream

```http
GET /api/video/{job_id}/stream?labels=true&conf=true&target_fps=30
```

Response content type:

```text
multipart/x-mixed-replace; boundary=frame
```

## Traffic Metrics

PCE weights:

```text
motor/motorcycle = 0.30
car = 1.00
truck = 2.50
bus = 3.00
```

Alert labels:

```text
NORMAL
BUSY
CONGESTED
GRIDLOCK
```

When ByteTrack is enabled, video analytics and live metrics are based on tracked
objects rather than raw detections.
