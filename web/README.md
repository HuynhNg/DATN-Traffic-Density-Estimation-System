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
- Computes live traffic metrics: FPS, active objects, total vehicles by window,
  occupancy, PCE count, and congestion alert level.
- Exports realtime metric history to Excel for later analysis.
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
  -> users can edit the polygon ROI overlay; manual ROI overrides auto ROI
  -> each streamed frame is optionally tracked, rendered, and JPEG encoded
  -> backend updates live_metrics, live_series, and exportable metric history
  -> frontend displays MJPEG stream, live cards, alert badge, and bucketed chart
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
APP_INFERENCE_LOCK_ENABLED=true
APP_FRAME_SKIP=1
APP_JPEG_QUALITY=85
APP_STREAM_MAX_DIM=640
APP_SAVE_DIR=storage
APP_MAX_VIDEO_UPLOAD_MB=500
APP_CLEAN_STORAGE_ON_STARTUP=true
APP_AUTO_PROCESS_VIDEO=false
APP_LOG_DETECTIONS=false
APP_LIVE_HISTORY_MAX_ROWS=108000
APP_LIVE_FLOW_EVENTS_MAX_ROWS=50000
APP_LIVE_TRACK_STATES_MAX_ROWS=10000
APP_PCE_MOTOR=0.30
APP_PCE_CAR=1.00
APP_PCE_TRUCK=2.50
APP_PCE_BUS=3.00
APP_ALERT_ROLLING_WINDOW_SEC=60
APP_ALERT_REFERENCE_ROI_AREA_RATIO=0.35
APP_ALERT_MIN_ROI_AREA_RATIO=0.10
APP_ALERT_MIN_ROI_SCALE=0.50
APP_ALERT_MAX_ROI_SCALE=2.50
APP_ALERT_OCC_LOW=15
APP_ALERT_OCC_MID=30
APP_ALERT_OCC_HIGH=50
APP_ALERT_PCE_DENSITY_LOW=6
APP_ALERT_PCE_DENSITY_MID=12
APP_ALERT_PCE_DENSITY_HIGH=20
APP_ALERT_VEHICLE_DENSITY_LOW=5
APP_ALERT_VEHICLE_DENSITY_MID=10
APP_ALERT_VEHICLE_DENSITY_HIGH=18
APP_ALERT_OCCUPANCY_WEIGHT=0.40
APP_ALERT_PCE_DENSITY_WEIGHT=0.40
APP_ALERT_VEHICLE_DENSITY_WEIGHT=0.20
APP_ALERT_SCORE_BUSY=0.75
APP_ALERT_SCORE_CONGESTED=1.50
APP_ALERT_SCORE_GRIDLOCK=2.30
APP_ALERT_HYSTERESIS_SEC=3.0
```

`APP_MAX_VIDEO_UPLOAD_MB` limits `/api/video/upload`; files above this size are
rejected with HTTP `413`. `APP_CLEAN_STORAGE_ON_STARTUP=true` clears old
uploads and generated artifacts inside `APP_SAVE_DIR` every time the backend
starts.

`APP_INFERENCE_LOCK_ENABLED=true` serializes calls to the shared YOLO model
instance, which prevents realtime streaming and offline processing from calling
the same model concurrently. The `APP_LIVE_*_MAX_ROWS` options cap in-memory
realtime history, direction events, and track state data so long-running streams
do not grow memory indefinitely. Realtime `All` metrics and exports are based on
the history still retained in memory.

`APP_AUTO_PROCESS_VIDEO=false` avoids running offline video processing at the
same time as the realtime MJPEG stream. Enable it only when you explicitly need
processed MP4 files to be generated immediately after upload.

Realtime density alerts use rolling averages over
`APP_ALERT_ROLLING_WINDOW_SEC`, not a single frame. `APP_PCE_*` sets the PCE
weight per class. PCE and active vehicle counts are normalized by ROI size using:

```text
roi_scale = APP_ALERT_REFERENCE_ROI_AREA_RATIO
            / max(current_roi_area_ratio, APP_ALERT_MIN_ROI_AREA_RATIO)
roi_scale is clamped to APP_ALERT_MIN_ROI_SCALE..APP_ALERT_MAX_ROI_SCALE
```

The alert score combines average occupancy, ROI-normalized PCE density, and
ROI-normalized active vehicle density with the configured weights. This keeps a
small ROI from using the same raw PCE thresholds as a large ROI.
`APP_ALERT_HYSTERESIS_SEC` requires a new alert level to stay stable for a short
period before the displayed label changes, reducing alert flicker near
thresholds.

Set `APP_LOG_DETECTIONS=true` while debugging ByteTrack. The backend logs each
stream frame's bounding boxes before ROI filtering, after ROI filtering, and
after ByteTrack with class name, class id, confidence, bbox coordinates, and
track id when available.

ByteTrack options:

```text
APP_BYTETRACK_ENABLED=true
APP_BYTETRACK_CONF_HIGH=0.5
APP_BYTETRACK_IOU_HIGH=0.8
APP_BYTETRACK_TRACK_BUFFER=30
APP_BYTETRACK_MIN_BOX_AREA=400
APP_BYTETRACK_FRAME_SKIP=1
APP_BYTETRACK_REPAIR_ENABLED=true
APP_BYTETRACK_REPAIR_IOU=0.50
APP_FLOW_EXIT_TIMEOUT_SEC=2.0
APP_FLOW_DIRECTION_MIN_DX_RATIO=0.03
```

For BoxMOT ByteTrack, `APP_BYTETRACK_IOU_HIGH` maps to `match_thresh`. Higher
values are more tolerant during association. The repair options keep a previous
`track_id` when the current detection still overlaps strongly with the previous
tracked bbox, which reduces one-frame drops and track-id churn on small motors.
`APP_FLOW_EXIT_TIMEOUT_SEC` controls how long a track can disappear before it is
marked inactive. `APP_FLOW_DIRECTION_MIN_DX_RATIO` controls the minimum
horizontal movement, relative to frame width, required before a track is counted
as `Left to Right` or `Right to Left`.

Adaptive ROI options:

```text
APP_ROI_ENABLED=true
APP_ROI_MODE=mask
APP_ROI_ANCHOR=bottom_center
APP_ROI_MIN_BBOX_OVERLAP=0.10
APP_ROI_CALIB_FRAMES=100
APP_ROI_DRAW=true
APP_ROI_DRAW_ALPHA=0.25
```

For realtime stream, ROI calibration is non-blocking. The backend still detects
and streams the initial frames normally without ROI; once `APP_ROI_CALIB_FRAMES`
frames have been collected, ROI filtering and drawing are applied to subsequent
frames. Auto ROI is computed once and is not periodically recalibrated.

Set `APP_ROI_ENABLED=false` and `APP_ROI_DRAW=false` if you want to run full
frame detection without ROI calibration or overlay drawing.

ROI filtering keeps a detection when either the configured anchor point is
inside the ROI or at least `APP_ROI_MIN_BBOX_OVERLAP` of the bbox area overlaps
the ROI mask. This avoids dropping small motor boxes whose anchor point jitters
slightly outside the ROI boundary.

When auto ROI is ready in Video Mode, the frontend shows an editable polygon ROI
over the MJPEG stream. Drag the polygon to move it, drag individual vertices to
reshape it, or add/remove vertices when the road area needs more than four
points. The frontend saves normalized ROI coordinates to the backend, and that
manual ROI is used for subsequent detection frames.

`Total Vehicles` in realtime mode counts unique ByteTrack `track_id` values in
the selected metric range:

```text
1 Min  = unique vehicles seen in the latest 60 seconds
1 Hour = unique vehicles seen in the latest 3600 seconds
All    = unique vehicles seen across the whole current stream
```

`Left to Right` counts track IDs whose bbox center moves enough in the positive
X direction during the selected range. `Right to Left` counts track IDs whose
bbox center moves enough in the negative X direction during the selected range.
Each track is counted once after its horizontal displacement exceeds
`APP_FLOW_DIRECTION_MIN_DX_RATIO * frame_width`. For stricter lane-specific
counts, add a counting line/counter zone.

The bottom chart uses the same selector as the metric cards. `1 Min` and `All`
show vehicle counts grouped by minute; `1 Hour` shows vehicle counts grouped by
hour. Each bucket counts unique `track_id` values seen in that bucket.

If the stream has not been running long enough to fill the selected window, the
backend counts over the history that is already available. Accurate totals
require ByteTrack to be enabled so the same physical vehicle keeps a stable
`track_id` across frames. If no `track_id` is available, the backend falls back
to the latest active object count because it cannot reliably know whether a
detection in two frames is the same vehicle.

Density alerts use a different value: `Avg Active` is the average number of
vehicles present per processed frame in the alert rolling window. `PCE Density`
is the rolling average of PCE after ROI-size normalization, and `Avg Occupancy`
is the rolling average of bbox-covered ROI area. These rolling metrics drive
the alert label and alert score.

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

The backend rejects videos larger than `APP_MAX_VIDEO_UPLOAD_MB` with HTTP
`413`.

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
GET /api/video/{job_id}?avg_window=minute
```

`avg_window` accepts `minute`, `hour`, or `all`.

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
  "live_series": [{ "t": "12:00", "count": 14 }],
  "analytics": {
    "avg_objects": 3.2,
    "series": [{ "frame": 1, "count": 2 }]
  },
  "roi": {
    "type": "polygon",
    "points": [
      { "x": 0.18, "y": 0.42 },
      { "x": 0.62, "y": 0.4 },
      { "x": 0.82, "y": 0.96 },
      { "x": 0.08, "y": 0.96 }
    ]
  },
  "roi_source": "auto"
}
```

### Export Metrics Excel

```http
GET /api/video/{job_id}/metrics/export?avg_window=minute
```

Downloads an `.xlsx` workbook with:

- `Theo phút`: data grouped by minute.
- `Theo giờ`: data grouped by hour.

Each sheet includes:

- `Thời gian`: bucket start time.
- `Tổng số xe`: unique tracked vehicles in that minute/hour.
- `Xe đi vào (phải sang trái)`: vehicles moving right to left.
- `Xe đi ra (trái sang phải)`: vehicles moving left to right.
- `Xe máy`, `Ô tô`, `Xe buýt`, `Xe tải`: unique tracked vehicles by class.

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

### Set Manual ROI

```http
POST /api/video/{job_id}/roi
```

Body uses normalized image coordinates. Polygon ROI is the preferred format:

```json
{
  "type": "polygon",
  "points": [
    { "x": 0.18, "y": 0.42 },
    { "x": 0.62, "y": 0.4 },
    { "x": 0.82, "y": 0.96 },
    { "x": 0.08, "y": 0.96 }
  ]
}
```

Legacy rectangular ROI payloads are still accepted as `{ "x": 0.1, "y": 0.2,
"w": 0.7, "h": 0.5 }`.

Response:

```json
{
  "job_id": "uuid",
  "roi": {
    "type": "polygon",
    "points": [
      { "x": 0.18, "y": 0.42 },
      { "x": 0.62, "y": 0.4 },
      { "x": 0.82, "y": 0.96 },
      { "x": 0.08, "y": 0.96 }
    ]
  },
  "roi_source": "manual"
}
```

### Reset Manual ROI

```http
DELETE /api/video/{job_id}/roi
```

Clears the manual ROI and returns to the auto ROI when one is available.

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
