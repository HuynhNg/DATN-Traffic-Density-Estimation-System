# TrafficAI Realtime Vehicle Detection

Production-ready web system for realtime traffic detection with YOLOv8s custom model.

## Project Structure

- backend/ FastAPI + YOLOv8 inference
- frontend/ React + Vite + Tailwind
- docs/ API documentation

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

## API Docs

See [docs/api.md](docs/api.md).
