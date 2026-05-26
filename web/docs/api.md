# TrafficAI API

Base URL: `http://localhost:8000`

## Image Detection

`POST /api/image`

Query params:
- `labels` bool
- `conf` bool

Form-data:
- `file`: image file

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

## Video Upload

`POST /api/video/upload`

Query params:
- `labels` bool
- `conf` bool

Form-data:
- `file`: video file

Response:

```json
{
  "job_id": "uuid",
  "fps": 29.97,
  "total_frames": 932
}
```

## Video Status

`GET /api/video/{job_id}`

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
    "objects_in_frame": 4
  },
  "live_series": [
    {"t": "12:00:01", "count": 4}
  ],
  "analytics": {
    "avg_objects": 3.2,
    "series": [
      {"frame": 1, "count": 2}
    ]
  }
}
```

## Download Processed Video

`GET /api/video/{job_id}/result`

## MJPEG Stream (Annotated)

`GET /api/video/{job_id}/stream`

Query params:
- `labels` bool
- `conf` bool
- `target_fps` int

Response:

`multipart/x-mixed-replace; boundary=frame`

