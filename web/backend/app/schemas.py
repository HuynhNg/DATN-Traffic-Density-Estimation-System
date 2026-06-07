from __future__ import annotations

from pydantic import BaseModel


class Detection(BaseModel):
    """One detected or tracked object in an image/frame."""

    object_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int | None = None


class ImageResponse(BaseModel):
    """Response payload for image detection."""

    image_b64: str
    detections: list[Detection]
    inference_ms: float
    processing_ms: float


class LiveMetrics(BaseModel):
    """Realtime traffic metrics for a video job."""

    fps: float
    avg_objects: float
    total_vehicles: int | None = None
    objects_in_frame: int
    pce_count: float | None = None
    occupancy_pct: float | None = None
    alert_level: int | None = None
    alert_label: str | None = None
    alert_message: str | None = None


class VideoJobStatus(BaseModel):
    """Status payload for background video processing."""

    job_id: str
    status: str
    progress: float
    result_url: str | None
    fps: float | None = None
    total_frames: int | None = None
    live_metrics: LiveMetrics | None = None
    live_series: list[dict] | None = None
    analytics: dict | None
