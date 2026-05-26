from __future__ import annotations

from pydantic import BaseModel


# One detected object in an image/frame.
class Detection(BaseModel):
    object_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


# Response payload for image detection.
class ImageResponse(BaseModel):
    image_b64: str
    detections: list[Detection]
    inference_ms: float


# Status payload for background video processing.
class VideoJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    result_url: str | None
    analytics: dict | None
