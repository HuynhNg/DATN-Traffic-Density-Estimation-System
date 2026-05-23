from __future__ import annotations

from pydantic import BaseModel


class Detection(BaseModel):
    object_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class ImageResponse(BaseModel):
    image_b64: str
    detections: list[Detection]
    inference_ms: float


class VideoJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    result_url: str | None
    analytics: dict | None
