from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import cv2

from app.core.config import settings
from app.services.inference import InferenceEngine
from app.services.renderer import render_boxes


@dataclass
class VideoJob:
    job_id: str
    status: str
    progress: float
    result_path: str | None
    analytics: dict | None
    fps: float | None = None
    total_frames: int | None = None
    input_path: str | None = None
    live_metrics: dict | None = None
    live_series: list[dict[str, Any]] | None = None


class VideoJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, VideoJob] = {}

    def create(self) -> VideoJob:
        job_id = str(uuid.uuid4())
        job = VideoJob(job_id=job_id, status="queued", progress=0.0, result_path=None, analytics=None)
        self.jobs[job_id] = job
        return job

    def get(self, job_id: str) -> VideoJob | None:
        return self.jobs.get(job_id)


def process_video(
    engine: InferenceEngine,
    job: VideoJob,
    input_path: str,
    show_labels: bool,
    show_conf: bool,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        job.status = "failed"
        return

    fps = job.fps or cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    os.makedirs(settings.save_dir, exist_ok=True)
    output_path = os.path.join(settings.save_dir, f"{job.job_id}.mp4")
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    total_frames = job.total_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    job.status = "processing"

    series: list[dict[str, Any]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if settings.frame_skip > 1 and frame_idx % settings.frame_skip != 0:
            writer.write(frame)
            continue

        detections, _ = engine.detect(frame)
        annotated = render_boxes(frame, detections, show_labels, show_conf)
        writer.write(annotated)

        series.append(
            {
                "frame": frame_idx,
                "count": len(detections),
            }
        )

        if total_frames > 0:
            job.progress = min(frame_idx / total_frames, 1.0)

    cap.release()
    writer.release()

    job.status = "done"
    job.progress = 1.0
    job.result_path = output_path
    avg = sum(s["count"] for s in series) / max(len(series), 1)
    job.analytics = {
        "avg_objects": round(avg, 2),
        "series": series[-200:],
    }
