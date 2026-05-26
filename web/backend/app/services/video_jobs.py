from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import cv2

from app.core.config import settings
from app.services.adaptive_roi import (
    ROIUpdater,
    apply_roi_to_frame,
    build_roi_config_from_settings,
    calibrate_roi_from_capture,
    draw_roi_overlay,
    filter_detections_by_roi_xyxy,
    remap_detections_to_original,
)
from app.services.inference import InferenceEngine
from app.services.renderer import render_boxes
from app.services.tracking import ByteTrackWrapper, detections_to_array, tracks_to_detections

import logging

logger = logging.getLogger("app.video")


@dataclass
# In-memory record for a video processing job.
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


# Simple in-memory job registry.
class VideoJobStore:
    # Initialize in-memory job storage.
    def __init__(self) -> None:
        self.jobs: dict[str, VideoJob] = {}

    # Create and register a new video job.
    def create(self) -> VideoJob:
        job_id = str(uuid.uuid4())
        job = VideoJob(job_id=job_id, status="queued", progress=0.0, result_path=None, analytics=None)
        self.jobs[job_id] = job
        return job

    # Fetch a job by id.
    def get(self, job_id: str) -> VideoJob | None:
        return self.jobs.get(job_id)


# Process a video file and write annotated output to disk.
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

    roi = None
    roi_updater: ROIUpdater | None = None
    if settings.roi_enabled:
        roi_config = build_roi_config_from_settings(settings)
        roi = calibrate_roi_from_capture(cap, roi_config)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if roi is not None:
            roi_updater = ROIUpdater(roi_config, roi)
            roi_updater.start()

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
    tracker = None
    last_tracked: list[dict[str, Any]] | None = None
    track_stride = max(1, settings.bytetrack_frame_skip)
    if settings.bytetrack_enabled:
        try:
            tracker = ByteTrackWrapper(frame_rate=int(round(fps or 30)))
            logger.info("ByteTrack enabled for offline video job %s", job.job_id)
        except RuntimeError:
            logger.exception("ByteTrack failed to initialize for job %s", job.job_id)
            job.status = "failed"
            return
    else:
        logger.info("ByteTrack disabled for offline video job %s", job.job_id)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if settings.frame_skip > 1 and frame_idx % settings.frame_skip != 0:
            writer.write(frame)
            continue

        run_tracker = tracker is not None and frame_idx % track_stride == 0
        if roi_updater is not None:
            roi_updater.push_frame(frame)
            current_roi = roi_updater.current
            processed, offset = apply_roi_to_frame(frame, current_roi, mode=settings.roi_mode)
            detections, _ = engine.detect(processed)
            detections = remap_detections_to_original(detections, offset)
            detections = filter_detections_by_roi_xyxy(
                detections,
                current_roi,
                anchor=settings.roi_anchor,
            )
            det_array = detections_to_array(detections)
            if run_tracker:
                tracked = tracks_to_detections(tracker.update(det_array, frame), settings.class_names)
                last_tracked = tracked
            else:
                tracked = last_tracked or detections
            annotated = render_boxes(frame, tracked, show_labels, show_conf)
            if settings.roi_draw:
                annotated = draw_roi_overlay(annotated, current_roi, alpha=settings.roi_draw_alpha)
        else:
            detections, _ = engine.detect(frame)
            det_array = detections_to_array(detections)
            if run_tracker:
                tracked = tracks_to_detections(tracker.update(det_array, frame), settings.class_names)
                last_tracked = tracked
            else:
                tracked = last_tracked or detections
            annotated = render_boxes(frame, tracked, show_labels, show_conf)
        writer.write(annotated)

        series.append(
            {
                "frame": frame_idx,
                "count": len(tracked),
            }
        )

        if total_frames > 0:
            job.progress = min(frame_idx / total_frames, 1.0)

    cap.release()
    writer.release()

    if roi_updater is not None:
        roi_updater.stop()

    job.status = "done"
    job.progress = 1.0
    job.result_path = output_path
    avg = sum(s["count"] for s in series) / max(len(series), 1)
    job.analytics = {
        "avg_objects": round(avg, 2),
        "series": series[-200:],
    }
