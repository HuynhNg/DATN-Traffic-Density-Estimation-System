from __future__ import annotations

import base64
import time
import os
import uuid
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import settings
from app.services.inference import InferenceEngine
from app.services.adaptive_roi import (
    ROIUpdater,
    apply_roi_to_frame,
    build_roi_config_from_settings,
    compute_adaptive_roi,
    draw_roi_overlay,
    filter_detections_by_roi_xyxy,
    remap_detections_to_original,
)
from app.services.frame_utils import resize_to_max
from app.services.renderer import render_boxes
from app.services.analytics import AnalyticsTracker
from app.services.video_jobs import VideoJobStore, process_video


router = APIRouter(prefix="/api")


# Resolve shared inference engine from router state.
async def get_engine() -> InferenceEngine:
    return router.engine  # type: ignore[attr-defined]


@router.get("/health")
# Lightweight health check endpoint.
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/image")
# Run detection on a single image and return annotated JPEG plus metadata.
async def detect_image(
    file: UploadFile = File(...),
    labels: bool = True,
    conf: bool = True,
    engine: InferenceEngine = Depends(get_engine),
) -> dict[str, Any]:
    start = time.perf_counter()
    content = await file.read()
    np_arr = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    detections, infer_ms = engine.detect(frame)
    annotated = render_boxes(frame, detections, labels, conf)
    _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality])
    b64 = base64.b64encode(buffer).decode("ascii")
    total_ms = (time.perf_counter() - start) * 1000.0

    return {
        "image_b64": b64,
        "detections": detections,
        "inference_ms": round(infer_ms, 2),
        "processing_ms": round(total_ms, 2),
    }


@router.post("/video/upload")
# Upload a video file and enqueue background processing.
async def upload_video(
    file: UploadFile = File(...),
    labels: bool = True,
    conf: bool = True,
) -> dict[str, Any]:
    os.makedirs(settings.save_dir, exist_ok=True)
    temp_id = str(uuid.uuid4())
    input_path = os.path.join(settings.save_dir, f"{temp_id}_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Invalid video")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0:
        raise HTTPException(status_code=400, detail="Cannot read video FPS")

    job = router.video_jobs.create()  # type: ignore[attr-defined]
    job.fps = fps
    job.total_frames = total_frames
    job.input_path = input_path

    import asyncio

    asyncio.create_task(
        asyncio.to_thread(
            process_video,
            router.engine,
            job,
            input_path,
            labels,
            conf,
        )
    )

    return {"job_id": job.job_id, "fps": fps, "total_frames": total_frames}


@router.get("/video/{job_id}")
# Fetch processing status and metrics for a video job.
async def get_video_status(job_id: str) -> dict[str, Any]:
    job = router.video_jobs.get(job_id)  # type: ignore[attr-defined]
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "result_url": f"/api/video/{job_id}/result" if job.result_path else None,
        "analytics": job.analytics,
        "fps": job.fps,
        "total_frames": job.total_frames,
        "live_metrics": job.live_metrics,
        "live_series": job.live_series,
    }


@router.get("/video/{job_id}/result")
# Download the processed video artifact.
async def download_video(job_id: str) -> FileResponse:
    job = router.video_jobs.get(job_id)  # type: ignore[attr-defined]
    if not job or not job.result_path:
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(job.result_path, filename=f"{job_id}.mp4")


@router.get("/video/{job_id}/stream")
# Stream annotated frames as MJPEG for realtime preview.
async def stream_video(
    job_id: str,
    labels: bool = True,
    conf: bool = True,
    target_fps: int = 12,
    engine: InferenceEngine = Depends(get_engine),
) -> StreamingResponse:
    job = router.video_jobs.get(job_id)  # type: ignore[attr-defined]
    if not job or not job.input_path:
        raise HTTPException(status_code=404, detail="Job not found")

    # Generator that yields multipart JPEG frames for MJPEG streaming.
    def frame_generator():
        cap = cv2.VideoCapture(job.input_path)
        if not cap.isOpened():
            return

        roi = None
        roi_updater: ROIUpdater | None = None
        if settings.roi_enabled:
            roi_config = build_roi_config_from_settings(settings)
            calib_frames: list[np.ndarray] = []
            while len(calib_frames) < roi_config.calib_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                calib_frames.append(resize_to_max(frame, settings.stream_max_dim))
            if len(calib_frames) >= 2:
                roi = compute_adaptive_roi(calib_frames, roi_config)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if roi is not None:
                roi_updater = ROIUpdater(roi_config, roi)
                roi_updater.start()

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        safe_target = max(1, min(int(target_fps or 12), 60))
        skip = max(1, round(source_fps / safe_target)) if source_fps else 1
        tracker = AnalyticsTracker()

        while True:
            if skip > 1:
                for _ in range(skip - 1):
                    if not cap.grab():
                        cap.release()
                        return
                ret, frame = cap.retrieve()
            else:
                ret, frame = cap.read()
            if not ret:
                break

            resized = resize_to_max(frame, settings.stream_max_dim)
            if roi_updater is not None:
                roi_updater.push_frame(resized)
                current_roi = roi_updater.current
                processed, offset = apply_roi_to_frame(resized, current_roi, mode=settings.roi_mode)
                detections, _ = engine.detect(processed)
                detections = remap_detections_to_original(detections, offset)
                detections = filter_detections_by_roi_xyxy(
                    detections,
                    current_roi,
                    anchor=settings.roi_anchor,
                )
                annotated = render_boxes(resized, detections, labels, conf)
                if settings.roi_draw:
                    annotated = draw_roi_overlay(annotated, current_roi, alpha=settings.roi_draw_alpha)
            else:
                detections, _ = engine.detect(resized)
                annotated = render_boxes(resized, detections, labels, conf)
            metrics = tracker.update(len(detections))
            job.live_metrics = metrics
            if job.live_series is None:
                job.live_series = []
            job.live_series.append(
                {"t": time.strftime("%H:%M:%S"), "count": metrics["objects_in_frame"]}
            )
            job.live_series = job.live_series[-60:]
            ok, buffer = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality],
            )
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

        cap.release()

        if roi_updater is not None:
            roi_updater.stop()

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
