from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import settings
from app.services.adaptive_roi import (
    ROIConfig,
    ROIUpdater,
    apply_roi_to_frame,
    build_roi_config_from_settings,
    compute_adaptive_roi,
    draw_roi_overlay,
    filter_detections_by_roi_xyxy,
    remap_detections_to_original,
)
from app.services.analytics import AnalyticsTracker
from app.services.frame_utils import resize_to_max
from app.services.inference import InferenceEngine
from app.services.renderer import render_boxes
from app.services.tracking import (
    ByteTrackWrapper,
    detections_to_array,
    tracks_to_detections,
)
from app.services.traffic_metrics import compute_metrics, decide_alert
from app.services.video_jobs import VideoJob, process_video

logger = logging.getLogger("app.stream")


router = APIRouter(prefix="/api")


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
}
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class StreamROIState:
    config: ROIConfig | None
    updater: ROIUpdater | None = None
    calibration_frames: list[np.ndarray] | None = None
    failed: bool = False

    @classmethod
    def from_settings(cls) -> "StreamROIState":
        if not settings.roi_enabled:
            return cls(config=None)

        return cls(
            config=build_roi_config_from_settings(settings),
            calibration_frames=[],
        )


def _validate_upload(
    file: UploadFile,
    allowed_types: set[str],
    allowed_exts: set[str],
    label: str,
) -> None:
    content_type = (file.content_type or "").lower()
    suffix = Path(file.filename or "").suffix.lower()
    if content_type in allowed_types:
        return
    if suffix in allowed_exts:
        return
    allowed = ", ".join(sorted(allowed_exts))
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported {label} type. Allowed: {allowed}",
    )


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality],
    )
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buffer.tobytes()


def _save_upload(file: UploadFile) -> str:
    os.makedirs(settings.save_dir, exist_ok=True)
    temp_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name if file.filename else "upload.bin"
    input_path = os.path.join(settings.save_dir, f"{temp_id}_{safe_name}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return input_path


def _read_video_metadata(input_path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Invalid video")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0:
        raise HTTPException(status_code=400, detail="Cannot read video FPS")

    return fps, total_frames


def _update_stream_roi_state(
    state: StreamROIState,
    frame: np.ndarray,
    job_id: str,
) -> ROIUpdater | None:
    if state.config is None or state.failed:
        return None
    if state.updater is not None:
        return state.updater

    if state.calibration_frames is None:
        state.calibration_frames = []
    state.calibration_frames.append(frame.copy())

    if len(state.calibration_frames) < state.config.calib_frames:
        return None

    try:
        roi = compute_adaptive_roi(state.calibration_frames, state.config)
        state.updater = ROIUpdater(state.config, roi)
        state.updater.start()
        state.calibration_frames = None
        logger.info("Adaptive ROI enabled for stream job %s", job_id)
    except Exception:
        state.failed = True
        state.calibration_frames = None
        logger.exception("Adaptive ROI calibration failed for stream job %s", job_id)

    return None


def _init_bytetracker(job_id: str, source_fps: float) -> ByteTrackWrapper | None:
    if not settings.bytetrack_enabled:
        logger.info("ByteTrack disabled for stream job %s", job_id)
        return None

    try:
        tracker = ByteTrackWrapper(frame_rate=int(round(source_fps or 30)))
        logger.info("ByteTrack enabled for stream job %s", job_id)
        return tracker
    except RuntimeError:
        logger.exception("ByteTrack failed to initialize for stream job %s", job_id)
        return None


def _detect_stream_frame(
    frame: np.ndarray,
    engine: InferenceEngine,
    roi_updater: ROIUpdater | None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if roi_updater is None:
        roi_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        detections, _ = engine.detect(frame)
        return detections, roi_mask

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
    return detections, current_roi.combined_mask


def _track_stream_detections(
    detections: list[dict[str, Any]],
    frame: np.ndarray,
    bytetracker: ByteTrackWrapper | None,
    run_tracker: bool,
    last_tracked: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if bytetracker is None:
        return detections
    if not run_tracker:
        return last_tracked or detections

    det_array = detections_to_array(detections)
    return tracks_to_detections(
        bytetracker.update(det_array, frame),
        settings.class_names,
    )


def _update_live_metrics(
    job: VideoJob,
    tracker: AnalyticsTracker,
    tracked: list[dict[str, Any]],
    roi_mask: np.ndarray,
) -> None:
    metrics = tracker.update(len(tracked))
    occupancy, pce_total = compute_metrics(tracked, roi_mask)
    alert_level, alert_message = decide_alert(occupancy, pce_total)

    job.live_metrics = {
        **metrics,
        "occupancy_pct": round(occupancy, 2),
        "pce_count": round(pce_total, 2),
        "alert_level": alert_level,
        "alert_label": alert_message[0],
        "alert_message": alert_message[1],
    }

    if job.live_series is None:
        job.live_series = []
    job.live_series.append(
        {"t": time.strftime("%H:%M:%S"), "count": metrics["objects_in_frame"]}
    )
    job.live_series = job.live_series[-60:]


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
    _validate_upload(file, ALLOWED_IMAGE_TYPES, ALLOWED_IMAGE_EXTS, "image")
    start = time.perf_counter()
    content = await file.read()
    np_arr = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    detections, infer_ms = engine.detect(frame)
    annotated = render_boxes(frame, detections, labels, conf)
    b64 = base64.b64encode(_encode_jpeg(annotated)).decode("ascii")
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
    _validate_upload(file, ALLOWED_VIDEO_TYPES, ALLOWED_VIDEO_EXTS, "video")
    input_path = _save_upload(file)
    fps, total_frames = _read_video_metadata(input_path)

    job = router.video_jobs.create()  # type: ignore[attr-defined]
    job.fps = fps
    job.total_frames = total_frames
    job.input_path = input_path

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

        roi_state = StreamROIState.from_settings()
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        safe_target = max(1, min(int(target_fps or 12), 60))
        skip = max(1, round(source_fps / safe_target)) if source_fps else 1
        tracker = AnalyticsTracker()
        bytetracker = _init_bytetracker(job.job_id, source_fps)
        last_tracked: list[dict[str, Any]] | None = None
        track_stride = max(1, settings.bytetrack_frame_skip)
        stream_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                stream_idx += 1
                run_tracker = bytetracker is not None and stream_idx % track_stride == 0

                resized = resize_to_max(frame, settings.stream_max_dim)
                roi_updater = _update_stream_roi_state(
                    roi_state,
                    resized,
                    job.job_id,
                )
                detections, roi_mask = _detect_stream_frame(resized, engine, roi_updater)
                tracked = _track_stream_detections(
                    detections,
                    resized,
                    bytetracker,
                    run_tracker,
                    last_tracked,
                )
                if run_tracker:
                    last_tracked = tracked

                annotated = render_boxes(resized, tracked, labels, conf)
                if roi_updater is not None and settings.roi_draw:
                    annotated = draw_roi_overlay(
                        annotated,
                        roi_updater.current,
                        alpha=settings.roi_draw_alpha,
                    )

                _update_live_metrics(job, tracker, tracked, roi_mask)

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + _encode_jpeg(annotated)
                    + b"\r\n"
                )

                for _ in range(skip - 1):
                    if not cap.grab():
                        return
        finally:
            cap.release()
            if roi_state.updater is not None:
                roi_state.updater.stop()

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
