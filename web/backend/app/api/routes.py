from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.core.config import settings
from app.services.adaptive_roi import (
    ROIConfig,
    ROIResult,
    ROIUpdater,
    apply_roi_to_frame,
    build_roi_config_from_settings,
    compute_adaptive_roi,
    draw_roi_overlay,
    filter_detections_by_roi_xyxy,
    normalize_roi_payload,
    roi_box_from_result,
    roi_from_payload,
    roi_polygon_from_result,
    remap_detections_to_original,
)
from app.services.analytics import AnalyticsTracker
from app.services.frame_utils import resize_to_max
from app.services.inference import InferenceEngine
from app.services.metrics_export import build_metrics_workbook
from app.services.renderer import render_boxes
from app.services.tracking import (
    ByteTrackWrapper,
    detections_to_array,
    repair_tracked_detections,
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
AVG_WINDOW_SECONDS = {"minute": 60.0, "hour": 3600.0}
UPLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass
class StreamROIState:
    config: ROIConfig | None
    updater: ROIUpdater | None = None
    calibration_frames: list[np.ndarray] | None = None
    executor: ThreadPoolExecutor | None = None
    future: Future[ROIResult] | None = None
    failed: bool = False

    @classmethod
    def from_settings(cls) -> "StreamROIState":
        if not settings.roi_enabled:
            return cls(config=None)

        return cls(
            config=build_roi_config_from_settings(settings),
            calibration_frames=[],
            executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-roi"),
        )

    def stop(self) -> None:
        if self.updater is not None:
            self.updater.stop()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)


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


def _save_upload(file: UploadFile, max_bytes: int | None = None) -> str:
    os.makedirs(settings.save_dir, exist_ok=True)
    temp_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name if file.filename else "upload.bin"
    input_path = os.path.join(settings.save_dir, f"{temp_id}_{safe_name}")

    written = 0
    try:
        with open(input_path, "wb") as f:
            while True:
                chunk = file.file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Video file is too large. "
                            f"Maximum allowed size is {settings.max_video_upload_mb} MB."
                        ),
                    )
                f.write(chunk)
    except Exception:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise

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


def _start_video_processing(
    job: VideoJob,
    labels: bool,
    conf: bool,
) -> None:
    if not job.input_path:
        raise HTTPException(status_code=400, detail="Job has no input video")
    if job.status == "processing":
        return
    if job.status == "done" and job.result_path:
        return

    job.status = "queued"
    asyncio.create_task(
        asyncio.to_thread(
            process_video,
            router.engine,
            job,
            job.input_path,
            labels,
            conf,
        )
    )


def _validate_roi_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        roi_payload = normalize_roi_payload(payload)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid ROI payload")

    if roi_payload["type"] == "box" and (roi_payload["w"] <= 0 or roi_payload["h"] <= 0):
        raise HTTPException(status_code=400, detail="ROI width and height must be positive")
    return roi_payload


def _normalize_avg_window(avg_window: str) -> str:
    window = avg_window.lower().strip()
    if window in AVG_WINDOW_SECONDS or window == "all":
        return window
    raise HTTPException(
        status_code=400,
        detail="avg_window must be one of: minute, hour, all",
    )


def _track_ids_from_row(row: dict[str, Any]) -> set[int]:
    track_ids = row.get("track_ids")
    if not isinstance(track_ids, list):
        return set()
    return {int(track_id) for track_id in track_ids if track_id is not None}


def _total_vehicles_for_window(
    history: list[dict[str, Any]],
    avg_window: str,
    now: float | None = None,
) -> int:
    if not history:
        return 0

    rows = sorted(history, key=lambda row: float(row.get("timestamp", 0.0)))
    current_ts = now if now is not None else float(rows[-1].get("timestamp", time.time()))
    if avg_window == "all":
        window_rows = rows
    else:
        cutoff = current_ts - AVG_WINDOW_SECONDS[avg_window]
        window_rows = [
            row for row in rows if float(row.get("timestamp", 0.0)) >= cutoff
        ]

    unique_track_ids: set[int] = set()
    for row in window_rows:
        unique_track_ids.update(_track_ids_from_row(row))
    if unique_track_ids:
        return len(unique_track_ids)

    # Without tracker IDs, the backend cannot know whether two detections across
    # frames are the same physical vehicle. Fall back to the latest frame count.
    return int(rows[-1].get("objects_in_frame", 0))


def _bucket_label(bucket_start: float, avg_window: str) -> str:
    if avg_window == "hour":
        return time.strftime("%H:00", time.localtime(bucket_start))
    return time.strftime("%H:%M", time.localtime(bucket_start))


def _vehicle_series_for_window(
    history: list[dict[str, Any]],
    avg_window: str,
    limit: int = 60,
) -> list[dict[str, Any]]:
    if not history:
        return []

    rows = sorted(history, key=lambda row: float(row.get("timestamp", 0.0)))
    bucket_seconds = 3600 if avg_window == "hour" else 60
    buckets: dict[int, dict[str, Any]] = {}

    for row in rows:
        row_ts = float(row.get("timestamp", 0.0))
        bucket_start = int(row_ts // bucket_seconds) * bucket_seconds
        bucket = buckets.setdefault(
            bucket_start,
            {
                "track_ids": set(),
                "fallback_count": 0,
            },
        )
        track_ids = _track_ids_from_row(row)
        bucket["track_ids"].update(track_ids)
        bucket["fallback_count"] = max(
            int(bucket["fallback_count"]),
            int(row.get("objects_in_frame", 0)),
        )

    series: list[dict[str, Any]] = []
    for bucket_start in sorted(buckets)[-limit:]:
        bucket = buckets[bucket_start]
        track_ids = bucket["track_ids"]
        count = len(track_ids) if track_ids else int(bucket["fallback_count"])
        series.append(
            {
                "t": _bucket_label(float(bucket_start), avg_window),
                "count": count,
            }
        )
    return series


def _flow_count_for_window(
    events: list[dict[str, Any]],
    event_type: str,
    avg_window: str,
    now: float | None = None,
) -> int:
    if not events:
        return 0

    current_ts = now if now is not None else float(events[-1].get("timestamp", time.time()))
    if avg_window == "all":
        window_events = events
    else:
        cutoff = current_ts - AVG_WINDOW_SECONDS[avg_window]
        window_events = [
            event for event in events if float(event.get("timestamp", 0.0)) >= cutoff
        ]
    return sum(1 for event in window_events if event.get("type") == event_type)


def _live_metrics_for_window(job: VideoJob, avg_window: str) -> dict[str, Any] | None:
    if job.live_metrics is None:
        return None

    metrics = dict(job.live_metrics)
    history = job.live_history or []
    flow_events = job.live_flow_events or []
    now = (
        float(history[-1].get("timestamp", time.time()))
        if history
        else time.time()
    )
    total_vehicles = _total_vehicles_for_window(history, avg_window)
    metrics["total_vehicles"] = total_vehicles
    metrics["avg_objects"] = total_vehicles
    left_to_right = _flow_count_for_window(flow_events, "left_to_right", avg_window, now)
    right_to_left = _flow_count_for_window(flow_events, "right_to_left", avg_window, now)
    metrics["vehicles_left_to_right"] = left_to_right
    metrics["vehicles_right_to_left"] = right_to_left
    metrics["vehicles_in"] = left_to_right
    metrics["vehicles_out"] = right_to_left
    metrics["avg_window"] = avg_window
    return metrics


def _track_center(track: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(track["x1"]) + float(track["x2"])) / 2.0,
        (float(track["y1"]) + float(track["y2"])) / 2.0,
    )


def _update_flow_events(
    job: VideoJob,
    tracked: list[dict[str, Any]],
    now: float,
    frame_width: int,
) -> None:
    if job.live_track_states is None:
        job.live_track_states = {}
    if job.live_flow_events is None:
        job.live_flow_events = []

    active_ids: set[int] = set()
    for track in tracked:
        raw_track_id = track.get("track_id")
        if raw_track_id is None:
            continue

        track_id = int(raw_track_id)
        active_ids.add(track_id)
        center = _track_center(track)
        state = job.live_track_states.get(track_id)
        if state is None or not state.get("active", False):
            state = {
                "first_seen": now,
                "first_center": center,
                "active": True,
                "direction_counted": False,
            }

        first_center = state.get("first_center") or center
        dx = center[0] - float(first_center[0])
        min_dx = max(float(frame_width) * settings.flow_direction_min_dx_ratio, 1.0)
        if not state.get("direction_counted", False) and abs(dx) >= min_dx:
            event_type = "left_to_right" if dx > 0 else "right_to_left"
            job.live_flow_events.append(
                {
                    "timestamp": round(now, 3),
                    "time": time.strftime("%H:%M:%S", time.localtime(now)),
                    "type": event_type,
                    "track_id": track_id,
                    "class_name": track.get("class_name", ""),
                    "x": round(center[0], 2),
                    "y": round(center[1], 2),
                    "dx": round(dx, 2),
                }
            )
            state["direction_counted"] = True

        state.update(
            {
                "last_seen": now,
                "last_center": center,
                "class_name": track.get("class_name", ""),
                "active": True,
            }
        )
        job.live_track_states[track_id] = state

    timeout = max(float(settings.flow_exit_timeout_sec), 0.0)
    for track_id, state in list(job.live_track_states.items()):
        if track_id in active_ids or not state.get("active", False):
            continue

        last_seen = float(state.get("last_seen", now))
        if now - last_seen < timeout:
            continue

        state["active"] = False
        job.live_track_states[track_id] = state


def _update_stream_roi_state(
    state: StreamROIState,
    frame: np.ndarray,
    job: VideoJob,
) -> ROIUpdater | None:
    if state.config is None or state.failed:
        return None
    if state.updater is not None:
        return state.updater

    if state.future is not None:
        if not state.future.done():
            return None
        try:
            roi = state.future.result()
            state.updater = ROIUpdater(state.config, roi)
            state.updater.start()
            job.roi = roi_polygon_from_result(roi)
            job.roi_box = roi_box_from_result(roi)
            job.roi_source = "auto"
            logger.info("Adaptive ROI enabled for stream job %s", job.job_id)
        except Exception:
            state.failed = True
            logger.exception("Adaptive ROI calibration failed for stream job %s", job.job_id)
        finally:
            state.future = None
        return None

    if state.calibration_frames is None:
        state.calibration_frames = []
    state.calibration_frames.append(frame.copy())

    if len(state.calibration_frames) < state.config.calib_frames:
        return None

    if state.executor is None:
        state.failed = True
        state.calibration_frames = None
        logger.error("Adaptive ROI executor is not available for stream job %s", job.job_id)
        return None

    frames = state.calibration_frames
    state.calibration_frames = None
    state.future = state.executor.submit(compute_adaptive_roi, frames, state.config)
    logger.info("Adaptive ROI calibration started for stream job %s", job.job_id)

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


def _format_detection_for_log(det: dict[str, Any]) -> str:
    track_id = det.get("track_id")
    track_text = f" track={track_id}" if track_id is not None else ""
    return (
        f"{det.get('class_name', det.get('class_id'))}"
        f"(id={det.get('class_id')})"
        f" conf={float(det.get('confidence', 0.0)):.3f}"
        f" bbox=({int(det.get('x1', 0))},{int(det.get('y1', 0))},"
        f"{int(det.get('x2', 0))},{int(det.get('y2', 0))})"
        f"{track_text}"
    )


def _log_detections(
    stage: str,
    job_id: str,
    frame_idx: int,
    detections: list[dict[str, Any]],
) -> None:
    if not settings.log_detections:
        return
    details = " | ".join(_format_detection_for_log(det) for det in detections)
    logger.info(
        "bbox_debug stage=%s job=%s frame=%d count=%d %s",
        stage,
        job_id,
        frame_idx,
        len(detections),
        details,
    )


def _detect_stream_frame(
    frame: np.ndarray,
    engine: InferenceEngine,
    roi: ROIResult | None,
    job_id: str,
    frame_idx: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if roi is None:
        roi_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        detections, _ = engine.detect(frame)
        _log_detections("raw_no_roi", job_id, frame_idx, detections)
        return detections, roi_mask

    processed, offset = apply_roi_to_frame(frame, roi, mode=settings.roi_mode)
    detections, _ = engine.detect(processed)
    detections = remap_detections_to_original(detections, offset)
    _log_detections("raw_before_roi_filter", job_id, frame_idx, detections)
    detections = filter_detections_by_roi_xyxy(
        detections,
        roi,
        anchor=settings.roi_anchor,
        min_bbox_overlap=settings.roi_min_bbox_overlap,
    )
    _log_detections("after_roi_filter", job_id, frame_idx, detections)
    return detections, roi.combined_mask


def _resolve_stream_roi(
    job: VideoJob,
    frame_shape: tuple[int, int],
    roi_updater: ROIUpdater | None,
) -> ROIResult | None:
    if job.manual_roi is not None:
        return roi_from_payload(frame_shape, job.manual_roi)
    if job.manual_roi_box is not None:
        return roi_from_payload(frame_shape, job.manual_roi_box)
    if roi_updater is not None:
        return roi_updater.current
    return None


def _track_stream_detections(
    detections: list[dict[str, Any]],
    frame: np.ndarray,
    bytetracker: ByteTrackWrapper | None,
    run_tracker: bool,
    last_tracked: list[dict[str, Any]] | None,
    job_id: str,
    frame_idx: int,
) -> list[dict[str, Any]]:
    if bytetracker is None:
        _log_detections("tracking_disabled", job_id, frame_idx, detections)
        return detections
    if not run_tracker:
        tracked = last_tracked or detections
        _log_detections("tracking_reused_last", job_id, frame_idx, tracked)
        return tracked

    det_array = detections_to_array(detections)
    tracked = tracks_to_detections(
        bytetracker.update(det_array, frame),
        settings.class_names,
    )
    if settings.bytetrack_repair_enabled:
        tracked = repair_tracked_detections(
            detections,
            tracked,
            last_tracked,
            settings.bytetrack_repair_iou,
        )
    _log_detections("after_bytetrack", job_id, frame_idx, tracked)
    return tracked


def _update_live_metrics(
    job: VideoJob,
    tracker: AnalyticsTracker,
    tracked: list[dict[str, Any]],
    roi_mask: np.ndarray,
) -> None:
    metrics = tracker.update(len(tracked))
    occupancy, pce_total = compute_metrics(tracked, roi_mask)
    alert_level, alert_message = decide_alert(occupancy, pce_total)
    now = time.time()
    _update_flow_events(job, tracked, now, roi_mask.shape[1])
    flow_events = job.live_flow_events or []

    metric_row = {
        "timestamp": round(now, 3),
        "time": time.strftime("%H:%M:%S", time.localtime(now)),
        "fps": metrics["fps"],
        "objects_in_frame": metrics["objects_in_frame"],
        "track_ids": sorted(
            {
                int(track["track_id"])
                for track in tracked
                if track.get("track_id") is not None
            }
        ),
        "track_classes": [
            {
                "track_id": int(track["track_id"]),
                "class_name": track.get("class_name", ""),
            }
            for track in tracked
            if track.get("track_id") is not None
        ],
        "occupancy_pct": round(occupancy, 2),
        "pce_count": round(pce_total, 2),
        "alert_level": alert_level,
        "alert_label": alert_message[0],
    }

    if job.live_history is None:
        job.live_history = []
    job.live_history.append(metric_row)

    total_vehicles = _total_vehicles_for_window(job.live_history, "minute", now)
    left_to_right = _flow_count_for_window(flow_events, "left_to_right", "minute", now)
    right_to_left = _flow_count_for_window(flow_events, "right_to_left", "minute", now)
    job.live_metrics = {
        **metrics,
        "avg_objects": total_vehicles,
        "total_vehicles": total_vehicles,
        "vehicles_left_to_right": left_to_right,
        "vehicles_right_to_left": right_to_left,
        "vehicles_in": left_to_right,
        "vehicles_out": right_to_left,
        "avg_window": "minute",
        "occupancy_pct": round(occupancy, 2),
        "pce_count": round(pce_total, 2),
        "alert_level": alert_level,
        "alert_label": alert_message[0],
        "alert_message": alert_message[1],
    }

    if job.live_series is None:
        job.live_series = []
    job.live_series.append(
        {"t": metric_row["time"], "count": metrics["objects_in_frame"]}
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
    max_bytes = max(int(settings.max_video_upload_mb), 1) * 1024 * 1024
    input_path = _save_upload(file, max_bytes=max_bytes)
    fps, total_frames = _read_video_metadata(input_path)

    job = router.video_jobs.create()  # type: ignore[attr-defined]
    job.fps = fps
    job.total_frames = total_frames
    job.input_path = input_path

    if settings.auto_process_video:
        _start_video_processing(job, labels, conf)

    return {"job_id": job.job_id, "fps": fps, "total_frames": total_frames}


@router.get("/video/{job_id}")
# Fetch processing status and metrics for a video job.
async def get_video_status(
    job_id: str,
    avg_window: str = Query("minute"),
) -> dict[str, Any]:
    job = router.video_jobs.get(job_id)  # type: ignore[attr-defined]
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    normalized_window = _normalize_avg_window(avg_window)

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "result_url": f"/api/video/{job_id}/result" if job.result_path else None,
        "analytics": job.analytics,
        "fps": job.fps,
        "total_frames": job.total_frames,
        "live_metrics": _live_metrics_for_window(job, normalized_window),
        "live_series": _vehicle_series_for_window(
            job.live_history or [],
            normalized_window,
        ),
        "roi": job.manual_roi or job.roi,
        "roi_box": job.manual_roi_box or job.roi_box,
        "roi_source": "manual" if job.manual_roi or job.manual_roi_box else job.roi_source,
    }


@router.get("/video/{job_id}/metrics/export")
# Export realtime metrics history as an Excel workbook.
async def export_video_metrics(
    job_id: str,
    avg_window: str = Query("minute"),
) -> Response:
    job = router.video_jobs.get(job_id)  # type: ignore[attr-defined]
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    normalized_window = _normalize_avg_window(avg_window)
    history = list(job.live_history or [])
    latest_metrics = _live_metrics_for_window(job, normalized_window)
    if history and latest_metrics is not None:
        history[-1] = {
            **history[-1],
            "total_vehicles": latest_metrics["total_vehicles"],
            "vehicles_left_to_right": latest_metrics["vehicles_left_to_right"],
            "vehicles_right_to_left": latest_metrics["vehicles_right_to_left"],
            "vehicles_in": latest_metrics["vehicles_left_to_right"],
            "vehicles_out": latest_metrics["vehicles_right_to_left"],
        }

    content = build_metrics_workbook(
        job.job_id,
        history,
        normalized_window,
        job.live_flow_events or [],
    )
    filename = f"traffic_metrics_{job.job_id}.xlsx"
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/video/{job_id}/process")
# Start offline video processing for a previously uploaded video.
async def start_video_processing(
    job_id: str,
    labels: bool = True,
    conf: bool = True,
) -> dict[str, Any]:
    job = router.video_jobs.get(job_id)  # type: ignore[attr-defined]
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    _start_video_processing(job, labels, conf)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "result_url": f"/api/video/{job_id}/result" if job.result_path else None,
    }


@router.post("/video/{job_id}/roi")
# Set a manual normalized ROI for a video job.
async def set_video_roi(
    job_id: str,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    job = router.video_jobs.get(job_id)  # type: ignore[attr-defined]
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    roi_payload = _validate_roi_payload(payload)
    job.manual_roi = roi_payload
    job.manual_roi_box = roi_payload if roi_payload["type"] == "box" else None
    return {
        "job_id": job.job_id,
        "roi": roi_payload,
        "roi_box": job.manual_roi_box,
        "roi_source": "manual",
    }


@router.delete("/video/{job_id}/roi")
# Reset manual ROI and let the auto ROI apply again when available.
async def reset_video_roi(job_id: str) -> dict[str, Any]:
    job = router.video_jobs.get(job_id)  # type: ignore[attr-defined]
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.manual_roi_box = None
    job.manual_roi = None
    return {
        "job_id": job.job_id,
        "roi": job.roi,
        "roi_box": job.roi_box,
        "roi_source": job.roi_source,
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
    target_fps: int = 30,
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
        safe_target = max(1, min(int(target_fps or 30), 60))
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
                    job,
                )
                current_roi = _resolve_stream_roi(
                    job,
                    resized.shape[:2],
                    roi_updater,
                )
                detections, roi_mask = _detect_stream_frame(
                    resized,
                    engine,
                    current_roi,
                    job.job_id,
                    stream_idx,
                )
                tracked = _track_stream_detections(
                    detections,
                    resized,
                    bytetracker,
                    run_tracker,
                    last_tracked,
                    job.job_id,
                    stream_idx,
                )
                if run_tracker:
                    last_tracked = tracked

                annotated = render_boxes(resized, tracked, labels, conf)
                if current_roi is not None and settings.roi_draw:
                    annotated = draw_roi_overlay(
                        annotated,
                        current_roi,
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
            roi_state.stop()

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
