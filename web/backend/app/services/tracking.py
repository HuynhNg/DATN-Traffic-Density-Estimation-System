from __future__ import annotations

import logging
from inspect import signature
from typing import Any

import numpy as np

from app.core.config import settings

try:
    from boxmot.trackers.bytetrack.bytetrack import ByteTrack
except Exception as exc:  # pragma: no cover - optional dependency
    ByteTrack = None
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None

logger = logging.getLogger("app.tracking")


def detections_to_array(detections: list[dict[str, Any]]) -> np.ndarray:
    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    rows = []
    for det in detections:
        rows.append(
            [
                det["x1"],
                det["y1"],
                det["x2"],
                det["y2"],
                det["confidence"],
                det["class_id"],
            ]
        )
    return np.array(rows, dtype=np.float32)


def tracks_to_detections(tracks: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    if tracks is None:
        return []
    tracks = np.asarray(tracks)
    if tracks.size == 0:
        return []
    if tracks.ndim == 1:
        tracks = tracks.reshape(1, -1)

    detections: list[dict[str, Any]] = []
    max_class_id = max(len(class_names) - 1, 0)

    if hasattr(tracks, "id") and hasattr(tracks, "conf") and hasattr(tracks, "cls"):
        for i in range(len(tracks)):
            x1, y1, x2, y2 = tracks.xyxy[i]
            class_id = int(tracks.cls[i])
            class_id = class_id if 0 <= class_id <= max_class_id else 0
            name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            track_id = int(tracks.id[i])
            detections.append(
                {
                    "object_id": track_id,
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": name,
                    "confidence": float(tracks.conf[i]),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
            )
        return detections

    for idx, row in enumerate(tracks):
        x1, y1, x2, y2 = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        track_id = int(row[4]) if row.shape[0] > 4 else idx
        conf = float(row[5]) if row.shape[0] > 5 else 0.0
        class_id = int(row[6]) if row.shape[0] > 6 else 0
        class_id = class_id if 0 <= class_id <= max_class_id else 0
        name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        detections.append(
            {
                "object_id": track_id,
                "track_id": track_id,
                "class_id": class_id,
                "class_name": name,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    return detections


def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    x1 = max(float(a["x1"]), float(b["x1"]))
    y1 = max(float(a["y1"]), float(b["y1"]))
    x2 = min(float(a["x2"]), float(b["x2"]))
    y2 = min(float(a["y2"]), float(b["y2"]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, float(a["x2"] - a["x1"])) * max(0.0, float(a["y2"] - a["y1"]))
    area_b = max(0.0, float(b["x2"] - b["x1"])) * max(0.0, float(b["y2"] - b["y1"]))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _best_iou_match(
    det: dict[str, Any],
    candidates: list[dict[str, Any]],
    used_indexes: set[int],
    min_iou: float,
) -> tuple[int | None, dict[str, Any] | None, float]:
    best_index: int | None = None
    best_candidate: dict[str, Any] | None = None
    best_iou = 0.0
    for index, candidate in enumerate(candidates):
        if index in used_indexes:
            continue
        if candidate.get("class_id") != det.get("class_id"):
            continue
        iou = bbox_iou(det, candidate)
        if iou >= min_iou and iou > best_iou:
            best_index = index
            best_candidate = candidate
            best_iou = iou
    return best_index, best_candidate, best_iou


def repair_tracked_detections(
    detections: list[dict[str, Any]],
    tracked: list[dict[str, Any]],
    last_tracked: list[dict[str, Any]] | None,
    min_iou: float,
) -> list[dict[str, Any]]:
    if not detections:
        return tracked
    if not last_tracked:
        return tracked or detections

    min_iou = max(0.0, min(float(min_iou), 1.0))
    repaired: list[dict[str, Any]] = []
    used_tracked: set[int] = set()
    used_last: set[int] = set()

    for det in detections:
        tracked_index, tracked_match, _tracked_iou = _best_iou_match(
            det,
            tracked,
            used_tracked,
            min_iou,
        )
        last_index, last_match, _last_iou = _best_iou_match(
            det,
            last_tracked,
            used_last,
            min_iou,
        )

        if tracked_match is not None and tracked_index is not None:
            candidate = dict(tracked_match)
            used_tracked.add(tracked_index)
            if last_match is not None and last_index is not None:
                previous_track_id = last_match.get("track_id")
                if previous_track_id is not None:
                    candidate["track_id"] = previous_track_id
                    candidate["object_id"] = previous_track_id
                used_last.add(last_index)
            repaired.append(candidate)
            continue

        if last_match is not None and last_index is not None:
            previous_track_id = last_match.get("track_id")
            candidate = dict(det)
            if previous_track_id is not None:
                candidate["track_id"] = previous_track_id
                candidate["object_id"] = previous_track_id
            used_last.add(last_index)
            repaired.append(candidate)
            continue

        repaired.append(dict(det))

    return repaired


class ByteTrackWrapper:
    def __init__(self, frame_rate: int) -> None:
        if ByteTrack is None:
            raise RuntimeError(
                "boxmot is not available; install it to enable ByteTrack.",
                _IMPORT_ERROR,
            )
        params = signature(ByteTrack).parameters
        kwargs: dict[str, Any] = {}
        if "track_thresh" in params:
            kwargs["track_thresh"] = settings.bytetrack_conf_high
        if "match_thresh" in params:
            kwargs["match_thresh"] = settings.bytetrack_iou_high
        if "track_buffer" in params:
            kwargs["track_buffer"] = settings.bytetrack_track_buffer
        if "frame_rate" in params:
            kwargs["frame_rate"] = frame_rate
        if "min_box_area" in params:
            kwargs["min_box_area"] = settings.bytetrack_min_box_area
        self._tracker = ByteTrack(**kwargs)
        logger.info("ByteTrack initialized with %s", kwargs)

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        return self._tracker.update(detections, frame)
