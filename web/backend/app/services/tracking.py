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
