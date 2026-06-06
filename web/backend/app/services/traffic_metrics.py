from __future__ import annotations

from typing import Any

import numpy as np

PCE_WEIGHTS = {
    "motorcycle": 0.30,
    "motor": 0.30,
    "car": 1.00,
    "truck": 2.50,
    "bus": 3.00,
}

OCC_THRESHOLDS = {"low": 15.0, "mid": 30.0, "high": 50.0}
PCE_THRESHOLDS = {"low": 6.0, "mid": 12.0, "high": 20.0}


def _pce_weight(class_name: str) -> float:
    return PCE_WEIGHTS.get(class_name.lower(), 1.0)


def compute_metrics(tracks: list[dict[str, Any]], roi_mask: np.ndarray) -> tuple[float, float]:
    roi_area = float(np.sum(roi_mask > 0))
    if roi_area <= 0:
        return 0.0, 0.0

    pce_total = 0.0
    occupied_mask = np.zeros_like(roi_mask, dtype=np.uint8)

    h, w = roi_mask.shape[:2]
    for track in tracks:
        x1, y1, x2, y2 = (int(track["x1"]), int(track["y1"]), int(track["x2"]), int(track["y2"]))
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = roi_mask[y1:y2, x1:x2]
        occupied_mask[y1:y2, x1:x2] = np.where(crop > 0, 255, occupied_mask[y1:y2, x1:x2])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if 0 <= cy < h and 0 <= cx < w and roi_mask[cy, cx] > 0:
            pce_total += _pce_weight(track.get("class_name", ""))

    occupied_pixels = float(np.sum(occupied_mask > 0))
    occupancy = occupied_pixels / roi_area * 100.0
    return occupancy, pce_total


def decide_alert(occupancy: float, pce_count: float) -> tuple[int, tuple[str, str]]:
    def classify(value: float, thresholds: dict[str, float]) -> int:
        if value < thresholds["low"]:
            return 0
        if value < thresholds["mid"]:
            return 1
        if value < thresholds["high"]:
            return 2
        return 3

    occ_level = classify(occupancy, OCC_THRESHOLDS)
    pce_level = classify(pce_count, PCE_THRESHOLDS)

    alert_matrix = [
        [0, 0, 0, 1],
        [0, 0, 1, 2],
        [0, 1, 2, 3],
        [1, 2, 3, 3],
    ]

    level = alert_matrix[pce_level][occ_level]

    messages = {
        0: ("NORMAL", "Traffic is clear"),
        1: ("BUSY", "Traffic is increasing"),
        2: ("CONGESTED", "Localized congestion"),
        3: ("GRIDLOCK", "Severe congestion"),
    }
    return level, messages[level]
