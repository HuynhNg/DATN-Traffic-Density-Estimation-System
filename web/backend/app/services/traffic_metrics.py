from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_PCE_WEIGHTS = {
    "motorcycle": 0.30,
    "motor": 0.30,
    "car": 1.00,
    "truck": 2.50,
    "bus": 3.00,
}


@dataclass
class TrafficMetricConfig:
    pce_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PCE_WEIGHTS))
    reference_roi_area_ratio: float = 0.35
    min_roi_area_ratio: float = 0.10
    min_roi_scale: float = 0.50
    max_roi_scale: float = 2.50
    min_bbox_overlap: float = 0.10

    occ_low: float = 15.0
    occ_mid: float = 30.0
    occ_high: float = 50.0
    pce_density_low: float = 6.0
    pce_density_mid: float = 12.0
    pce_density_high: float = 20.0
    vehicle_density_low: float = 5.0
    vehicle_density_mid: float = 10.0
    vehicle_density_high: float = 18.0

    occupancy_weight: float = 0.40
    pce_density_weight: float = 0.40
    vehicle_density_weight: float = 0.20
    score_busy: float = 0.75
    score_congested: float = 1.50
    score_gridlock: float = 2.30


def build_traffic_metric_config(settings: Any) -> TrafficMetricConfig:
    return TrafficMetricConfig(
        pce_weights={
            "motorcycle": float(settings.pce_motor),
            "motor": float(settings.pce_motor),
            "car": float(settings.pce_car),
            "truck": float(settings.pce_truck),
            "bus": float(settings.pce_bus),
        },
        reference_roi_area_ratio=float(settings.alert_reference_roi_area_ratio),
        min_roi_area_ratio=float(settings.alert_min_roi_area_ratio),
        min_roi_scale=float(settings.alert_min_roi_scale),
        max_roi_scale=float(settings.alert_max_roi_scale),
        min_bbox_overlap=float(settings.roi_min_bbox_overlap),
        occ_low=float(settings.alert_occ_low),
        occ_mid=float(settings.alert_occ_mid),
        occ_high=float(settings.alert_occ_high),
        pce_density_low=float(settings.alert_pce_density_low),
        pce_density_mid=float(settings.alert_pce_density_mid),
        pce_density_high=float(settings.alert_pce_density_high),
        vehicle_density_low=float(settings.alert_vehicle_density_low),
        vehicle_density_mid=float(settings.alert_vehicle_density_mid),
        vehicle_density_high=float(settings.alert_vehicle_density_high),
        occupancy_weight=float(settings.alert_occupancy_weight),
        pce_density_weight=float(settings.alert_pce_density_weight),
        vehicle_density_weight=float(settings.alert_vehicle_density_weight),
        score_busy=float(settings.alert_score_busy),
        score_congested=float(settings.alert_score_congested),
        score_gridlock=float(settings.alert_score_gridlock),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _pce_weight(class_name: str, config: TrafficMetricConfig) -> float:
    return config.pce_weights.get(class_name.lower(), 1.0)


def compute_metrics(
    tracks: list[dict[str, Any]],
    roi_mask: np.ndarray,
    config: TrafficMetricConfig | None = None,
) -> dict[str, float]:
    metric_config = config or TrafficMetricConfig()
    roi_area = float(np.sum(roi_mask > 0))
    frame_area = float(max(roi_mask.shape[0] * roi_mask.shape[1], 1))
    if roi_area <= 0:
        return {
            "occupancy_pct": 0.0,
            "pce_count": 0.0,
            "pce_density": 0.0,
            "vehicle_density": 0.0,
            "roi_area_ratio": 0.0,
            "roi_scale": 1.0,
        }

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
        overlap_pixels = float(np.sum(crop > 0))
        occupied_mask[y1:y2, x1:x2] = np.where(crop > 0, 255, occupied_mask[y1:y2, x1:x2])

        bbox_area = float(max((x2 - x1) * (y2 - y1), 1))
        overlap_ratio = overlap_pixels / bbox_area
        if overlap_ratio >= metric_config.min_bbox_overlap:
            pce_total += _pce_weight(track.get("class_name", ""), metric_config)

    occupied_pixels = float(np.sum(occupied_mask > 0))
    occupancy = occupied_pixels / roi_area * 100.0
    roi_area_ratio = roi_area / frame_area
    scale_base = max(roi_area_ratio, metric_config.min_roi_area_ratio)
    roi_scale = metric_config.reference_roi_area_ratio / scale_base
    roi_scale = _clamp(roi_scale, metric_config.min_roi_scale, metric_config.max_roi_scale)

    return {
        "occupancy_pct": occupancy,
        "pce_count": pce_total,
        "pce_density": pce_total * roi_scale,
        "vehicle_density": float(len(tracks)) * roi_scale,
        "roi_area_ratio": roi_area_ratio,
        "roi_scale": roi_scale,
    }


def _level(value: float, low: float, mid: float, high: float) -> int:
    if value < low:
        return 0
    if value < mid:
        return 1
    if value < high:
        return 2
    return 3


def _level_score(value: float, low: float, mid: float, high: float) -> float:
    value = max(float(value), 0.0)
    low = max(float(low), 1e-6)
    mid = max(float(mid), low + 1e-6)
    high = max(float(high), mid + 1e-6)

    if value < low:
        return value / low
    if value < mid:
        return 1.0 + (value - low) / (mid - low)
    if value < high:
        return 2.0 + (value - mid) / (high - mid)
    return 3.0


def decide_alert(
    occupancy: float,
    pce_density: float,
    vehicle_density: float,
    config: TrafficMetricConfig | None = None,
) -> tuple[int, tuple[str, str], float, dict[str, int]]:
    metric_config = config or TrafficMetricConfig()
    occ_level = _level(
        occupancy,
        metric_config.occ_low,
        metric_config.occ_mid,
        metric_config.occ_high,
    )
    pce_level = _level(
        pce_density,
        metric_config.pce_density_low,
        metric_config.pce_density_mid,
        metric_config.pce_density_high,
    )
    vehicle_level = _level(
        vehicle_density,
        metric_config.vehicle_density_low,
        metric_config.vehicle_density_mid,
        metric_config.vehicle_density_high,
    )
    occ_score = _level_score(
        occupancy,
        metric_config.occ_low,
        metric_config.occ_mid,
        metric_config.occ_high,
    )
    pce_score = _level_score(
        pce_density,
        metric_config.pce_density_low,
        metric_config.pce_density_mid,
        metric_config.pce_density_high,
    )
    vehicle_score = _level_score(
        vehicle_density,
        metric_config.vehicle_density_low,
        metric_config.vehicle_density_mid,
        metric_config.vehicle_density_high,
    )
    score = (
        metric_config.occupancy_weight * occ_score
        + metric_config.pce_density_weight * pce_score
        + metric_config.vehicle_density_weight * vehicle_score
    )
    if score < metric_config.score_busy:
        level = 0
    elif score < metric_config.score_congested:
        level = 1
    elif score < metric_config.score_gridlock:
        level = 2
    else:
        level = 3

    messages = {
        0: ("NORMAL", "Thông thoáng, lưu lượng xe thấp"),
        1: ("BUSY", "Đông đúc, lưu lượng xe đang tăng"),
        2: ("CONGESTED", "Ùn tắc, mật độ xe cao"),
        3: ("GRIDLOCK", "Tắc nghẽn, giao thông quá tải"),
    }
    return level, messages[level], score, {
        "occupancy": round(occ_score, 3),
        "pce_density": round(pce_score, 3),
        "vehicle_density": round(vehicle_score, 3),
        "bands": {
            "occupancy": occ_level,
            "pce_density": pce_level,
            "vehicle_density": vehicle_level,
        },
    }
