from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
# Configuration for adaptive ROI estimation.
class ROIConfig:
    calib_frames: int = 100
    min_area_ratio: float = 0.05
    otsu_bias: float = 1.05
    morph_kernel_size: int = 15
    farneback_scale: float = 0.5
    max_zones: int = 2


@dataclass
# Result of ROI computation, including masks and convex hulls.
class ROIResult:
    masks: list[np.ndarray]
    hulls: list[np.ndarray]
    combined_mask: np.ndarray
    frame_shape: tuple[int, int]
    timestamp: float = field(default_factory=time.time)

    # Number of ROI zones detected.
    def n_zones(self) -> int:
        return len(self.masks)

    # Time since this ROI result was generated.
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


def normalize_roi_box(box: dict[str, Any]) -> dict[str, float]:
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("w", 1.0))
    h = float(box.get("h", 1.0))

    x = max(0.0, min(x, 0.99))
    y = max(0.0, min(y, 0.99))
    w = max(0.01, min(w, 1.0 - x))
    h = max(0.01, min(h, 1.0 - y))

    return {"x": x, "y": y, "w": w, "h": h}


def _normalize_roi_point(point: dict[str, Any]) -> dict[str, float]:
    x = max(0.0, min(float(point.get("x", 0.0)), 1.0))
    y = max(0.0, min(float(point.get("y", 0.0)), 1.0))
    return {"x": x, "y": y}


def normalize_roi_polygon(points: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(points, list):
        raise ValueError("ROI polygon points must be a list")

    normalized: list[dict[str, float]] = []
    seen: set[tuple[float, float]] = set()
    for raw_point in points:
        if not isinstance(raw_point, dict):
            raise ValueError("ROI polygon point must be an object")
        point = _normalize_roi_point(raw_point)
        key = (round(point["x"], 6), round(point["y"], 6))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(point)

    if len(normalized) < 3:
        raise ValueError("ROI polygon needs at least 3 unique points")

    return normalized


def normalize_roi_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("ROI payload must be an object")

    roi_type = payload.get("type")
    if roi_type == "polygon":
        return {
            "type": "polygon",
            "points": normalize_roi_polygon(payload.get("points", [])),
        }

    if roi_type in (None, "box"):
        box = normalize_roi_box(payload)
        return {"type": "box", **box}

    raise ValueError(f"Unsupported ROI type: {roi_type!r}")


def roi_box_from_result(roi: ROIResult) -> dict[str, float]:
    coords = cv2.findNonZero(roi.combined_mask)
    h, w = roi.frame_shape
    if coords is None:
        return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    x, y, box_w, box_h = cv2.boundingRect(coords)
    return normalize_roi_box(
        {
            "x": x / max(w, 1),
            "y": y / max(h, 1),
            "w": box_w / max(w, 1),
            "h": box_h / max(h, 1),
        }
    )


def _polygon_payload_from_box(box: dict[str, float]) -> dict[str, Any]:
    x = box["x"]
    y = box["y"]
    x2 = x + box["w"]
    y2 = y + box["h"]
    return {
        "type": "polygon",
        "points": [
            {"x": x, "y": y},
            {"x": x2, "y": y},
            {"x": x2, "y": y2},
            {"x": x, "y": y2},
        ],
    }


def _simplify_hull_points(hull: np.ndarray, max_points: int = 12) -> np.ndarray:
    points = hull.reshape(-1, 2).astype(np.int32)
    if len(points) <= max_points:
        return points

    contour = points.reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    for ratio in (0.005, 0.01, 0.02, 0.035, 0.05):
        approx = cv2.approxPolyDP(contour, ratio * perimeter, True).reshape(-1, 2)
        if 3 <= len(approx) <= max_points:
            return approx.astype(np.int32)

    return points[:max_points]


def roi_polygon_from_result(roi: ROIResult) -> dict[str, Any]:
    coords = cv2.findNonZero(roi.combined_mask)
    if coords is None:
        return _polygon_payload_from_box({"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})

    hull = cv2.convexHull(coords)
    simplified = _simplify_hull_points(hull)
    if len(simplified) < 3:
        return _polygon_payload_from_box(roi_box_from_result(roi))

    h, w = roi.frame_shape
    x_denom = max(w - 1, 1)
    y_denom = max(h - 1, 1)
    points = [
        {
            "x": max(0.0, min(float(x) / x_denom, 1.0)),
            "y": max(0.0, min(float(y) / y_denom, 1.0)),
        }
        for x, y in simplified
    ]
    return {"type": "polygon", "points": normalize_roi_polygon(points)}


def roi_from_box(
    frame_shape: tuple[int, int],
    box: dict[str, Any],
) -> ROIResult:
    h, w = frame_shape
    roi_box = normalize_roi_box(box)
    x1 = int(round(roi_box["x"] * w))
    y1 = int(round(roi_box["y"] * h))
    x2 = int(round((roi_box["x"] + roi_box["w"]) * w))
    y2 = int(round((roi_box["y"] + roi_box["h"]) * h))

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    hull = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    return ROIResult(
        masks=[mask],
        hulls=[hull],
        combined_mask=mask,
        frame_shape=(h, w),
    )


def roi_from_polygon(
    frame_shape: tuple[int, int],
    points: list[dict[str, Any]],
) -> ROIResult:
    h, w = frame_shape
    polygon = normalize_roi_polygon(points)
    pixel_points = np.array(
        [
            [
                int(round(point["x"] * max(w - 1, 1))),
                int(round(point["y"] * max(h - 1, 1))),
            ]
            for point in polygon
        ],
        dtype=np.int32,
    )
    pixel_points[:, 0] = np.clip(pixel_points[:, 0], 0, max(w - 1, 0))
    pixel_points[:, 1] = np.clip(pixel_points[:, 1], 0, max(h - 1, 0))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pixel_points], 255)
    hull = cv2.convexHull(pixel_points)
    return ROIResult(
        masks=[mask],
        hulls=[hull],
        combined_mask=mask,
        frame_shape=(h, w),
    )


def roi_from_payload(
    frame_shape: tuple[int, int],
    payload: dict[str, Any],
) -> ROIResult:
    roi_payload = normalize_roi_payload(payload)
    if roi_payload["type"] == "polygon":
        return roi_from_polygon(frame_shape, roi_payload["points"])
    return roi_from_box(frame_shape, roi_payload)


# Build ROI config from global settings.
def build_roi_config_from_settings(settings: Any) -> ROIConfig:
    return ROIConfig(
        calib_frames=settings.roi_calib_frames,
        min_area_ratio=settings.roi_min_area_ratio,
        otsu_bias=settings.roi_otsu_bias,
        morph_kernel_size=settings.roi_morph_kernel_size,
        farneback_scale=settings.roi_farneback_scale,
        max_zones=settings.roi_max_zones,
    )


# Capture initial frames and compute ROI for calibration.
def calibrate_roi_from_capture(
    cap: cv2.VideoCapture,
    config: ROIConfig,
) -> ROIResult | None:
    frames: list[np.ndarray] = []
    while len(frames) < config.calib_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if len(frames) < 2:
        return None
    return compute_adaptive_roi(frames, config)


# Estimate motion magnitude map using optical flow.
def _compute_motion_map(
    frames: list[np.ndarray],
    scale: float,
) -> np.ndarray:
    h_orig, w_orig = frames[0].shape[:2]
    h_s = max(1, int(h_orig * scale))
    w_s = max(1, int(w_orig * scale))

    magnitude_acc = np.zeros((h_s, w_s), dtype=np.float32)
    count = 0

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (w_s, h_s))

    for frame in frames[1:]:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, (w_s, h_s))

        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray,
            next=curr_gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        mag, _ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        magnitude_acc += mag
        count += 1
        prev_gray = curr_gray

    if count == 0:
        logger.warning("Not enough frames to build motion map (need >= 2).")
        return np.zeros((h_orig, w_orig), dtype=np.float32)

    motion_map_small = magnitude_acc / count
    return cv2.resize(motion_map_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)


# Convert motion map to a clean binary mask.
def _binarize_and_clean(
    motion_map: np.ndarray,
    otsu_bias: float,
    kernel_size: int,
) -> np.ndarray:
    motion_8u = cv2.normalize(
        motion_map,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    otsu_thresh, _ = cv2.threshold(
        motion_8u,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    adjusted_thresh = min(int(otsu_thresh * otsu_bias), 254)

    _, binary = cv2.threshold(motion_8u, adjusted_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


# Extract ROI zones from a binary mask.
def _extract_roi_zones(
    binary: np.ndarray,
    min_area_ratio: float,
    max_zones: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    h, w = binary.shape
    frame_area = h * w
    min_zone_area = frame_area * min_area_ratio

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    valid_components: list[tuple[int, int]] = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_zone_area:
            valid_components.append((lbl, area))

    if not valid_components:
        full_mask = np.ones((h, w), dtype=np.uint8) * 255
        hull = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        return [full_mask], [hull]

    valid_components.sort(key=lambda x: x[1], reverse=True)
    selected = valid_components[:max_zones]

    masks: list[np.ndarray] = []
    hulls: list[np.ndarray] = []

    for lbl, _area in selected:
        component_mask = np.where(labels == lbl, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        hull_pts = cv2.convexHull(contours[0])

        zone_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(zone_mask, hull_pts, 255)

        masks.append(zone_mask)
        hulls.append(hull_pts)

    return masks, hulls


# Compute ROI masks from a list of frames.
def compute_adaptive_roi(
    frames: list[np.ndarray],
    config: ROIConfig | None = None,
) -> ROIResult:
    if config is None:
        config = ROIConfig()

    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(frames)}.")

    h, w = frames[0].shape[:2]
    motion_map = _compute_motion_map(frames, scale=config.farneback_scale)
    binary = _binarize_and_clean(
        motion_map,
        otsu_bias=config.otsu_bias,
        kernel_size=config.morph_kernel_size,
    )
    masks, hulls = _extract_roi_zones(
        binary,
        min_area_ratio=config.min_area_ratio,
        max_zones=config.max_zones,
    )

    combined = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        combined = cv2.bitwise_or(combined, mask)

    return ROIResult(
        masks=masks,
        hulls=hulls,
        combined_mask=combined,
        frame_shape=(h, w),
    )


# Keep detections whose anchor point or bbox overlap falls inside the ROI mask.
def filter_detections_by_roi_xyxy(
    detections: list[dict[str, Any]],
    roi: ROIResult,
    anchor: str = "bottom_center",
    min_bbox_overlap: float = 0.0,
) -> list[dict[str, Any]]:
    h, w = roi.frame_shape
    mask = roi.combined_mask
    valid: list[dict[str, Any]] = []
    min_bbox_overlap = max(0.0, min(float(min_bbox_overlap), 1.0))

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            continue

        cx = int((x1 + x2) / 2)

        if anchor == "bottom_center":
            px, py = cx, int(y2)
        else:
            px, py = cx, int((y1 + y2) / 2)

        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))

        if mask[py, px] > 0:
            valid.append(det)
            continue

        if min_bbox_overlap <= 0:
            continue

        bbox_area = float((x2 - x1) * (y2 - y1))
        roi_pixels_in_bbox = float(np.sum(mask[y1:y2, x1:x2] > 0))
        if bbox_area > 0 and roi_pixels_in_bbox / bbox_area >= min_bbox_overlap:
            valid.append(det)

    return valid


# Apply ROI mask or crop to a frame before inference.
def apply_roi_to_frame(
    frame: np.ndarray,
    roi: ROIResult,
    mode: str = "mask",
) -> tuple[np.ndarray, tuple[int, int]]:
    if mode == "mask":
        masked = cv2.bitwise_and(frame, frame, mask=roi.combined_mask)
        return masked, (0, 0)

    if mode == "crop":
        coords = cv2.findNonZero(roi.combined_mask)
        if coords is None:
            return frame, (0, 0)

        x0, y0, crop_w, crop_h = cv2.boundingRect(coords)
        pad = 10
        h, w = frame.shape[:2]
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x0 + crop_w + 2 * pad)
        y1 = min(h, y0 + crop_h + 2 * pad)

        cropped = frame[y0:y1, x0:x1]
        return cropped, (x0, y0)

    raise ValueError(f"mode must be 'mask' or 'crop', got {mode!r}")


# Translate detection boxes back to original frame coordinates.
def remap_detections_to_original(
    detections: list[dict[str, Any]],
    crop_offset: tuple[int, int],
) -> list[dict[str, Any]]:
    ox, oy = crop_offset
    if ox == 0 and oy == 0:
        return detections

    remapped: list[dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        remapped.append(
            {
                **det,
                "x1": x1 + ox,
                "y1": y1 + oy,
                "x2": x2 + ox,
                "y2": y2 + oy,
            }
        )
    return remapped


# Draw semi-transparent ROI overlays on a frame.
def draw_roi_overlay(
    frame: np.ndarray,
    roi: ROIResult,
    alpha: float = 0.25,
    colors: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    if colors is None:
        colors = [(0, 200, 100), (200, 50, 200), (0, 180, 255)]

    overlay = frame.copy()
    output = frame.copy()

    for idx, (mask, hull) in enumerate(zip(roi.masks, roi.hulls)):
        color = colors[idx % len(colors)]
        overlay[mask > 0] = color
        cv2.polylines(output, [hull], isClosed=True, color=color, thickness=2)

    return cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)


# Background worker that keeps ROI up to date.
class ROIUpdater:
    # Static ROI holder. Recalibration is intentionally disabled after startup.
    def __init__(self, config: ROIConfig, initial_roi: ROIResult) -> None:
        self._config = config
        self._current_roi: ROIResult = initial_roi
        self.update_count = 0
        self.last_update_time: float | None = initial_roi.timestamp

    @property
    # Thread-safe accessor for the current ROI.
    def current(self) -> ROIResult:
        return self._current_roi

    # Add a frame to the rolling buffer.
    def push_frame(self, frame: np.ndarray) -> None:
        return None

    # Start background ROI update thread.
    def start(self) -> None:
        return None

    # Stop background ROI update thread.
    def stop(self, timeout: float = 5.0) -> None:
        return None
