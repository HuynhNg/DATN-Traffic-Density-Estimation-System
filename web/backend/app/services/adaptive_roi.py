from __future__ import annotations

import logging
import threading
import time
from collections import deque
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
    update_interval_sec: float = 600.0
    rolling_window_size: int = 100
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


# Build ROI config from global settings.
def build_roi_config_from_settings(settings: Any) -> ROIConfig:
    return ROIConfig(
        calib_frames=settings.roi_calib_frames,
        min_area_ratio=settings.roi_min_area_ratio,
        otsu_bias=settings.roi_otsu_bias,
        morph_kernel_size=settings.roi_morph_kernel_size,
        farneback_scale=settings.roi_farneback_scale,
        update_interval_sec=settings.roi_update_interval_sec,
        rolling_window_size=settings.roi_rolling_window_size,
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


# Keep detections whose anchor point falls inside the ROI mask.
def filter_detections_by_roi_xyxy(
    detections: list[dict[str, Any]],
    roi: ROIResult,
    anchor: str = "bottom_center",
) -> list[dict[str, Any]]:
    h, w = roi.frame_shape
    mask = roi.combined_mask
    valid: list[dict[str, Any]] = []

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cx = int((x1 + x2) / 2)

        if anchor == "bottom_center":
            px, py = cx, int(y2)
        else:
            px, py = cx, int((y1 + y2) / 2)

        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))

        if mask[py, px] > 0:
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
    # Maintain a rolling window and periodically recompute ROI.
    def __init__(self, config: ROIConfig, initial_roi: ROIResult) -> None:
        self._config = config
        self._roi_lock = threading.Lock()
        self._current_roi: ROIResult = initial_roi

        self._frame_buffer: deque[np.ndarray] = deque(
            maxlen=config.rolling_window_size
        )
        self._buffer_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.update_count = 0
        self.last_update_time: float | None = None

    @property
    # Thread-safe accessor for the current ROI.
    def current(self) -> ROIResult:
        with self._roi_lock:
            return self._current_roi

    # Add a frame to the rolling buffer.
    def push_frame(self, frame: np.ndarray) -> None:
        with self._buffer_lock:
            self._frame_buffer.append(frame.copy())

    # Start background ROI update thread.
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._update_loop,
            name="roi-updater",
            daemon=True,
        )
        self._thread.start()

    # Stop background ROI update thread.
    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # Periodically recompute ROI using buffered frames.
    def _update_loop(self) -> None:
        while not self._stop_event.is_set():
            stopped = self._stop_event.wait(timeout=self._config.update_interval_sec)
            if stopped:
                break

            with self._buffer_lock:
                frames_snapshot = list(self._frame_buffer)

            if len(frames_snapshot) < 10:
                continue

            try:
                new_roi = compute_adaptive_roi(frames_snapshot, self._config)
                with self._roi_lock:
                    self._current_roi = new_roi
                self.update_count += 1
                self.last_update_time = time.time()
            except Exception:
                logger.exception("ROI update failed; keeping previous ROI.")
