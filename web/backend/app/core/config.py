from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


# Centralized configuration loaded from environment variables.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    model_path: str = "models/best.pt"
    conf: float = 0.25
    iou: float = 0.45
    img_size: int = 640
    device: str = "auto"
    use_half: bool = True
    inference_lock_enabled: bool = True

    frame_skip: int = 1
    max_fps: int = 30
    jpeg_quality: int = 85
    stream_max_dim: int = 640
    save_dir: str = "storage"
    max_video_upload_mb: int = 500
    clean_storage_on_startup: bool = True
    auto_process_video: bool = False
    log_detections: bool = False
    live_history_max_rows: int = 108000
    live_flow_events_max_rows: int = 50000
    live_track_states_max_rows: int = 10000

    pce_motor: float = 0.30
    pce_car: float = 1.00
    pce_truck: float = 2.50
    pce_bus: float = 3.00

    alert_rolling_window_sec: float = 60.0
    alert_reference_roi_area_ratio: float = 0.35
    alert_min_roi_area_ratio: float = 0.10
    alert_min_roi_scale: float = 0.50
    alert_max_roi_scale: float = 2.50
    alert_occ_low: float = 15.0
    alert_occ_mid: float = 30.0
    alert_occ_high: float = 50.0
    alert_pce_density_low: float = 6.0
    alert_pce_density_mid: float = 12.0
    alert_pce_density_high: float = 20.0
    alert_vehicle_density_low: float = 5.0
    alert_vehicle_density_mid: float = 10.0
    alert_vehicle_density_high: float = 18.0
    alert_occupancy_weight: float = 0.40
    alert_pce_density_weight: float = 0.40
    alert_vehicle_density_weight: float = 0.20
    alert_score_busy: float = 0.75
    alert_score_congested: float = 1.50
    alert_score_gridlock: float = 2.30
    alert_hysteresis_sec: float = 3.0

    roi_enabled: bool = False
    roi_mode: str = "mask"
    roi_anchor: str = "bottom_center"
    roi_min_bbox_overlap: float = 0.10
    roi_calib_frames: int = 100
    roi_min_area_ratio: float = 0.05
    roi_otsu_bias: float = 1.05
    roi_morph_kernel_size: int = 15
    roi_farneback_scale: float = 0.5
    roi_max_zones: int = 2
    roi_draw: bool = False
    roi_draw_alpha: float = 0.25

    bytetrack_enabled: bool = True
    bytetrack_conf_high: float = 0.5
    bytetrack_iou_high: float = 0.8
    bytetrack_track_buffer: int = 30
    bytetrack_min_box_area: int = 400
    bytetrack_frame_skip: int = 1
    bytetrack_repair_enabled: bool = True
    bytetrack_repair_iou: float = 0.50

    flow_exit_timeout_sec: float = 2.0
    flow_direction_min_dx_ratio: float = 0.03

    class_names: list[str] = ["car", "truck", "bus", "motor"]
    allowed_origins: list[str] = ["*"]


# Instantiate settings at import time for app-wide use.
settings = Settings()
