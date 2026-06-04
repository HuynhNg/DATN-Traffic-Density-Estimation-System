from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


# Centralized configuration loaded from environment variables.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    model_path: str = "models/best.pt"
    conf: float = 0.25
    iou: float = 0.45
    img_size: int = 640
    device: str = "auto"
    use_half: bool = True

    frame_skip: int = 1
    max_fps: int = 30
    jpeg_quality: int = 85
    stream_max_dim: int = 640
    save_dir: str = "storage"

    roi_enabled: bool = False
    roi_mode: str = "mask"
    roi_anchor: str = "bottom_center"
    roi_calib_frames: int = 100
    roi_min_area_ratio: float = 0.05
    roi_otsu_bias: float = 1.05
    roi_morph_kernel_size: int = 15
    roi_farneback_scale: float = 0.5
    roi_update_interval_sec: float = 600.0
    roi_rolling_window_size: int = 100
    roi_max_zones: int = 2
    roi_draw: bool = False
    roi_draw_alpha: float = 0.25

    bytetrack_enabled: bool = True
    bytetrack_conf_high: float = 0.5
    bytetrack_iou_high: float = 0.3
    bytetrack_track_buffer: int = 30
    bytetrack_min_box_area: int = 400
    bytetrack_frame_skip: int = 1


    class_names: list[str] = ["car", "truck", "bus", "motor"]
    allowed_origins: list[str] = ["*"]


# Instantiate settings at import time for app-wide use.
settings = Settings()
