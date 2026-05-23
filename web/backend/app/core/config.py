from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    class_names: list[str] = ["car", "truck", "bus", "motor"]
    allowed_origins: list[str] = ["*"]


settings = Settings()
