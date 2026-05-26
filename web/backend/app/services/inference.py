from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ultralytics import YOLO

from app.core.config import settings

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None


# YOLOv8 inference wrapper with device selection.
class InferenceEngine:
    # Load the YOLO model and configure runtime options.
    def __init__(self) -> None:
        self.logger = logging.getLogger("app.inference")
        model_path = settings.model_path
        self.model = YOLO(model_path)
        self.device = self._resolve_device()
        self.model.to(self.device)
        self._configure_runtime()
        self._log_runtime(model_path)

    # Choose device based on settings and availability.
    def _resolve_device(self) -> str:
        if settings.device != "auto":
            return settings.device
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    # Enable CUDA optimizations when available.
    def _configure_runtime(self) -> None:
        if torch is None:
            return
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            try:
                self.model.fuse()
            except Exception:
                pass

    # Log runtime device and model details.
    def _log_runtime(self, model_path: str) -> None:
        if torch is None:
            self.logger.info("Runtime: device=%s model=%s", self.device, model_path)
            return
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "cpu"
        self.logger.info("Runtime: device=%s gpu=%s model=%s", self.device, gpu_name, model_path)

    # Run inference and return detections with elapsed time in ms.
    def detect(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=settings.conf,
            iou=settings.iou,
            imgsz=settings.img_size,
            device=self.device,
            half=settings.use_half and self.device.startswith("cuda"),
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections: list[dict[str, Any]] = []
        if not results:
            return detections, elapsed_ms

        boxes = results[0].boxes
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = (
                settings.class_names[class_id]
                if class_id < len(settings.class_names)
                else str(class_id)
            )
            detections.append(
                {
                    "object_id": idx,
                    "class_id": class_id,
                    "class_name": name,
                    "confidence": conf,
                    "x1": int(xyxy[0]),
                    "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]),
                    "y2": int(xyxy[3]),
                }
            )

        return detections, elapsed_ms
