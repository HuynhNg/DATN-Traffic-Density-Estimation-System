from __future__ import annotations

import logging
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics import YOLO

from app.core.config import settings

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None


PYTORCH_MODEL_EXTS = {".pt", ".pth"}


# YOLOv8 inference wrapper with device selection.
class InferenceEngine:
    # Load the YOLO model and configure runtime options.
    def __init__(self) -> None:
        self.logger = logging.getLogger("app.inference")
        model_path = settings.model_path
        self.model_path = Path(model_path)
        self.model_format = self._resolve_model_format(self.model_path)
        self._validate_runtime_dependencies()
        self.model = YOLO(model_path)
        self.device = self._resolve_device()
        self._move_model_to_device()
        self._configure_runtime()
        self._log_runtime(model_path)

    def _resolve_model_format(self, model_path: Path) -> str:
        suffix = model_path.suffix.lower()
        if suffix in PYTORCH_MODEL_EXTS:
            return "pytorch"
        if suffix == ".onnx":
            return "onnx"
        if suffix == ".engine":
            return "tensorrt"
        return suffix.lstrip(".") or "unknown"

    def _uses_pytorch_runtime(self) -> bool:
        return self.model_format == "pytorch"

    def _validate_runtime_dependencies(self) -> None:
        if self.model_format == "onnx" and find_spec("onnxruntime") is None:
            raise RuntimeError(
                "APP_MODEL_PATH points to an ONNX model, but onnxruntime is not "
                "installed. Install onnxruntime for CPU or onnxruntime-gpu for "
                "NVIDIA GPU inference."
            )
        if self.model_format == "tensorrt" and find_spec("tensorrt") is None:
            raise RuntimeError(
                "APP_MODEL_PATH points to a TensorRT engine, but tensorrt is not "
                "installed in this environment."
            )

    # Choose device based on settings and availability.
    def _resolve_device(self) -> str:
        if settings.device != "auto":
            return settings.device
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _move_model_to_device(self) -> None:
        if not self._uses_pytorch_runtime():
            return
        self.model.to(self.device)

    # Enable CUDA optimizations when available.
    def _configure_runtime(self) -> None:
        if torch is None or not self._uses_pytorch_runtime():
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
            self.logger.info(
                "Runtime: format=%s device=%s model=%s",
                self.model_format,
                self.device,
                model_path,
            )
            return
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "cpu"
        self.logger.info(
            "Runtime: format=%s device=%s gpu=%s model=%s",
            self.model_format,
            self.device,
            gpu_name,
            model_path,
        )

    def _class_name(self, result: Any, class_id: int) -> str:
        if class_id < len(settings.class_names):
            return settings.class_names[class_id]

        names = getattr(result, "names", None) or getattr(self.model, "names", None)
        if isinstance(names, dict) and class_id in names:
            return str(names[class_id])
        if isinstance(names, list) and class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _predict_kwargs(self, frame: np.ndarray) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": settings.conf,
            "iou": settings.iou,
            "imgsz": settings.img_size,
            "device": self.device,
            "verbose": False,
        }
        if self._uses_pytorch_runtime():
            kwargs["half"] = settings.use_half and self.device.startswith("cuda")
        return kwargs

    # Run inference and return detections with elapsed time in ms.
    def detect(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        start = time.perf_counter()
        results = self.model.predict(**self._predict_kwargs(frame))
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections: list[dict[str, Any]] = []
        if not results:
            return detections, elapsed_ms

        result = results[0]
        boxes = result.boxes
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = self._class_name(result, class_id)
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
