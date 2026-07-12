from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


DEFAULT_MODELS = (
    "models/best.pt",
    "models/best.onnx",
    "models/best_fp16.engine",
    "models/best.engine",
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO PyTorch, ONNX, and TensorRT models."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image or video path used as the benchmark input.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model paths to benchmark. Defaults to known files in models/.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0", help="Use 0 for GPU or cpu for CPU.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Optional path to write benchmark results as JSON.",
    )
    return parser.parse_args()


def resolve_models(model_args: list[str] | None) -> list[Path]:
    candidates = model_args or list(DEFAULT_MODELS)
    models = [Path(path) for path in candidates if Path(path).exists()]
    if not models:
        raise FileNotFoundError(
            "No model files found. Pass explicit paths with --models."
        )
    return models


def load_frames(source: Path, max_frames: int, frame_stride: int) -> list[Any]:
    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTS:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Cannot read image: {source}")
        return [frame]

    if suffix not in VIDEO_EXTS:
        raise ValueError(f"Unsupported source type: {source.suffix}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")

    frames: list[Any] = []
    frame_idx = 0
    stride = max(1, frame_stride)
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                frames.append(frame)
            frame_idx += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from: {source}")
    return frames


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def predict_once(
    model: YOLO,
    frame: Any,
    args: argparse.Namespace,
) -> tuple[float, int]:
    start = time.perf_counter()
    results = model.predict(
        source=frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    count = len(results[0].boxes) if results else 0
    return elapsed_ms, count


def benchmark_model(
    model_path: Path,
    frames: list[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        model = YOLO(str(model_path))

        warmup_frames = frames[: max(0, min(args.warmup, len(frames)))]
        for frame in warmup_frames:
            predict_once(model, frame, args)

        times: list[float] = []
        counts: list[int] = []
        for frame in frames:
            elapsed_ms, count = predict_once(model, frame, args)
            times.append(elapsed_ms)
            counts.append(count)

        avg_ms = statistics.mean(times)
        return {
            "model": str(model_path),
            "frames": len(frames),
            "avg_ms": round(avg_ms, 2),
            "p50_ms": round(percentile(times, 0.50), 2),
            "p95_ms": round(percentile(times, 0.95), 2),
            "fps": round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0,
            "avg_objects": round(statistics.mean(counts), 2),
            "error": "",
        }
    except Exception as exc:
        return {
            "model": str(model_path),
            "frames": 0,
            "avg_ms": "-",
            "p50_ms": "-",
            "p95_ms": "-",
            "fps": "-",
            "avg_objects": "-",
            "error": str(exc),
        }


def print_table(results: list[dict[str, Any]]) -> None:
    headers = [
        "model",
        "frames",
        "avg_ms",
        "p50_ms",
        "p95_ms",
        "fps",
        "avg_objects",
        "error",
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in results))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in results:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    models = resolve_models(args.models)
    frames = load_frames(source, args.max_frames, args.frame_stride)
    results = [benchmark_model(model_path, frames, args) for model_path in models]

    print_table(results)

    if args.json_path:
        output_path = Path(args.json_path)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {output_path}")


if __name__ == "__main__":
    main()
