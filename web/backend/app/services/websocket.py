from __future__ import annotations

import base64
import json
from typing import Any

import cv2
import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.core.config import settings
from app.services.analytics import AnalyticsTracker
from app.services.inference import InferenceEngine
from app.services.frame_utils import resize_to_max
from app.services.renderer import render_boxes


async def handle_stream(websocket: WebSocket, engine: InferenceEngine) -> None:
    await websocket.accept()
    tracker = AnalyticsTracker()
    show_labels = True
    show_conf = True

    while True:
        try:
            message = await websocket.receive()
        except WebSocketDisconnect:
            break
        except Exception:
            break
        if "text" in message and message["text"]:
            payload = json.loads(message["text"])
            show_labels = bool(payload.get("labels", show_labels))
            show_conf = bool(payload.get("conf", show_conf))
            continue

        data = message.get("bytes")
        if not data:
            continue

        np_arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        resized = resize_to_max(frame, settings.stream_max_dim)
        detections, infer_ms = engine.detect(resized)
        annotated = render_boxes(resized, detections, show_labels, show_conf)

        _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality])
        b64 = base64.b64encode(buffer).decode("ascii")

        metrics = tracker.update(len(detections))
        payload: dict[str, Any] = {
            "image_b64": b64,
            "detections": detections,
            "inference_ms": round(infer_ms, 2),
            "metrics": metrics,
        }
        try:
            await websocket.send_text(json.dumps(payload))
        except WebSocketDisconnect:
            break
        except Exception:
            break
