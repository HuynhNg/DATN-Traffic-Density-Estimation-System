from __future__ import annotations

from typing import Any

import cv2
import numpy as np


COLOR_MAP = {
    "car": (65, 113, 246),
    "truck": (255, 138, 76),
    "bus": (246, 90, 90),
    "motor": (66, 199, 120),
}


def render_boxes(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    show_labels: bool,
    show_conf: bool,
) -> np.ndarray:
    canvas = frame.copy()
    for det in detections:
        label = det["class_name"]
        color = COLOR_MAP.get(label, (200, 200, 200))
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            text = label
            if show_conf:
                text += f" {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                canvas,
                text,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas
