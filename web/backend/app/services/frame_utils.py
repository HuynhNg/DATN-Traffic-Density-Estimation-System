from __future__ import annotations

import cv2
import numpy as np


# Resize a frame so its longest side does not exceed max_dim.
def resize_to_max(frame: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return frame

    scale = max_dim / float(longest)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
