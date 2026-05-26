from __future__ import annotations

import time


# Track FPS and object counts over time.
class AnalyticsTracker:
    def __init__(self) -> None:
        self.last_ts = time.perf_counter()
        self.frame_count = 0
        self.avg_objects = 0.0
        self.total_objects = 0

    # Update rolling metrics with the latest frame count.
    def update(self, objects_in_frame: int) -> dict:
        now = time.perf_counter()
        self.frame_count += 1
        self.total_objects += objects_in_frame
        self.avg_objects = self.total_objects / max(self.frame_count, 1)
        fps = 1.0 / max(now - self.last_ts, 1e-6)
        self.last_ts = now
        return {
            "fps": round(fps, 2),
            "avg_objects": round(self.avg_objects, 2),
            "objects_in_frame": objects_in_frame,
        }
