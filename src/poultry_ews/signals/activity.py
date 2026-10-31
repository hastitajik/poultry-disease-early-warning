"""Signal 1 — Flock Activity Index.

Dense optical flow (Farneback) is computed on the RECTIFIED floor image, so
flow is in metres. Only pixels that belong to birds count:

  activity_index = moving bird pixels / all bird pixels        (0..1)
  mean_speed     = mean speed of bird pixels                    (m/s)

Normalising by bird pixels makes the index independent of how many birds are
in view and of their size (chicks vs 5-week broilers). Comparison is always
against the baseline for the same age and hour of day (baseline.py).
"""
from __future__ import annotations

import cv2
import numpy as np

FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
                 poly_sigma=1.2, flags=0)


def flow_speed(prev_gray: np.ndarray, gray: np.ndarray, ppm: float, fps: float) -> np.ndarray:
    """Per-pixel speed in m/s between two rectified grey frames."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return (mag / ppm * fps).astype(np.float32)


class ActivityAccumulator:
    def __init__(self, moving_speed_m_s: float = 0.04):
        self.moving_speed = moving_speed_m_s
        self.bird_px = 0.0
        self.moving_px = 0.0
        self.speed_sum = 0.0
        self.pairs = 0

    def update(self, speed: np.ndarray, bird: np.ndarray):
        """speed: m/s map; bird: bool map (bird pixels in the current frame)."""
        n = float(bird.sum())
        if n == 0:
            return
        s = speed[bird]
        self.bird_px += n
        self.moving_px += float((s > self.moving_speed).sum())
        self.speed_sum += float(s.sum())
        self.pairs += 1

    @staticmethod
    def metrics(bird_px: float, moving_px: float, speed_sum: float) -> dict[str, float | None]:
        if bird_px <= 0:
            return {"activity_index": None, "mean_speed": None}
        return {"activity_index": moving_px / bird_px, "mean_speed": speed_sum / bird_px}
