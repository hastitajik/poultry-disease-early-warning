"""Signal 3 — Mortality (dead-bird candidates).

Idea (Hao et al., Agriculture 2025): a dead bird stays in the same place
while the rest of the flock moves. Per clip we mark floor pixels that are
"bird" in almost every frame but (almost) never move, group them into blobs
of roughly one bird's size, and track those blobs across clips. A blob that
stays still for `alert_after_min` minutes WHILE the flock was active is
raised as a candidate for the farmer / vet to inspect.

Limits: sleeping birds are still too (hence the flock-activity gate); a dead
bird lying against resting birds merges with them into a too-large blob and
is missed until a trained detector (stage 2) separates individual birds.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StillBlob:
    x_m: float
    y_m: float
    area_m2: float


class StillnessAccumulator:
    def __init__(self, shape: tuple[int, int], still_speed_m_s: float):
        self.present = np.zeros(shape, np.uint16)
        self.moving = np.zeros(shape, np.uint16)
        self.n = 0
        self.still_speed = still_speed_m_s

    def update(self, bird: np.ndarray, speed: np.ndarray | None):
        self.present += bird.astype(np.uint16)
        if speed is not None:
            self.moving += ((speed > self.still_speed) & bird).astype(np.uint16)
            self.n += 1

    def blobs(self, ppm: float, bird_area_m2: float, min_presence_frac: float,
              max_moving_frac: float, min_area_factor: float, max_area_factor: float
              ) -> list[StillBlob]:
        if self.n < 3:
            return []
        frames = self.n + 1  # presence counted on every frame, motion on pairs
        still = (self.present >= min_presence_frac * frames) & \
                (self.moving <= max_moving_frac * self.n)
        still = cv2.morphologyEx(still.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(still, connectivity=8)
        px_area = 1.0 / (ppm * ppm)
        lo, hi = bird_area_m2 * min_area_factor, bird_area_m2 * max_area_factor
        out = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA] * px_area
            if lo <= a <= hi:
                cx, cy = cents[i]
                out.append(StillBlob(cx / ppm, cy / ppm, a))
        return out


def merge_blobs(blobs: list[tuple[float, float, float]], radius_m: float
                ) -> list[tuple[float, float, float]]:
    """Merge blobs seen by overlapping cameras (same bird, slightly different position)."""
    merged: list[list[float]] = []
    for x, y, a in sorted(blobs, key=lambda b: -b[2]):
        for m in merged:
            if (m[0] - x) ** 2 + (m[1] - y) ** 2 <= radius_m ** 2:
                break
        else:
            merged.append([x, y, a])
    return [tuple(m) for m in merged]  # type: ignore[misc]
