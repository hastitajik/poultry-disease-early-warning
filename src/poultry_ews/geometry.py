"""Image -> floor-plane geometry.

The cameras look at the floor at an angle, so a bird far from the camera
covers fewer pixels and a step of 10 cm there is fewer pixels than near the
camera. All measurements are therefore made on a RECTIFIED floor map in
metres (homography from 4+ clicked points), at `floor_px_per_m` resolution.
Several cameras over one pen share the same floor map -> fusion is a sum.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from .config import CalibrationVersion, PenConfig

# zone labels on the floor map
OPEN, WALL, CORNER, DRINKER, FEEDER = 0, 1, 2, 3, 4
ZONE_NAMES = {OPEN: "open", WALL: "wall", CORNER: "corner", DRINKER: "drinker", FEEDER: "feeder"}


@dataclass
class FloorCanvas:
    width_m: float
    length_m: float
    ppm: float  # pixels per metre

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols) of the floor map."""
        return int(np.ceil(self.length_m * self.ppm)), int(np.ceil(self.width_m * self.ppm))

    @property
    def size_wh(self) -> tuple[int, int]:
        r, c = self.shape
        return c, r

    def px_to_m(self, x: float, y: float) -> tuple[float, float]:
        return x / self.ppm, y / self.ppm

    def zone_map(self, pen: PenConfig) -> np.ndarray:
        """uint8 label map. Later assignments override: wall < corner < drinker < feeder."""
        rows, cols = self.shape
        z = np.full((rows, cols), OPEN, np.uint8)
        band = int(round(pen.wall_band_m * self.ppm))
        if band > 0:
            z[:, :band] = WALL
            z[:, cols - band:] = WALL
        cs = int(round(pen.corner_size_m * self.ppm))
        if cs > 0:
            for r0, c0 in ((0, 0), (0, cols - cs), (rows - cs, 0), (rows - cs, cols - cs)):
                z[max(r0, 0):r0 + cs, max(c0, 0):c0 + cs] = CORNER
        for circles, label in ((pen.drinkers, DRINKER), (pen.feeders, FEEDER)):
            for c in circles:
                cv2.circle(z, (int(round(c.x * self.ppm)), int(round(c.y * self.ppm))),
                           int(round(c.r * self.ppm)), int(label), thickness=-1)
        return z


class CameraGeometry:
    """Calibration of one camera, scaled to the working image size."""

    def __init__(self, cal: CalibrationVersion, canvas: FloorCanvas, work_width: int):
        self.cal = cal
        self.canvas = canvas
        full_w, full_h = cal.image_size
        self.scale = work_width / full_w
        self.work_size = (work_width, int(round(full_h * self.scale)))  # (w, h)
        ip = cal.image_points * self.scale
        H, _ = cv2.findHomography(ip.astype(np.float32), cal.floor_points.astype(np.float32), 0)
        if H is None:
            raise ValueError("degenerate calibration points (homography failed)")
        self.H_img_to_m = H
        S = np.diag([canvas.ppm, canvas.ppm, 1.0])
        self.M = S @ H                          # image px -> floor-map px
        self.M_inv = np.linalg.inv(self.M)      # floor-map px -> image px

    # ---- masks ------------------------------------------------------------
    @cached_property
    def image_mask(self) -> np.ndarray:
        """ROI minus exclusions, in working image coordinates (uint8 0/255)."""
        w, h = self.work_size
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(self.cal.roi * self.scale).astype(np.int32)], 255)
        for ex in self.cal.exclude:
            if isinstance(ex, tuple) and ex[0] == "circle":
                _, cx, cy, r = ex
                cv2.circle(m, (int(cx * self.scale), int(cy * self.scale)),
                           max(1, int(r * self.scale)), 0, thickness=-1)
            else:
                cv2.fillPoly(m, [np.round(ex * self.scale).astype(np.int32)], 0)
        return m

    @cached_property
    def coverage(self) -> np.ndarray:
        """Floor-map pixels this camera actually sees (bool)."""
        return self.to_floor(self.image_mask, nearest=True) > 0

    # ---- warping ----------------------------------------------------------
    def to_floor(self, img: np.ndarray, nearest: bool = False) -> np.ndarray:
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpPerspective(img, self.M, self.canvas.size_wh, flags=interp,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def floor_px_to_image(self, pts_px: np.ndarray) -> np.ndarray:
        """Floor-map pixels (N,2) -> working-image pixels (N,2)."""
        pts = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.M_inv).reshape(-1, 2)

    def floor_m_to_image_full(self, pts_m: np.ndarray) -> np.ndarray:
        """Floor metres (N,2) -> FULL-resolution image pixels."""
        px = np.asarray(pts_m, np.float64) * self.canvas.ppm
        return self.floor_px_to_image(px) / self.scale

    def image_full_to_m(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, np.float64).reshape(-1, 1, 2) * self.scale
        return cv2.perspectiveTransform(pts, self.H_img_to_m).reshape(-1, 2)
