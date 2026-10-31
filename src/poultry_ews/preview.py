"""Calibration preview: draw ROI, exclusions, floor grid, zones and segmentation on a frame.

Run after every calibration change and look at the result before trusting any signal:
    pews preview --config configs/farm.yaml --camera cam1 --image frame.png --out preview.jpg
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .analyzer import camera_geometry
from .config import Config
from .geometry import CORNER, DRINKER, FEEDER, OPEN, WALL
from .segmentation import build_segmenter

ZONE_COLORS = {OPEN: (60, 60, 60), WALL: (200, 140, 40), CORNER: (200, 60, 160),
               DRINKER: (230, 180, 0), FEEDER: (0, 160, 255)}


def render_preview(cfg: Config, camera_id: str, image_path: str | Path, when: datetime) -> np.ndarray:
    cam = cfg.camera(camera_id)
    geo = camera_geometry(cfg, cam, when)
    pen = cfg.pen(cam.pen).at(when)
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"cannot read {image_path}")
    img = cv2.resize(img, geo.work_size, interpolation=cv2.INTER_AREA)
    seg = build_segmenter({**cfg.analysis["segmenter"], "kind": "color"}
                          if cfg.analysis["segmenter"]["kind"] == "external"
                          else cfg.analysis["segmenter"], root=cfg.root)
    mask = cv2.bitwise_and(seg(img), geo.image_mask)

    left = img.copy()
    # excluded area darkened
    excl = (geo.image_mask == 0)
    left[excl] = (left[excl] * 0.35).astype(np.uint8)
    # birds in red
    b = mask > 0
    left[b] = (0.45 * left[b] + np.array([0, 0, 140])).clip(0, 255).astype(np.uint8)
    # floor grid every 0.5 m
    ppm = geo.canvas.ppm
    W, L = pen.width_m, pen.length_m
    for x in np.arange(0, W + 1e-6, 0.5):
        _polyline(left, geo, [(x * ppm, y * ppm) for y in np.linspace(0, L, 40)], (0, 255, 0))
    for y in np.arange(0, L + 1e-6, 0.5):
        _polyline(left, geo, [(x * ppm, y * ppm) for x in np.linspace(0, W, 20)], (0, 255, 0))
    # equipment zones
    for circles, col in ((pen.feeders, ZONE_COLORS[FEEDER]), (pen.drinkers, ZONE_COLORS[DRINKER])):
        for c in circles:
            ang = np.linspace(0, 2 * np.pi, 48)
            _polyline(left, geo, [((c.x + c.r * np.cos(t)) * ppm, (c.y + c.r * np.sin(t)) * ppm)
                                  for t in ang], col, 2)
    cv2.polylines(left, [np.round(geo.cal.roi * geo.scale).astype(np.int32)], True, (255, 255, 255), 2)

    # right: rectified floor with zones + birds
    zone = geo.canvas.zone_map(pen)
    right = np.zeros((*zone.shape, 3), np.uint8)
    for lab, col in ZONE_COLORS.items():
        right[zone == lab] = col
    birds = (geo.to_floor(mask, nearest=True) > 127) & geo.coverage
    right[birds] = (255, 255, 255)
    right[~geo.coverage] = (right[~geo.coverage] * 0.25).astype(np.uint8)
    scale = left.shape[0] / right.shape[0]
    right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]), interpolation=cv2.INTER_NEAREST)
    _legend(right)
    return np.hstack([left, right])


def _polyline(img, geo, pts_floor_px, color, thick=1):
    pts = geo.floor_px_to_image(np.asarray(pts_floor_px))
    cv2.polylines(img, [np.round(pts).astype(np.int32)], False, color, thick, cv2.LINE_AA)


def _legend(img):
    items = [("feeder", ZONE_COLORS[FEEDER]), ("drinker", ZONE_COLORS[DRINKER]),
             ("wall", ZONE_COLORS[WALL]), ("corner", ZONE_COLORS[CORNER]), ("bird", (255, 255, 255))]
    for i, (name, col) in enumerate(items):
        y = 20 + 22 * i
        cv2.rectangle(img, (8, y - 12), (22, y + 2), col, -1)
        cv2.putText(img, name, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
