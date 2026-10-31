"""Analyse ONE clip from ONE camera -> per-camera result for one time slot."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import CameraConfig, Config
from .geometry import CameraGeometry, FloorCanvas
from .segmentation import Segmenter, build_segmenter
from .signals.activity import ActivityAccumulator, flow_speed
from .signals.mortality import StillBlob, StillnessAccumulator
from .video import effective_fps, iter_frames


@dataclass
class CameraClipResult:
    camera: str
    pen: str
    start: datetime
    duration_s: float
    n_frames: int
    bird_px: float
    moving_px: float
    speed_sum: float
    occupancy: np.ndarray          # float32 floor map, mean bird presence 0..1
    coverage: np.ndarray           # bool floor map
    blobs: list[StillBlob] = field(default_factory=list)
    snapshot: np.ndarray | None = None   # BGR working-size frame (middle of clip)
    geometry: CameraGeometry | None = None


def camera_geometry(cfg: Config, cam: CameraConfig, when: datetime) -> CameraGeometry:
    pen = cfg.pen(cam.pen)
    canvas = FloorCanvas(pen.width_m, pen.length_m, float(cfg.analysis["floor_px_per_m"]))
    return CameraGeometry(cam.calibration_for(when), canvas, int(cfg.analysis["work_width"]))


def analyze_clip(cfg: Config, camera_id: str, clip: str | Path, start: datetime,
                 segmenter: Segmenter | None = None, max_seconds: float | None = None
                 ) -> CameraClipResult:
    cam = cfg.camera(camera_id)
    a = cfg.analysis
    geo = camera_geometry(cfg, cam, start)
    ppm = geo.canvas.ppm
    fps = effective_fps(clip, float(a["analysis_fps"]), cam.source_fps)
    if segmenter is None:
        segmenter = build_segmenter(a["segmenter"], clip_path=Path(clip),
                                    full_width=geo.cal.image_size[0],
                                    work_width=int(a["work_width"]), root=cfg.root)

    act = ActivityAccumulator(float(a["activity"]["moving_speed_m_s"]))
    still = StillnessAccumulator(geo.canvas.shape, float(a["mortality"]["still_speed_m_s"]))
    occ_sum = np.zeros(geo.canvas.shape, np.float32)
    cov = geo.coverage
    roi = geo.image_mask

    prev_gray = None
    n = 0
    t_last = 0.0
    snapshot = None
    for fr in iter_frames(clip, float(a["analysis_fps"]), int(a["work_width"]),
                          cam.source_fps, max_seconds):
        img = fr.image
        if img.shape[1] != geo.work_size[0] or img.shape[0] != geo.work_size[1]:
            img = cv2.resize(img, geo.work_size, interpolation=cv2.INTER_AREA)
        mask = cv2.bitwise_and(segmenter(img, fr.index), roi)
        bird = (geo.to_floor(mask, nearest=True) > 127) & cov
        gray = geo.to_floor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        speed = None
        if prev_gray is not None:
            speed = flow_speed(prev_gray, gray, ppm, fps)
            act.update(speed, bird)
        still.update(bird, speed)
        occ_sum += bird
        prev_gray = gray
        n += 1
        t_last = fr.t_sec
        snapshot = img  # last frame of the clip = most recent view
    if n == 0:
        raise IOError(f"no frames read from {clip}")

    age = cfg.age_days(start)
    m = a["mortality"]
    blobs = still.blobs(ppm, cfg.bird_area_m2(age), float(m["min_presence_frac"]),
                        float(m["max_moving_frac"]), float(m["min_area_factor"]),
                        float(m["max_area_factor"]))
    return CameraClipResult(
        camera=camera_id, pen=cam.pen, start=start,
        duration_s=t_last + 1.0 / fps if n > 1 else 0.0, n_frames=n,
        bird_px=act.bird_px, moving_px=act.moving_px, speed_sum=act.speed_sum,
        occupancy=occ_sum / n, coverage=cov, blobs=blobs, snapshot=snapshot, geometry=geo)
