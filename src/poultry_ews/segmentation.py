"""Bird segmentation: which pixels are birds?

Stage 1  ColorSegmenter    no training. White/yellow birds are brighter and
                           less saturated than the litter. Good enough for
                           activity and distribution; weak for single birds
                           in dense huddles.
Stage 2  YoloSegmenter     a YOLO-seg model trained on your own frames
                           (see docs/labeling_guide.md). `pip install ultralytics`.
Stage 3  ExternalDetections  boxes produced by another tool (e.g. the MVBroTrack
                           multi-view pipeline) saved as <clip>.dets.jsonl.

Every segmenter returns a uint8 mask (0/255) in WORKING image coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class Segmenter(Protocol):
    def __call__(self, frame_bgr: np.ndarray, frame_index: int = 0) -> np.ndarray: ...


class ColorSegmenter:
    def __init__(self, contrast: float = 18, v_min: int = 140, s_max: int = 120,
                 bg_kernel_frac: float = 0.09, open_px: int = 3):
        self.contrast = contrast
        self.v_min = v_min
        self.s_max = s_max
        self.bg_kernel_frac = bg_kernel_frac
        self.open_px = open_px

    def __call__(self, frame_bgr: np.ndarray, frame_index: int = 0) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        L = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)[..., 0]
        k = int(w * self.bg_kernel_frac) | 1
        k = max(k, 3)
        # median of a downscaled image ~ local floor brightness, cheap even for big kernels
        small = cv2.resize(L, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        bg = cv2.medianBlur(small, max(3, (k // 4) | 1))
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        Hc, S, V = cv2.split(hsv)
        bright = (L.astype(np.int16) - bg.astype(np.int16)) > self.contrast
        pale = (V >= self.v_min) & (S <= self.s_max)
        red = ((Hc < 12) | (Hc > 165)) & (S > 110)
        m = (bright & pale & ~red).astype(np.uint8) * 255
        if self.open_px > 0:
            k2 = np.ones((self.open_px, self.open_px), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2)
        return m


class YoloSegmenter:
    """Ultralytics YOLO (seg or detect) model. Imported lazily."""

    def __init__(self, weights: str | Path, conf: float = 0.35, classes: list[int] | None = None):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("YoloSegmenter needs `pip install ultralytics`") from e
        self.model = YOLO(str(weights))
        self.conf = conf
        self.classes = classes

    def __call__(self, frame_bgr: np.ndarray, frame_index: int = 0) -> np.ndarray:  # pragma: no cover
        h, w = frame_bgr.shape[:2]
        res = self.model.predict(frame_bgr, conf=self.conf, classes=self.classes, verbose=False)[0]
        m = np.zeros((h, w), np.uint8)
        if res.masks is not None and len(res.masks):
            for poly in res.masks.xy:
                cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
        elif res.boxes is not None:
            for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                _fill_box_ellipse(m, x1, y1, x2, y2)
        return m


class ExternalDetections:
    """Per-frame boxes from another tool.

    File: <clip><suffix>, JSON lines:
      {"frame": 120, "boxes": [[x1, y1, x2, y2], ...]}   # FULL-resolution pixels
    Frames without a line reuse the nearest earlier line.
    """

    def __init__(self, clip_path: Path, suffix: str, full_width: int, work_width: int):
        f = Path(str(clip_path) + suffix)
        if not f.exists():
            raise FileNotFoundError(f"external detections not found: {f}")
        self.by_frame: dict[int, list] = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                self.by_frame[int(rec["frame"])] = rec.get("boxes", [])
        self.keys = sorted(self.by_frame)
        self.scale = work_width / full_width

    def __call__(self, frame_bgr: np.ndarray, frame_index: int = 0) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        m = np.zeros((h, w), np.uint8)
        if not self.keys:
            return m
        i = np.searchsorted(self.keys, frame_index, side="right") - 1
        boxes = self.by_frame[self.keys[max(i, 0)]]
        for x1, y1, x2, y2 in boxes:
            s = self.scale
            _fill_box_ellipse(m, x1 * s, y1 * s, x2 * s, y2 * s)
        return m


def _fill_box_ellipse(m, x1, y1, x2, y2):
    c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    axes = (max(1, int((x2 - x1) / 2)), max(1, int((y2 - y1) / 2)))
    cv2.ellipse(m, c, axes, 0, 0, 360, 255, -1)


def build_segmenter(cfg_seg: dict, clip_path: Path | None = None,
                    full_width: int = 0, work_width: int = 0, root: Path | None = None) -> Segmenter:
    kind = cfg_seg.get("kind", "color")
    if kind == "color":
        return ColorSegmenter(contrast=cfg_seg.get("contrast", 18),
                              v_min=cfg_seg.get("v_min", 140), s_max=cfg_seg.get("s_max", 120))
    if kind == "yolo":
        w = Path(cfg_seg["weights"])
        if root is not None and not w.is_absolute():
            w = root / w
        return YoloSegmenter(w, conf=cfg_seg.get("conf", 0.35))
    if kind == "external":
        if clip_path is None:
            raise ValueError("external segmenter needs the clip path")
        return ExternalDetections(clip_path, cfg_seg.get("external_suffix", ".dets.jsonl"),
                                  full_width, work_width)
    raise ValueError(f"unknown segmenter kind {kind!r}")
