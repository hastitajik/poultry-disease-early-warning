"""Frame sources: video clips (mp4 / raw .h264) or still images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".h264", ".264", ".avi", ".mkv", ".mov", ".dav", ".ts"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Frame:
    index: int          # index in the original stream
    t_sec: float        # seconds from clip start
    image: np.ndarray   # BGR, resized to work size


def is_media(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXT | IMAGE_EXT


def _resize(img: np.ndarray, work_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == work_width:
        return img
    scale = work_width / w
    return cv2.resize(img, (work_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def iter_frames(path: str | Path, analysis_fps: float, work_width: int,
                fallback_fps: float = 15.0, max_seconds: float | None = None
                ) -> Iterator[Frame]:
    """Yield frames sampled at ~analysis_fps.

    Uses grab() for skipped frames so they are not decoded (much faster on
    long clips). Raw .h264 streams often report fps=0; then fallback_fps is used.
    """
    path = Path(path)
    if path.suffix.lower() in IMAGE_EXT:
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"cannot read image {path}")
        yield Frame(0, 0.0, _resize(img, work_width))
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not (1.0 <= src_fps <= 120.0):
            src_fps = fallback_fps
        step = max(1, int(round(src_fps / analysis_fps)))
        idx = 0
        while True:
            if idx % step == 0:
                ok, img = cap.read()
                if not ok:
                    break
                t = idx / src_fps
                if max_seconds is not None and t > max_seconds:
                    break
                yield Frame(idx, t, _resize(img, work_width))
            else:
                if not cap.grab():
                    break
            idx += 1
    finally:
        cap.release()


def effective_fps(path: str | Path, analysis_fps: float, fallback_fps: float = 15.0) -> float:
    """The real sampling rate produced by iter_frames (src_fps / step)."""
    path = Path(path)
    if path.suffix.lower() in IMAGE_EXT:
        return analysis_fps
    cap = cv2.VideoCapture(str(path))
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()
    if not (1.0 <= src_fps <= 120.0):
        src_fps = fallback_fps
    step = max(1, int(round(src_fps / analysis_fps)))
    return src_fps / step
