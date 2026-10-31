"""Command line: pews <command> --config configs/farm.yaml ...

  process       analyse clips (files or folders) of one camera, in time order
  watch         keep polling every camera's input_dir and process new clips
  serve         start the web panel / API
  preview       draw calibration + segmentation on a frame (check before trusting signals)
  calibrate     click calibration points on a frame (needs a desktop OpenCV)
  to-metres     convert image pixels to floor metres (for feeder / drinker positions)
  extract-frames  sample frames from clips for labelling (stage 2 detector)
  log-mortality   record daily deaths / culls
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path


def _cfg(args):
    from .config import load_config
    return load_config(args.config, getattr(args, "rules", None))


def cmd_process(args):
    from .pipeline import Pipeline, find_clips, process_many
    cfg = _cfg(args)
    pipe = Pipeline(cfg)
    clips = find_clips([Path(p) for p in args.paths])
    if not clips:
        print("no video/image files found", file=sys.stderr)
        return 1
    ok, err = process_many(pipe, args.camera, clips, args.max_seconds)
    print(f"processed {ok} clips, {err} errors -> {cfg.db_path}")
    return 0 if err == 0 else 2


def cmd_watch(args):
    from .pipeline import Pipeline, watch
    cfg = _cfg(args)
    watch(Pipeline(cfg), poll_s=args.poll, once=args.once)
    return 0


def cmd_serve(args):
    import uvicorn
    from .api.app import create_app
    uvicorn.run(create_app(_cfg(args)), host=args.host, port=args.port)
    return 0


def cmd_preview(args):
    import cv2
    from .preview import render_preview
    from .timeparse import TimestampError, parse_clip_time
    cfg = _cfg(args)
    when = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None
    if when is None:
        try:
            when = parse_clip_time(args.image)
        except TimestampError:
            when = datetime.now()
    out = render_preview(cfg, args.camera, args.image, when)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out} (calibration valid on {when.date()})")
    return 0


def cmd_calibrate(args):
    from .calibrate import run_calibration
    cfg = _cfg(args)
    cam = cfg.camera(args.camera)
    pen = cfg.pen(cam.pen)
    run_calibration(args.image, args.out or cam.calibration_file, cam.id, pen.width_m, pen.length_m,
                    datetime.strptime(args.valid_from, "%Y-%m-%d").date())
    return 0


def cmd_to_metres(args):
    import numpy as np
    from .analyzer import camera_geometry
    cfg = _cfg(args)
    geo = camera_geometry(cfg, cfg.camera(args.camera), datetime.strptime(args.date, "%Y-%m-%d"))
    pts = np.array([[float(v) for v in p.split(",")] for p in args.px])
    for p, m in zip(pts, geo.image_full_to_m(pts)):
        print(f"px ({p[0]:.0f}, {p[1]:.0f}) -> x={m[0]:.2f} m, y={m[1]:.2f} m")
    return 0


def cmd_extract(args):
    import cv2
    from .pipeline import find_clips
    from .video import iter_frames
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for clip in find_clips([Path(p) for p in args.paths]):
        for fr in iter_frames(clip, 1.0 / args.every, args.width, fallback_fps=args.fps):
            name = out / f"{clip.stem}_{int(fr.t_sec):05d}s.jpg"
            cv2.imwrite(str(name), fr.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
    print(f"wrote {n} frames to {out}")
    return 0


def cmd_log(args):
    from .pipeline import Pipeline
    pct = Pipeline(_cfg(args)).daily_mortality(args.pen, args.date, args.deaths, args.culls, args.note)
    print(f"{args.pen} {args.date}: daily mortality {pct:.2f}%" if pct is not None else "logged")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="pews", description="Poultry disease early-warning system")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", default="configs/farm.yaml")
        sp.add_argument("--rules", default=None, help="rules.yaml (default: configs/rules.yaml)")
        return sp

    s = common(sub.add_parser("process", help="analyse clips of one camera"))
    s.add_argument("--camera", required=True)
    s.add_argument("--max-seconds", type=float, default=None, help="analyse only the start of each clip")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_process)

    s = common(sub.add_parser("watch", help="process new clips as they arrive"))
    s.add_argument("--poll", type=float, default=30)
    s.add_argument("--once", action="store_true", help="one pass, then exit")
    s.set_defaults(fn=cmd_watch)

    s = common(sub.add_parser("serve", help="web panel"))
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = common(sub.add_parser("preview", help="check calibration on a frame"))
    s.add_argument("--camera", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--date", default=None, help="YYYY-MM-DD (which calibration version)")
    s.add_argument("--out", default="preview.jpg")
    s.set_defaults(fn=cmd_preview)

    s = common(sub.add_parser("calibrate", help="click calibration on a frame"))
    s.add_argument("--camera", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--valid-from", required=True, help="YYYY-MM-DD from which this framing is valid")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_calibrate)

    s = common(sub.add_parser("to-metres", help="image pixels -> floor metres"))
    s.add_argument("--camera", required=True)
    s.add_argument("--date", required=True)
    s.add_argument("px", nargs="+", help="x,y in full-resolution pixels")
    s.set_defaults(fn=cmd_to_metres)

    s = sub.add_parser("extract-frames", help="sample frames for labelling")
    s.add_argument("paths", nargs="+")
    s.add_argument("--every", type=float, default=60, help="seconds between frames")
    s.add_argument("--width", type=int, default=2304)
    s.add_argument("--fps", type=float, default=15, help="fallback fps for raw .h264")
    s.add_argument("--out", default="data/labeling")
    s.set_defaults(fn=cmd_extract)

    s = common(sub.add_parser("log-mortality", help="record daily deaths"))
    s.add_argument("--pen", required=True)
    s.add_argument("--date", required=True)
    s.add_argument("--deaths", type=int, required=True)
    s.add_argument("--culls", type=int, default=0)
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_log)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
