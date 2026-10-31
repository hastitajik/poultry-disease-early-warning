"""Web panel + JSON API.

    pews serve --config configs/farm.yaml --port 8000
    open http://<server>:8000/

Optional HTTP basic auth: set PEWS_USER and PEWS_PASSWORD in the environment.
Run it on the farm LAN / VPN; it is not hardened for the open internet.
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..alerts import ACTIONS, SIGNAL_TITLES, signal_states
from ..config import Config
from ..pipeline import Pipeline
from ..store import iso, parse_iso

STATIC = Path(__file__).parent / "static"

SIGNAL_METRICS = {
    "activity": ["activity_index", "mean_speed"],
    "distribution": ["resource_frac", "periphery_frac", "frac_feeder", "frac_drinker",
                     "frac_wall", "frac_corner", "dispersion_entropy", "bird_coverage"],
    "mortality": ["still_candidates", "daily_mortality_pct"],
    "circadian": ["rhythm_corr", "rhythm_amplitude"],
}


class AckIn(BaseModel):
    by: str
    note: str = ""
    resolve: bool = False


class ResolveIn(BaseModel):
    outcome: str          # dead | alive | other
    by: str
    note: str = ""


class DailyLogIn(BaseModel):
    pen: str
    date: str             # YYYY-MM-DD
    deaths: int
    culls: int = 0
    note: str = ""


class EnvIn(BaseModel):
    pen: str
    ts: str | None = None
    temp_c: float | None = None
    rh: float | None = None


def _auth_dependency():
    user, pw = os.environ.get("PEWS_USER"), os.environ.get("PEWS_PASSWORD")
    if not (user and pw):
        return lambda: None
    security = HTTPBasic()

    def check(c: HTTPBasicCredentials = Depends(security)):
        if not (secrets.compare_digest(c.username, user) and secrets.compare_digest(c.password, pw)):
            raise HTTPException(401, "bad credentials", headers={"WWW-Authenticate": "Basic"})
    return check


def _row(r) -> dict:
    d = dict(r)
    for k in ("details", "snapshots"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except ValueError:
                pass
    return d


def create_app(cfg: Config, pipe: Pipeline | None = None) -> FastAPI:
    pipe = pipe or Pipeline(cfg)
    st = pipe.store
    auth = _auth_dependency()
    app = FastAPI(title="Poultry EWS", dependencies=[Depends(auth)])
    media = cfg.data_dir / "media"
    media.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(media)), name="media")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    def pen_or_404(pen: str):
        try:
            return cfg.pen(pen)
        except KeyError:
            raise HTTPException(404, f"unknown pen {pen}")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/overview")
    def overview():
        now = datetime.now()
        out = []
        for p in cfg.pens:
            last = st.q1("SELECT MAX(slot_ts) AS t FROM camera_results WHERE pen=?", (p.id,))
            last_ts = last["t"] if last else None
            out.append({
                "id": p.id, "name": p.name, "cameras": p.cameras,
                "last_slot": last_ts,
                "stale_min": (now - parse_iso(last_ts)).total_seconds() / 60 if last_ts else None,
                "signals": signal_states(cfg, st, p.id),
            })
        # flock age at the newest data (not "today": the panel may show an old flock)
        lasts = [parse_iso(p["last_slot"]) for p in out if p["last_slot"]]
        ref = max(lasts) if lasts else now
        return {"farm": cfg.farm.get("name", ""), "flock": cfg.flock_id,
                "age_days": cfg.age_days(ref), "placement_date": str(cfg.placement_date),
                "now": iso(now), "pens": out, "signal_titles": SIGNAL_TITLES, "actions": ACTIONS}

    @app.get("/api/pens/{pen}/metrics")
    def metrics(pen: str, names: str = Query(""), hours: float = 48, until: str | None = None):
        pen_or_404(pen)
        end = parse_iso(until) if until else None
        if end is None:
            last = st.q1("SELECT MAX(slot_ts) AS t FROM metrics WHERE pen=?", (pen,))
            end = parse_iso(last["t"]) if last and last["t"] else datetime.now()
        start = end - timedelta(hours=hours)
        wanted = [n for n in names.split(",") if n] or sum(SIGNAL_METRICS.values(), [])
        out = {}
        for n in wanted:
            rows = st.metric_series(pen, n, since=start, until=end)
            out[n] = [{"t": r["slot_ts"], "v": r["value"], "z": r["z"], "b": r["baseline_median"]}
                      for r in rows]
        return {"pen": pen, "from": iso(start), "to": iso(end), "series": out}

    @app.get("/api/pens/{pen}/heatmap")
    def heatmap(pen: str, cell_m: float = 0.1):
        p = pen_or_404(pen)
        d = cfg.data_dir / "grids" / pen
        files = sorted(d.glob("*_fused.npz")) if d.exists() else []
        if not files:
            return {"pen": pen, "grid": None}
        f = files[-1]
        g = np.load(f)
        occ = g["occupancy"].astype(np.float32)
        cov = g["coverage"]
        k = max(1, int(round(cell_m * float(cfg.analysis["floor_px_per_m"]))))
        r2, c2 = occ.shape[0] // k, occ.shape[1] // k
        o = occ[:r2 * k, :c2 * k].reshape(r2, k, c2, k).mean(axis=(1, 3))
        cv = cov[:r2 * k, :c2 * k].reshape(r2, k, c2, k).mean(axis=(1, 3)) > 0.5
        o = np.where(cv, o, -1.0)
        slot = datetime.strptime(f.stem[:13], "%Y%m%d_%H%M")
        lay = p.at(slot)
        return {"pen": pen, "slot": iso(slot), "width_m": p.width_m, "length_m": p.length_m,
                "cell_m": k / float(cfg.analysis["floor_px_per_m"]),
                "grid": np.round(o, 3).tolist(),
                "feeders": [vars(c) for c in lay.feeders], "drinkers": [vars(c) for c in lay.drinkers],
                "candidates": [dict(c) for c in st.candidates(pen, ("open", "alerted"))]}

    @app.get("/api/pens/{pen}/cameras")
    def cameras(pen: str):
        p = pen_or_404(pen)
        out = []
        for cid in p.cameras:
            f = media / f"latest_{cid}.jpg"
            r = st.q1("SELECT MAX(slot_ts) AS t FROM camera_results WHERE camera=?", (cid,))
            out.append({"camera": cid, "image": f"/media/latest_{cid}.jpg" if f.exists() else None,
                        "last_slot": r["t"] if r else None})
        return out

    @app.get("/api/events")
    def events(pen: str | None = None, status: str = "open", limit: int = 100):
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if pen:
            sql += " AND pen=?"
            args.append(pen)
        if status != "all":
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY (status='open') DESC, updated_ts DESC LIMIT ?"
        args.append(limit)
        return [_row(r) for r in st.q(sql, args)]

    @app.post("/api/events/{event_id}/ack")
    def ack(event_id: int, body: AckIn):
        if st.event(event_id) is None:
            raise HTTPException(404, "no such event")
        st.update_event(event_id, status="resolved" if body.resolve else "acknowledged",
                        ack_by=body.by, ack_note=body.note, ack_ts=iso(datetime.now()))
        return _row(st.event(event_id))

    @app.get("/api/pens/{pen}/candidates")
    def candidates(pen: str, status: str = "open,alerted"):
        pen_or_404(pen)
        return [dict(c) for c in st.candidates(pen, tuple(status.split(",")))]

    @app.post("/api/candidates/{cid}/resolve")
    def resolve_candidate(cid: int, body: ResolveIn):
        if body.outcome not in ("dead", "alive", "other"):
            raise HTTPException(400, "outcome must be dead | alive | other")
        c = st.q1("SELECT * FROM candidates WHERE id=?", (cid,))
        if c is None:
            raise HTTPException(404, "no such candidate")
        status = {"dead": "confirmed_dead", "alive": "alive", "other": "other"}[body.outcome]
        st.update_candidate(cid, status=status, resolved_by=body.by, note=body.note)
        if c["event_id"]:
            st.update_event(c["event_id"], status="resolved", active=0, ack_by=body.by,
                            ack_note=f"{body.outcome}: {body.note}", ack_ts=iso(datetime.now()))
        return dict(st.q1("SELECT * FROM candidates WHERE id=?", (cid,)))

    @app.post("/api/daily-log")
    def daily_log(body: DailyLogIn):
        pen_or_404(body.pen)
        pct = pipe.daily_mortality(body.pen, body.date, body.deaths, body.culls, body.note)
        return {"pen": body.pen, "date": body.date, "daily_mortality_pct": pct}

    @app.get("/api/pens/{pen}/daily-log")
    def get_daily_log(pen: str):
        pen_or_404(pen)
        return [dict(r) for r in st.q("SELECT * FROM daily_log WHERE pen=? ORDER BY date DESC", (pen,))]

    @app.post("/api/environment")
    def environment(body: EnvIn):
        pen_or_404(body.pen)
        ts = parse_iso(body.ts) if body.ts else datetime.now()
        st.put_environment(body.pen, ts, body.temp_c, body.rh)
        return {"ok": True}

    return app


def app_from_env() -> FastAPI:  # pragma: no cover - used by `uvicorn --factory`
    from ..config import load_config
    return create_app(load_config(os.environ.get("PEWS_CONFIG", "configs/farm.yaml")))
