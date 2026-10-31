"""SQLite storage for clips, per-camera results, pen metrics, events and candidates."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    path TEXT PRIMARY KEY, camera TEXT, start_ts TEXT, slot_ts TEXT,
    status TEXT, error TEXT, processed_at TEXT
);
CREATE TABLE IF NOT EXISTS camera_results (
    slot_ts TEXT, camera TEXT, pen TEXT, start_ts TEXT, duration_s REAL, n_frames INTEGER,
    bird_px REAL, moving_px REAL, speed_sum REAL, grid_path TEXT, snapshot_path TEXT,
    PRIMARY KEY (slot_ts, camera)
);
CREATE TABLE IF NOT EXISTS still_blobs (
    slot_ts TEXT, camera TEXT, pen TEXT, x_m REAL, y_m REAL, area_m2 REAL
);
CREATE INDEX IF NOT EXISTS ix_blobs ON still_blobs(pen, slot_ts);
CREATE TABLE IF NOT EXISTS metrics (
    flock_id TEXT, pen TEXT, slot_ts TEXT, age_days INTEGER, hour INTEGER, name TEXT,
    value REAL, baseline_median REAL, baseline_mad REAL, z REAL, n_baseline INTEGER,
    PRIMARY KEY (flock_id, pen, slot_ts, name)
);
CREATE INDEX IF NOT EXISTS ix_metrics_name ON metrics(pen, name, slot_ts);
CREATE INDEX IF NOT EXISTS ix_metrics_age ON metrics(name, age_days, hour);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, pen TEXT, rule_id TEXT, signal TEXT,
    severity TEXT, status TEXT, active INTEGER, opened_ts TEXT, updated_ts TEXT, slot_ts TEXT,
    value REAL, z REAL, message TEXT, details TEXT, snapshots TEXT,
    ack_by TEXT, ack_note TEXT, ack_ts TEXT
);
CREATE INDEX IF NOT EXISTS ix_events ON events(pen, rule_id, status);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT, pen TEXT, first_slot TEXT, last_slot TEXT,
    x_m REAL, y_m REAL, area_m2 REAL, still_min REAL, active_seen INTEGER,
    status TEXT, event_id INTEGER, resolved_by TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS daily_log (
    pen TEXT, date TEXT, deaths INTEGER, culls INTEGER, note TEXT, PRIMARY KEY (pen, date)
);
CREATE TABLE IF NOT EXISTS environment (
    pen TEXT, ts TEXT, temp_c REAL, rh REAL, PRIMARY KEY (pen, ts)
);
"""


def iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def parse_iso(s: str) -> datetime:
    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @contextmanager
    def tx(self):
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def q(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, tuple(args)).fetchall()

    def q1(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Row | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    # ---- clips --------------------------------------------------------------
    def clip_done(self, path: str) -> bool:
        r = self.q1("SELECT status FROM clips WHERE path=?", (path,))
        return r is not None and r["status"] == "done"

    def mark_clip(self, path: str, camera: str, start: datetime | None, slot: datetime | None,
                  status: str, error: str = ""):
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO clips VALUES (?,?,?,?,?,?,?)",
                      (path, camera, iso(start) if start else None, iso(slot) if slot else None,
                       status, error, iso(datetime.now())))

    # ---- camera results -------------------------------------------------------
    def put_camera_result(self, slot: datetime, camera: str, pen: str, start: datetime,
                          duration_s: float, n_frames: int, bird_px: float, moving_px: float,
                          speed_sum: float, grid_path: str, snapshot_path: str,
                          blobs: list[tuple[float, float, float]]):
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO camera_results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (iso(slot), camera, pen, iso(start), duration_s, n_frames, bird_px,
                       moving_px, speed_sum, grid_path, snapshot_path))
            c.execute("DELETE FROM still_blobs WHERE slot_ts=? AND camera=?", (iso(slot), camera))
            c.executemany("INSERT INTO still_blobs VALUES (?,?,?,?,?,?)",
                          [(iso(slot), camera, pen, x, y, a) for x, y, a in blobs])

    def camera_results(self, pen: str, slot: datetime) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM camera_results WHERE pen=? AND slot_ts=?", (pen, iso(slot)))

    def blobs(self, pen: str, slot: datetime) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM still_blobs WHERE pen=? AND slot_ts=?", (pen, iso(slot)))

    # ---- metrics -------------------------------------------------------------
    def put_metric(self, flock_id: str, pen: str, slot: datetime, age_days: int, name: str,
                   value: float | None, base_med=None, base_mad=None, z=None, n_base=0):
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (flock_id, pen, iso(slot), age_days, slot.hour, name, value,
                       base_med, base_mad, z, n_base))

    def metric_series(self, pen: str, name: str, since: datetime | None = None,
                      until: datetime | None = None, flock_id: str | None = None
                      ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM metrics WHERE pen=? AND name=?"
        args: list[Any] = [pen, name]
        if flock_id:
            sql += " AND flock_id=?"
            args.append(flock_id)
        if since:
            sql += " AND slot_ts>=?"
            args.append(iso(since))
        if until:
            sql += " AND slot_ts<=?"
            args.append(iso(until))
        return self.q(sql + " ORDER BY slot_ts", args)

    def latest_metric(self, pen: str, name: str) -> sqlite3.Row | None:
        return self.q1("SELECT * FROM metrics WHERE pen=? AND name=? ORDER BY slot_ts DESC LIMIT 1",
                       (pen, name))

    # ---- events ----------------------------------------------------------------
    def open_event(self, pen: str, rule_id: str) -> sqlite3.Row | None:
        return self.q1("SELECT * FROM events WHERE pen=? AND rule_id=? AND status!='resolved' "
                       "ORDER BY updated_ts DESC LIMIT 1", (pen, rule_id))

    def create_event(self, **f) -> int:
        f.setdefault("status", "open")
        f.setdefault("active", 1)
        for k in ("details", "snapshots"):
            if not isinstance(f.get(k), str):
                f[k] = json.dumps(f.get(k) or ({} if k == "details" else []), ensure_ascii=False)
        cols = ",".join(f)
        with self.tx() as c:
            cur = c.execute(f"INSERT INTO events ({cols}) VALUES ({','.join('?' * len(f))})",
                            tuple(f.values()))
            return int(cur.lastrowid)

    def update_event(self, event_id: int, **f):
        for k in ("details", "snapshots"):
            if k in f and not isinstance(f[k], str):
                f[k] = json.dumps(f[k], ensure_ascii=False)
        sets = ",".join(f"{k}=?" for k in f)
        with self.tx() as c:
            c.execute(f"UPDATE events SET {sets} WHERE id=?", (*f.values(), event_id))

    def event(self, event_id: int) -> sqlite3.Row | None:
        return self.q1("SELECT * FROM events WHERE id=?", (event_id,))

    # ---- candidates ------------------------------------------------------------------
    def candidates(self, pen: str, statuses: tuple[str, ...] = ("open", "alerted")) -> list[sqlite3.Row]:
        marks = ",".join("?" * len(statuses))
        return self.q(f"SELECT * FROM candidates WHERE pen=? AND status IN ({marks}) ORDER BY id",
                      (pen, *statuses))

    def add_candidate(self, pen, slot: datetime, x, y, area, still_min, active) -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO candidates (pen, first_slot, last_slot, x_m, y_m, area_m2,"
                            " still_min, active_seen, status) VALUES (?,?,?,?,?,?,?,?, 'open')",
                            (pen, iso(slot), iso(slot), x, y, area, still_min, int(active)))
            return int(cur.lastrowid)

    def update_candidate(self, cid: int, **f):
        sets = ",".join(f"{k}=?" for k in f)
        with self.tx() as c:
            c.execute(f"UPDATE candidates SET {sets} WHERE id=?", (*f.values(), cid))

    # ---- logs ---------------------------------------------------------------------
    def put_daily_log(self, pen: str, date: str, deaths: int, culls: int, note: str = ""):
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO daily_log VALUES (?,?,?,?,?)",
                      (pen, date, deaths, culls, note))

    def put_environment(self, pen: str, ts: datetime, temp_c: float | None, rh: float | None):
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO environment VALUES (?,?,?,?)", (pen, iso(ts), temp_c, rh))

    def latest_environment(self, pen: str, before: datetime | None = None) -> sqlite3.Row | None:
        if before:
            return self.q1("SELECT * FROM environment WHERE pen=? AND ts<=? ORDER BY ts DESC LIMIT 1",
                           (pen, iso(before)))
        return self.q1("SELECT * FROM environment WHERE pen=? ORDER BY ts DESC LIMIT 1", (pen,))
