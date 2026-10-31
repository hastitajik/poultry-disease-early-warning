"""Recover the RECORDING time of a clip from its file name / folder.

DVR exports seen on this farm:
  20251118184503423_fps15.mp4              -> 2025-11-18 18:45:03.423
  2025-12-09/05.00.00-05.05.00RO000.h264   -> date from folder, time from name

Files that went through screenshot / export tools often get a SECOND date
appended (e.g. ..._20260425_221831). The first match is the recording time,
so we always take the earliest occurrence in the name.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

_DT_RUN = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{3})?(?!\d)")
_TIME_RANGE = re.compile(r"(?<!\d)(\d{2})[._:](\d{2})[._:](\d{2})-(\d{2})[._:](\d{2})[._:](\d{2})")
_ISO_LIKE = re.compile(
    r"(?<!\d)(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ T_-]+(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})(?!\d)")
_DATE = re.compile(r"(?<!\d)(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})(?!\d)")


class TimestampError(ValueError):
    pass


def _valid(y, mo, d, h=0, mi=0, s=0) -> datetime | None:
    try:
        return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
    except ValueError:
        return None


def _date_from_parents(path: Path) -> date | None:
    for parent in path.parents:
        m = _DATE.search(parent.name)
        if m:
            dt = _valid(*m.groups())
            if dt:
                return dt.date()
    return None


def parse_clip_time(path: str | Path, regex: str = "") -> datetime:
    """Return the recording start time of a clip (naive local time)."""
    p = Path(path)
    name = p.name

    if regex:
        m = re.search(regex, name)
        if not m:
            raise TimestampError(f"custom filename_regex did not match {name!r}")
        g = m.groupdict()
        if "year" not in g:
            d = _date_from_parents(p)
            if d is None:
                raise TimestampError(f"no date in regex or folder for {name!r}")
            g = {"year": d.year, "month": d.month, "day": d.day, **g}
        dt = _valid(g["year"], g["month"], g["day"], g.get("hour", 0), g.get("minute", 0),
                    g.get("second", 0))
        if dt is None:
            raise TimestampError(f"invalid date/time in {name!r}")
        return dt

    # 1) time range "HH.MM.SS-HH.MM.SS" (date from folder) — checked before the
    #    long digit run because such names often carry an export date as well.
    m = _TIME_RANGE.search(name)
    if m:
        d = _date_from_parents(p)
        if d is None:
            raise TimestampError(
                f"{name!r} has only a time of day; put clips in a YYYY-MM-DD folder")
        h, mi, s = m.groups()[:3]
        dt = _valid(d.year, d.month, d.day, h, mi, s)
        if dt:
            return dt

    # 2) earliest of: 14/17-digit run, or ISO-like date+time
    candidates = []
    for rx in (_DT_RUN, _ISO_LIKE):
        for m in rx.finditer(name):
            dt = _valid(*m.groups()[:6])
            if dt:
                if rx is _DT_RUN and m.group(7):
                    dt = dt.replace(microsecond=int(m.group(7)) * 1000)
                candidates.append((m.start(), dt))
    if candidates:
        return min(candidates)[1]

    raise TimestampError(f"cannot read a recording time from {name!r}")


def slot_start(ts: datetime, slot_minutes: int) -> datetime:
    """Floor a timestamp to its slot (e.g. 05:47:54 -> 05:45:00 for 5-minute slots)."""
    minute = (ts.minute // slot_minutes) * slot_minutes
    return ts.replace(minute=minute, second=0, microsecond=0)
