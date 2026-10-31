"""Alert engine: rules -> events, with persistence, cooldown and notifications.

Messages are in Persian for the farmer / vet. The recommended actions follow
the project PDF (signal 3): inspect the area, isolate sick birds, take samples,
call the vet, review feed / water / environment.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Callable

from .config import Config, Rule
from .store import Store, iso, parse_iso

log = logging.getLogger("pews.alerts")

SEV_ORDER = {"ok": 0, "warning": 1, "critical": 2}

SIGNAL_TITLES = {
    "activity": "شاخص فعالیت گله",
    "distribution": "توزیع فضایی",
    "mortality": "تلفات",
    "circadian": "الگوی شبانه‌روزی",
}

MESSAGES = {
    "activity_drop": "فعالیت گله کمتر از حالت عادی این سن و این ساعت است.",
    "activity_spike": "فعالیت گله ناگهان بیشتر از حالت عادی شده (هیجان/ترس/مزاحمت؟).",
    "feeder_drinker_avoidance": "سهم مرغ‌ها در اطراف دانخوری و آبخوری کمتر از معمول است.",
    "periphery_crowding": "تجمع مرغ‌ها کنار دیوارها و گوشه‌ها بیشتر از معمول است.",
    "daily_mortality": "تلفات روزانه بالاتر از حد تعیین‌شده است.",
    "rhythm_loss": "الگوی فعالیت شبانه‌روزی گله با روزهای قبل همخوانی ندارد.",
    "rhythm_flattening": "اختلاف فعالیت روز و شب کم شده (ریتم گله صاف شده).",
    "mortality_candidate": "یک مرغ مدت طولانی بدون هیچ حرکتی در یک نقطه مانده است — احتمال تلف‌شدن.",
}

ACTIONS = {
    "activity": ["بازدید از سالن و نگاه به رفتار پرنده‌ها",
                 "بررسی دانخوری، آبخوری، دما، رطوبت و تهویه",
                 "اگر ادامه داشت: اطلاع به دامپزشک"],
    "distribution": ["بازدید از ناحیه‌هایی که پرنده‌ها از آن دوری می‌کنند",
                     "بررسی دما (استرس گرمایی/سرمایی) و جریان هوا",
                     "بررسی علائم عصبی: لرزش، پیچ‌خوردن گردن، چرخیدن"],
    "mortality": ["بازدید فوری از نقطه‌ی مشخص‌شده روی تصویر",
                  "جداسازی پرنده‌های بیمار و خارج کردن تلفات",
                  "نمونه‌گیری برای آزمایشگاه و تماس با دامپزشک",
                  "مرور تغییرات اخیر دان، آب و شرایط محیطی"],
    "circadian": ["مقایسه با برنامه‌ی نوری سالن (تغییری داده شده؟)",
                  "پایش دقیق‌تر گله در ۲۴ ساعت آینده",
                  "اطلاع به دامپزشک در صورت همزمانی با کاهش فعالیت"],
}


def rule_level(rule: Rule, value: float | None, z: float | None) -> str | None:
    """'ok' / 'warning' / 'critical', or None when not computable (learning)."""
    if rule.kind == "zscore":
        if z is None:
            return None
        s = -z if rule.direction == "low" else z
        return "critical" if s >= rule.crit else "warning" if s >= rule.warn else "ok"
    if value is None:
        return None
    if rule.op == ">":
        return "critical" if value > rule.crit else "warning" if value > rule.warn else "ok"
    return "critical" if value < rule.crit else "warning" if value < rule.warn else "ok"


class Notifier:
    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url

    def send(self, event: dict[str, Any]):
        log.warning("EVENT %s [%s] %s: %s", event.get("id"), event.get("severity"),
                    event.get("rule_id"), event.get("message"))
        if not self.webhook_url:
            return
        try:
            req = urllib.request.Request(
                self.webhook_url, data=json.dumps(event, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:  # never break the pipeline because of a notification
            log.error("webhook failed: %s", e)


def env_hint(store: Store, pen: str, slot: datetime) -> str:
    e = store.latest_environment(pen, before=slot)
    if not e or e["temp_c"] is None:
        return ""
    if (slot - parse_iso(e["ts"])) > timedelta(hours=3):
        return ""
    return f" دمای ثبت‌شده: {e['temp_c']:.1f}°C" + (f"، رطوبت {e['rh']:.0f}%" if e["rh"] is not None else "")


class AlertEngine:
    def __init__(self, cfg: Config, store: Store, notifier: Notifier | None = None):
        self.cfg = cfg
        self.store = store
        self.notifier = notifier or Notifier(cfg.notifications.get("webhook_url", ""))
        self.slot = timedelta(minutes=int(cfg.analysis["slot_minutes"]))

    # ------------------------------------------------------------------------
    def evaluate(self, pen: str, slot: datetime,
                 snapshots: list[str] | Callable[[], list[str]] | None = None) -> list[int]:
        """Evaluate all rules for a pen at a slot. Returns ids of new/escalated events.

        `snapshots` may be a callable so images are only copied when an event is created.
        """
        touched = []
        for rule in self.cfg.rules:
            eid = self._evaluate_rule(rule, pen, slot, snapshots or [])
            if eid:
                touched.append(eid)
        return touched

    def _recent(self, rule: Rule, pen: str, slot: datetime):
        rows = self.store.q(
            "SELECT * FROM metrics WHERE pen=? AND name=? AND slot_ts<=? ORDER BY slot_ts DESC LIMIT ?",
            (pen, rule.metric, iso(slot), rule.persist))
        return rows

    def _evaluate_rule(self, rule: Rule, pen: str, slot: datetime,
                       snapshots: list[str] | Callable[[], list[str]]) -> int | None:
        rows = self._recent(rule, pen, slot)
        if not rows or rows[0]["slot_ts"] != iso(slot):
            return None  # this metric was not produced at this slot
        levels = [rule_level(rule, r["value"], r["z"]) for r in rows]
        current = levels[0]
        existing = self.store.open_event(pen, rule.id)

        # consecutive slots? allow one missing clip between rows
        consecutive = True
        if rule.kind == "zscore" or rule.persist > 1:
            ts = [parse_iso(r["slot_ts"]) for r in rows]
            max_gap = self.slot * 2 if rule.metric != "daily_mortality_pct" else timedelta(days=2)
            consecutive = all(a - b <= max_gap for a, b in zip(ts, ts[1:]))

        breached = (len(rows) >= rule.persist and consecutive and
                    all(lv in ("warning", "critical") for lv in levels))

        if not breached:
            if existing and existing["active"] and current == "ok":
                self.store.update_event(existing["id"], active=0, updated_ts=iso(slot))
            return None

        r0 = rows[0]
        severity = current
        now = iso(slot)
        if existing and parse_iso(existing["updated_ts"]) >= slot - timedelta(minutes=rule.cooldown_min):
            escalated = SEV_ORDER[severity] > SEV_ORDER.get(existing["severity"], 0)
            self.store.update_event(existing["id"], updated_ts=now, slot_ts=now, value=r0["value"],
                                    z=r0["z"], active=1,
                                    severity=severity if escalated else existing["severity"])
            if escalated:
                self._notify(existing["id"])
                return int(existing["id"])
            return None

        msg = MESSAGES.get(rule.id, f"{rule.metric} خارج از محدوده‌ی عادی") + \
            (env_hint(self.store, pen, slot) if rule.signal in ("distribution", "activity") else "")
        details = {
            "metric": rule.metric, "baseline_median": r0["baseline_median"],
            "n_baseline": r0["n_baseline"], "age_days": r0["age_days"],
            "actions": ACTIONS.get(rule.signal, []),
            "signal_title": SIGNAL_TITLES.get(rule.signal, rule.signal),
        }
        eid = self.store.create_event(
            pen=pen, rule_id=rule.id, signal=rule.signal, severity=severity, opened_ts=now,
            updated_ts=now, slot_ts=now, value=r0["value"], z=r0["z"], message=msg,
            details=details, snapshots=snapshots() if callable(snapshots) else snapshots)
        self._notify(eid)
        return eid

    # ------------------------------------------------------------------------
    def raise_mortality_candidate(self, pen: str, slot: datetime, cand: dict, snapshots: list[str]) -> int:
        details = {"candidate_id": cand["id"], "x_m": cand["x_m"], "y_m": cand["y_m"],
                   "still_min": cand["still_min"], "first_seen": cand["first_slot"],
                   "actions": ACTIONS["mortality"], "signal_title": SIGNAL_TITLES["mortality"]}
        msg = (MESSAGES["mortality_candidate"] +
               f" محل تقریبی: {cand['x_m']:.1f} متر از دیوار چپ، {cand['y_m']:.1f} متر از انتهای سالن؛"
               f" مدت بی‌حرکتی ≈ {cand['still_min']:.0f} دقیقه.")
        eid = self.store.create_event(
            pen=pen, rule_id="mortality_candidate", signal="mortality", severity="critical",
            opened_ts=iso(slot), updated_ts=iso(slot), slot_ts=iso(slot), value=cand["still_min"],
            z=None, message=msg, details=details, snapshots=snapshots)
        self._notify(eid)
        return eid

    def _notify(self, event_id: int):
        e = self.store.event(event_id)
        if e is None:
            return
        d = dict(e)
        d["details"] = json.loads(d["details"] or "{}")
        d["snapshots"] = json.loads(d["snapshots"] or "[]")
        self.notifier.send(d)


def signal_states(cfg: Config, store: Store, pen: str) -> dict[str, dict[str, Any]]:
    """Current state of each of the 4 signals for the dashboard."""
    out: dict[str, dict[str, Any]] = {}
    for sig in ("activity", "distribution", "mortality", "circadian"):
        out[sig] = {"title": SIGNAL_TITLES[sig], "state": "learning", "metrics": {}, "events": 0}
    for rule in cfg.rules:
        r = store.latest_metric(pen, rule.metric)
        st = out[rule.signal]
        lv = rule_level(rule, r["value"], r["z"]) if r else None
        if r:
            st["metrics"][rule.metric] = {
                "value": r["value"], "z": r["z"], "baseline": r["baseline_median"],
                "n_baseline": r["n_baseline"], "slot_ts": r["slot_ts"]}
        if lv is None:
            continue
        if st["state"] == "learning" or SEV_ORDER[lv] > SEV_ORDER.get(st["state"], -1):
            st["state"] = lv
    # active events push the state up
    for e in store.q("SELECT signal, severity FROM events WHERE pen=? AND status='open' AND active=1",
                     (pen,)):
        st = out.get(e["signal"])
        if st is None:
            continue
        st["events"] += 1
        if st["state"] == "learning" or SEV_ORDER[e["severity"]] > SEV_ORDER.get(st["state"], -1):
            st["state"] = e["severity"]
    # mortality: open candidates
    cands = store.candidates(pen, ("open", "alerted"))
    alerted = [c for c in cands if c["status"] == "alerted"]
    m = out["mortality"]
    m["candidates_watching"] = len(cands) - len(alerted)
    m["candidates_alerted"] = len(alerted)
    if alerted:
        m["state"] = "critical"
    elif m["state"] == "learning" and store.latest_metric(pen, "activity_index"):
        m["state"] = "ok"
    return out
