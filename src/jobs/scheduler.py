"""
Scheduling — spec §36.

Cadences supported (§36):
    once | hourly | daily | weekly | interval | cron | event

Design:
    - `Schedule` is a value object stored on a task version.
    - `next_fire_time(schedule, now)` returns the next UTC datetime.
    - Timezone-aware: schedules carry an explicit timezone and evaluate in it.
    - Version pinning (§36.2): a scheduled job runs against the TaskSpec
      version captured at scheduling time. That's enforced by storing
      `task_version` on the Schedule and never mutating it.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Cadence types
# ---------------------------------------------------------------------------

class Cadence(str, Enum):
    ONCE = "once"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    INTERVAL = "interval"
    CRON = "cron"
    EVENT = "event"


_WEEKDAYS = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}


@dataclass
class Schedule:
    cadence: Cadence = Cadence.ONCE
    timezone: str = "UTC"

    # once
    run_at: Optional[str] = None            # ISO 8601

    # daily / weekly
    at_time: str = "09:00"                  # "HH:MM" in the local tz
    weekdays: list[str] = field(default_factory=list)   # for weekly: ["mon","wed"]

    # interval
    interval_seconds: int = 0

    # cron
    cron_expression: str = ""               # "min hour dom month dow"

    # version pinning (§36.2)
    task_version: int = 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cadence"] = self.cadence.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Schedule":
        d = dict(data)
        d["cadence"] = Cadence(d.get("cadence", "once"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Next-fire calculation
# ---------------------------------------------------------------------------

def next_fire_time(schedule: Schedule, now: Optional[datetime] = None) -> Optional[datetime]:
    """
    Return the next UTC datetime the schedule should fire at, or None for
    one-shot schedules in the past or for event-driven schedules.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    try:
        tz = ZoneInfo(schedule.timezone)
    except Exception:
        tz = timezone.utc

    local_now = now.astimezone(tz)

    if schedule.cadence == Cadence.ONCE:
        return _next_once(schedule, now)

    if schedule.cadence == Cadence.HOURLY:
        next_local = local_now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return next_local.astimezone(timezone.utc)

    if schedule.cadence == Cadence.DAILY:
        hh, mm = _parse_hhmm(schedule.at_time)
        candidate = local_now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if candidate <= local_now:
            candidate += timedelta(days=1)
        return candidate.astimezone(timezone.utc)

    if schedule.cadence == Cadence.WEEKLY:
        hh, mm = _parse_hhmm(schedule.at_time)
        wanted = _weekdays_from_names(schedule.weekdays) or [local_now.weekday()]
        for offset in range(0, 14):
            candidate = (local_now + timedelta(days=offset)).replace(
                hour=hh, minute=mm, second=0, microsecond=0,
            )
            if candidate.weekday() not in wanted:
                continue
            if candidate > local_now:
                return candidate.astimezone(timezone.utc)
        return None

    if schedule.cadence == Cadence.INTERVAL:
        if schedule.interval_seconds <= 0:
            return None
        return now + timedelta(seconds=schedule.interval_seconds)

    if schedule.cadence == Cadence.CRON:
        return _next_cron(schedule.cron_expression, local_now, tz)

    if schedule.cadence == Cadence.EVENT:
        return None

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_once(schedule: Schedule, now: datetime) -> Optional[datetime]:
    if not schedule.run_at:
        return None
    try:
        dt = datetime.fromisoformat(schedule.run_at)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt if dt > now else None


def _parse_hhmm(s: str) -> tuple[int, int]:
    try:
        hh, mm = s.split(":")
        return int(hh) % 24, int(mm) % 60
    except (ValueError, AttributeError):
        return 9, 0


def _weekdays_from_names(names: list[str]) -> list[int]:
    out = []
    for n in names:
        idx = _WEEKDAYS.get(n.lower()[:3])
        if idx is not None:
            out.append(idx)
    return out


def _next_cron(expr: str, local_now: datetime, tz) -> Optional[datetime]:
    """
    Minimal cron parser — supports only `M H * * *`, `M H * * DOW`,
    `*/N H * * *`, `M */N * * *`. Full cron is deferred.
    """
    if not expr:
        return None
    parts = expr.split()
    if len(parts) != 5:
        return None
    minute_f, hour_f, _dom, _mon, dow_f = parts

    try:
        minutes = _cron_field(minute_f, 0, 59)
        hours = _cron_field(hour_f, 0, 23)
    except ValueError:
        return None

    dows: list[int]
    if dow_f == "*":
        dows = list(range(7))
    else:
        dows = []
        for tok in dow_f.split(","):
            idx = _WEEKDAYS.get(tok.strip().lower()[:3])
            if idx is None:
                try:
                    idx = int(tok) % 7
                except ValueError:
                    continue
            dows.append(idx)
        if not dows:
            dows = list(range(7))

    # Search the next 366 days, minute by minute — bounded and simple.
    for day_offset in range(0, 366):
        day = (local_now + timedelta(days=day_offset)).date()
        if day.weekday() not in dows:
            continue
        for h in sorted(hours):
            for m in sorted(minutes):
                candidate = datetime.combine(day, _time(h, m), tzinfo=tz)
                if candidate > local_now:
                    return candidate.astimezone(timezone.utc)
    return None


def _time(h: int, m: int):
    from datetime import time as _t
    return _t(hour=h, minute=m)


def _cron_field(field: str, lo: int, hi: int) -> list[int]:
    if field == "*":
        return list(range(lo, hi + 1))
    if field.startswith("*/"):
        try:
            step = int(field[2:])
        except ValueError:
            raise ValueError(f"bad cron step: {field}")
        if step <= 0:
            raise ValueError(f"cron step must be > 0: {field}")
        return list(range(lo, hi + 1, step))
    out = []
    for tok in field.split(","):
        try:
            v = int(tok)
        except ValueError:
            raise ValueError(f"bad cron token: {tok}")
        if not (lo <= v <= hi):
            raise ValueError(f"cron token out of range: {tok}")
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base = datetime(2026, 9, 24, 10, 30, 0, tzinfo=timezone.utc)

    # ONCE in the future
    s = Schedule(cadence=Cadence.ONCE, run_at="2026-09-24T12:00:00+00:00")
    nxt = next_fire_time(s, now=base)
    assert nxt == datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)

    # ONCE in the past -> None
    s = Schedule(cadence=Cadence.ONCE, run_at="2020-01-01T00:00:00+00:00")
    assert next_fire_time(s, now=base) is None

    # HOURLY
    s = Schedule(cadence=Cadence.HOURLY)
    nxt = next_fire_time(s, now=base)
    assert nxt == datetime(2026, 9, 24, 11, 0, tzinfo=timezone.utc)

    # DAILY at 09:00 UTC, now is 10:30 -> tomorrow
    s = Schedule(cadence=Cadence.DAILY, at_time="09:00", timezone="UTC")
    nxt = next_fire_time(s, now=base)
    assert nxt == datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)

    # DAILY at 15:00 UTC, now is 10:30 -> today
    s = Schedule(cadence=Cadence.DAILY, at_time="15:00", timezone="UTC")
    nxt = next_fire_time(s, now=base)
    assert nxt == datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc)

    # WEEKLY on Monday 09:00, now is Thursday -> next Monday
    s = Schedule(cadence=Cadence.WEEKLY, at_time="09:00",
                 weekdays=["mon"], timezone="UTC")
    nxt = next_fire_time(s, now=base)
    assert nxt is not None
    assert nxt.weekday() == 0

    # INTERVAL
    s = Schedule(cadence=Cadence.INTERVAL, interval_seconds=3600)
    nxt = next_fire_time(s, now=base)
    assert nxt == base + timedelta(hours=1)

    # CRON: every day at 08:00
    s = Schedule(cadence=Cadence.CRON, cron_expression="0 8 * * *", timezone="UTC")
    nxt = next_fire_time(s, now=base)
    assert nxt == datetime(2026, 9, 25, 8, 0, tzinfo=timezone.utc)

    # CRON: every 15 min
    s = Schedule(cadence=Cadence.CRON, cron_expression="*/15 * * * *", timezone="UTC")
    nxt = next_fire_time(s, now=base)
    assert nxt == datetime(2026, 9, 24, 10, 45, tzinfo=timezone.utc)

    # EVENT returns None
    s = Schedule(cadence=Cadence.EVENT)
    assert next_fire_time(s, now=base) is None

    # Timezone-aware: 09:00 in Karachi (UTC+5), now 10:30 UTC = 15:30 Karachi
    # -> next fire is tomorrow 09:00 Karachi = tomorrow 04:00 UTC
    s = Schedule(cadence=Cadence.DAILY, at_time="09:00", timezone="Asia/Karachi")
    nxt = next_fire_time(s, now=base)
    assert nxt == datetime(2026, 9, 25, 4, 0, tzinfo=timezone.utc)

    print("Scheduler OK.")