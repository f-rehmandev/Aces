"""Unit tests for scheduling (spec §36)."""
from datetime import datetime, timedelta, timezone

import pytest
import asyncio
from datetime import datetime, timedelta, timezone

from src.jobs.scheduler import (
    Schedule, Cadence, next_fire_time,
)


BASE = datetime(2026, 9, 24, 10, 30, 0, tzinfo=timezone.utc)   # Thursday


# --- ONCE --------------------------------------------------------------

def test_once_future_returns_time():
    s = Schedule(cadence=Cadence.ONCE, run_at="2026-09-24T12:00:00+00:00")
    assert next_fire_time(s, now=BASE) == datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def test_once_past_returns_none():
    s = Schedule(cadence=Cadence.ONCE, run_at="2020-01-01T00:00:00+00:00")
    assert next_fire_time(s, now=BASE) is None


def test_once_no_run_at_returns_none():
    s = Schedule(cadence=Cadence.ONCE)
    assert next_fire_time(s, now=BASE) is None


# --- HOURLY ------------------------------------------------------------

def test_hourly_next_hour():
    s = Schedule(cadence=Cadence.HOURLY)
    nxt = next_fire_time(s, now=BASE)
    assert nxt == datetime(2026, 9, 24, 11, 0, tzinfo=timezone.utc)


# --- DAILY -------------------------------------------------------------

def test_daily_before_time_returns_today():
    s = Schedule(cadence=Cadence.DAILY, at_time="15:00", timezone="UTC")
    nxt = next_fire_time(s, now=BASE)
    assert nxt == datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc)


def test_daily_after_time_returns_tomorrow():
    s = Schedule(cadence=Cadence.DAILY, at_time="09:00", timezone="UTC")
    nxt = next_fire_time(s, now=BASE)
    assert nxt == datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)


def test_daily_timezone_aware():
    # 09:00 in Karachi (UTC+5) = 04:00 UTC
    s = Schedule(cadence=Cadence.DAILY, at_time="09:00", timezone="Asia/Karachi")
    nxt = next_fire_time(s, now=BASE)
    # Now is 15:30 Karachi, so next 09:00 Karachi is tomorrow.
    assert nxt == datetime(2026, 9, 25, 4, 0, tzinfo=timezone.utc)


def test_daily_malformed_time_falls_back_to_default():
    s = Schedule(cadence=Cadence.DAILY, at_time="not-a-time")
    nxt = next_fire_time(s, now=BASE)
    assert nxt is not None


# --- WEEKLY ------------------------------------------------------------

def test_weekly_next_monday_from_thursday():
    s = Schedule(cadence=Cadence.WEEKLY, at_time="09:00",
                 weekdays=["mon"], timezone="UTC")
    nxt = next_fire_time(s, now=BASE)
    assert nxt is not None
    assert nxt.weekday() == 0      # Monday
    assert nxt.hour == 9


def test_weekly_empty_weekdays_falls_back_to_today():
    s = Schedule(cadence=Cadence.WEEKLY, at_time="15:00",
                 weekdays=[], timezone="UTC")
    nxt = next_fire_time(s, now=BASE)
    assert nxt is not None
    assert nxt.weekday() == BASE.weekday()


# --- INTERVAL ----------------------------------------------------------

def test_interval_returns_now_plus_seconds():
    s = Schedule(cadence=Cadence.INTERVAL, interval_seconds=3600)
    nxt = next_fire_time(s, now=BASE)
    assert nxt == BASE + timedelta(hours=1)


def test_interval_zero_returns_none():
    s = Schedule(cadence=Cadence.INTERVAL, interval_seconds=0)
    assert next_fire_time(s, now=BASE) is None


# --- CRON --------------------------------------------------------------

def test_cron_daily_at_8am():
    s = Schedule(cadence=Cadence.CRON, cron_expression="0 8 * * *", timezone="UTC")
    nxt = next_fire_time(s, now=BASE)
    assert nxt == datetime(2026, 9, 25, 8, 0, tzinfo=timezone.utc)


def test_cron_every_15_minutes():
    s = Schedule(cadence=Cadence.CRON, cron_expression="*/15 * * * *", timezone="UTC")
    nxt = next_fire_time(s, now=BASE)
    assert nxt == datetime(2026, 9, 24, 10, 45, tzinfo=timezone.utc)


def test_cron_specific_minute_list():
    s = Schedule(cadence=Cadence.CRON, cron_expression="0,30 * * * *", timezone="UTC")
    nxt = next_fire_time(s, now=BASE)
    assert nxt == datetime(2026, 9, 24, 11, 0, tzinfo=timezone.utc)


def test_cron_weekday_specific():
    s = Schedule(cadence=Cadence.CRON, cron_expression="0 8 * * mon", timezone="UTC")
    nxt = next_fire_time(s, now=BASE)
    assert nxt is not None
    assert nxt.weekday() == 0


def test_cron_malformed_returns_none():
    s = Schedule(cadence=Cadence.CRON, cron_expression="garbage", timezone="UTC")
    assert next_fire_time(s, now=BASE) is None


# --- EVENT -------------------------------------------------------------

def test_event_returns_none():
    s = Schedule(cadence=Cadence.EVENT)
    assert next_fire_time(s, now=BASE) is None


# --- Serialization -----------------------------------------------------

def test_schedule_to_dict():
    s = Schedule(cadence=Cadence.DAILY, at_time="08:00", timezone="UTC")
    d = s.to_dict()
    assert d["cadence"] == "daily"
    assert d["at_time"] == "08:00"


def test_schedule_from_dict_roundtrip():
    s = Schedule(cadence=Cadence.WEEKLY, at_time="10:00",
                 weekdays=["mon", "wed"], task_version=3)
    s2 = Schedule.from_dict(s.to_dict())
    assert s2.cadence == Cadence.WEEKLY
    assert s2.weekdays == ["mon", "wed"]
    assert s2.task_version == 3

def test_wire_to_jobs_schedule_converter():
    """
    Task 13 regression: TaskSpec.schedule.Schedule.to_jobs_schedule()
    must produce a jobs.scheduler.Schedule with matching fields.
    """
    from src.core.task_spec import Schedule as WireSchedule
    from src.jobs.scheduler import Cadence, Schedule as JobsSchedule

    wire = WireSchedule(
        cadence="weekly",
        timezone="Asia/Karachi",
        at_time="14:30",
        weekdays=["mon", "fri"],
        interval_seconds=0,
        cron_expression="",
    )
    jobs = wire.to_jobs_schedule()

    assert isinstance(jobs, JobsSchedule)
    assert jobs.cadence == Cadence.WEEKLY
    assert jobs.timezone == "Asia/Karachi"
    assert jobs.at_time == "14:30"
    assert jobs.weekdays == ["mon", "fri"]


def test_wire_to_jobs_schedule_unknown_cadence_falls_back_to_once():
    from src.core.task_spec import Schedule as WireSchedule
    from src.jobs.scheduler import Cadence

    wire = WireSchedule(cadence="not-a-real-cadence")
    jobs = wire.to_jobs_schedule()
    assert jobs.cadence == Cadence.ONCE

# --- version pinning ---------------------------------------------------

def test_task_version_pinned_in_dict():
    s = Schedule(cadence=Cadence.DAILY, task_version=5)
    assert s.to_dict()["task_version"] == 5

def test_scheduler_enqueues_pinned_task_version():
    async def scenario():
        async with _make_ctx(start_scheduler=False) as c:
            task = c.registry.get_task("acme", c.task_id)

            assert task is not None
            assert task.schedule.task_version == 1

            await c.scheduler.run_once()

            c.clock_state[0] = datetime(
                2026,
                1,
                1,
                9,
                1,
                tzinfo=timezone.utc,
            )

            result = await c.scheduler.run_once()

            assert result["fired"] == 1

            jobs = c.registry.list_jobs("acme")
            assert len(jobs) == 1

            job = jobs[0]

            assert job["task_version"] == 1

            queued = await c.backend.get(job["job_id"])

            assert queued is not None
            assert queued.payload["task_version"] == 1

        asyncio.run(scenario())