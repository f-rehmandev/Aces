"""
Scheduler loop — spec §36.

Background driver that periodically checks every task's schedule and
enqueues jobs when a schedule is due. Complements `src.jobs.scheduler`,
which computes *when* things should fire but does not run itself.

Design:

    - Tracks state on `TaskSpec.schedule` itself. `next_fire_at` is
      populated on the first tick; on every subsequent tick we fire if
      `now >= next_fire_at`, then compute the next one. That avoids
      needing a separate schedules table.
    - Only schedules with `enabled=True` and cadence in
      {hourly, daily, weekly, interval, cron} are considered. "once"
      schedules are run explicitly via POST /run — the loop never fires
      them, because that would surprise the caller who created them
      expecting a single manual trigger.
    - Fire = create a job record in the registry (state=queued) and
      enqueue it into the queue backend. The WorkerPool then consumes
      it exactly like a manual /run submission.
    - Failures during a tick never stop the loop: they're logged and
      the next tick proceeds. A schedule that fails to fire is
      retried on the next tick.
    - A tick is `run_once()` and exposed for tests. `start()` spawns
      a background task that calls `run_once()` on an interval.

Not implemented here (deferred):
    - Distributed locks — if two SchedulerLoop instances run against
      the same registry, both could fire the same schedule. A single
      process owns the loop today.
    - Backfilling missed runs — if the loop was down when a schedule
      was due, that run is skipped, not queued for catch-up. Roadmap.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from src.api.registry import ServiceRegistry
from src.jobs.scheduler import next_fire_time
from src.queue.backend import QueueBackend
from src.queue.types import QueuedJob


logger = logging.getLogger("jobs.scheduler_loop")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ONCE_CADENCES = ("once",)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# SchedulerLoop
# ---------------------------------------------------------------------------

class SchedulerLoop:
    """
    Polls every task's schedule and fires due ones.

    Constructor:
        registry       — ServiceRegistry to enumerate tasks and store jobs
        queue_backend  — where fired jobs are enqueued
        clock          — injectable for tests; defaults to datetime.now(utc)
        poll_interval  — seconds between ticks in `start()`
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        queue_backend: QueueBackend,
        *,
        clock: Optional[Callable[[], datetime]] = None,
        poll_interval: float = 30.0,
    ):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        self.registry = registry
        self.queue_backend = queue_backend
        self._clock = clock or _utc_now
        self.poll_interval = float(poll_interval)

        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Spawn the background loop. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._loop(), name="aces-scheduler",
        )
        logger.info(
            f"SchedulerLoop started (poll_interval={self.poll_interval}s)"
        )

    async def stop(self) -> None:
        """Signal stop and await the background task."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._stop_event = None
        logger.info("SchedulerLoop stopped cleanly")

    @property
    def is_started(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # The tick
    # ------------------------------------------------------------------
    async def run_once(self) -> dict:
        """
        One scheduler tick. Returns a small summary dict:

            {"checked": int, "fired": int, "errors": list[str]}

        Never raises — errors land in `errors` so a caller (the
        background loop, or a test) can see what went wrong without
        the tick exploding.
        """
        now = self._clock()
        checked = 0
        fired = 0
        errors: list[str] = []

        for client_id in self.registry.known_clients():
            for spec in self.registry.list_tasks(client_id):
                checked += 1
                try:
                    did_fire = await self._check_and_fire(spec, client_id, now)
                    if did_fire:
                        fired += 1
                except Exception as e:
                    errors.append(
                        f"task {spec.task_id}: "
                        f"{type(e).__name__}: {e}"
                    )
                    logger.warning(
                        f"scheduler tick error for task {spec.task_id}: "
                        f"{type(e).__name__}: {e}"
                    )

        return {"checked": checked, "fired": fired, "errors": errors}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _check_and_fire(self, spec, client_id: str, now: datetime) -> bool:
        """Return True if the schedule fired for this task on this tick."""
        sched = spec.schedule

        if not sched.enabled:
            return False
        if sched.cadence in _ONCE_CADENCES:
            return False

        jobs_sched = sched.to_jobs_schedule()

        # --- first observation: compute and store next_fire_at, don't fire ---
        if sched.next_fire_at is None:
            nf = next_fire_time(jobs_sched, now=now)
            sched.next_fire_at = nf.isoformat() if nf else None
            return False

        due_at = _parse_iso(sched.next_fire_at)
        if due_at is None:
            # Bad state — recompute and don't fire
            nf = next_fire_time(jobs_sched, now=now)
            sched.next_fire_at = nf.isoformat() if nf else None
            return False

        if now < due_at:
            return False

        # --- fire ---
        await self._enqueue_for_task(spec, client_id)
        sched.last_fired_at = now.isoformat()

        # --- schedule the next one (relative to NOW, not due_at) ---
        nf = next_fire_time(jobs_sched, now=now)
        sched.next_fire_at = nf.isoformat() if nf else None
        return True

    async def _enqueue_for_task(self, spec, client_id: str) -> str:
        """Create a job record and enqueue it. Returns the new job_id."""
        job_id = str(uuid.uuid4())

        task_version = int(
            getattr(spec.schedule, "task_version", 1) or 1
        )

        self.registry.save_job(
            client_id,
            job_id,
            {
                "job_id": job_id,
                "task_id": spec.task_id,
                "task_version": task_version,
                "client_id": client_id,
                "state": "queued",
                "source": "scheduler",
            },
        )

        payload = {
            "task_id": spec.task_id,
            "task_version": task_version,
            "client_id": client_id,
            "job_id": job_id,
        }

        await self.queue_backend.enqueue(
            QueuedJob(
                job_id=job_id,
                payload=payload,
                client_id=client_id,
                label=f"scheduled:{spec.task_id}",
            )
        )

        return job_id

    async def _loop(self) -> None:
        while not (
            self._stop_event
            and self._stop_event.is_set()
        ):
            try:
                await self.run_once()
            except Exception as e:
                logger.warning(
                    f"scheduler loop tick failed: "
                    f"{type(e).__name__}: {e}"
                )

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.poll_interval,
                )
                return
            except asyncio.TimeoutError:
                pass

# ---------------------------------------------------------------------------
# Smoke test — in-memory registry, in-memory queue, faked clock
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.core.task_spec import (
        FieldSpec, Schedule, Target, TaskSpec,
    )
    from src.queue.backend import InMemoryQueueBackend

    async def run():
        registry = ServiceRegistry()
        backend = InMemoryQueueBackend()

        spec = TaskSpec(
            natural_language_prompt="daily task",
            target=Target(start_urls=["https://example.com/a"]),
            fields=[FieldSpec(name="title")],
        )
        spec.client_id = "acme"
        spec.schedule = Schedule(
            cadence="daily", at_time="09:00", timezone="UTC",
        )
        registry.save_task("acme", spec)

        # Fake clock so tests run instantly.
        now = [datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)]
        def clock():
            return now[0]

        loop = SchedulerLoop(
            registry, backend, clock=clock, poll_interval=0.05,
        )

        # ---- 1. First tick: computes next_fire_at, does NOT fire ----
        r = await loop.run_once()
        assert r["checked"] == 1
        assert r["fired"] == 0
        assert spec.schedule.next_fire_at is not None
        assert (await backend.stats())["total"] == 0

        # ---- 2. Advance to just before due: still no fire ----
        now[0] = datetime(2026, 1, 1, 8, 59, 0, tzinfo=timezone.utc)
        r = await loop.run_once()
        assert r["fired"] == 0
        assert (await backend.stats())["total"] == 0

        # ---- 3. Advance past due: fires once ----
        now[0] = datetime(2026, 1, 1, 9, 1, 0, tzinfo=timezone.utc)
        r = await loop.run_once()
        assert r["fired"] == 1
        assert (await backend.stats())["total"] == 1
        assert spec.schedule.last_fired_at is not None
        # next_fire_at is now tomorrow at 09:00
        nf = datetime.fromisoformat(spec.schedule.next_fire_at)
        assert nf.date().isoformat() == "2026-01-02"

        # ---- 4. Same day, later: does not fire again ----
        now[0] = datetime(2026, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
        r = await loop.run_once()
        assert r["fired"] == 0
        assert (await backend.stats())["total"] == 1

        # ---- 5. Next day, past due: fires again ----
        now[0] = datetime(2026, 1, 2, 9, 1, 0, tzinfo=timezone.utc)
        r = await loop.run_once()
        assert r["fired"] == 1
        assert (await backend.stats())["total"] == 2

        # ---- 6. Disabled schedule never fires ----
        # Temporarily point next_fire_at at a past timestamp so the
        # disabled check is meaningfully exercised. We must clear it
        # again before re-enabling — otherwise the re-enabled schedule
        # would (correctly!) fire on the next tick, which would break
        # the step-7 assertions below.
        spec.schedule.enabled = False
        spec.schedule.next_fire_at = "2026-01-01T00:00:00+00:00"
        r = await loop.run_once()
        assert r["fired"] == 0
        spec.schedule.next_fire_at = None
        spec.schedule.enabled = True

        # ---- 7. "once" cadence is not managed by the loop ----
        spec2 = TaskSpec(
            natural_language_prompt="one-shot",
            target=Target(start_urls=["https://example.com/b"]),
            fields=[FieldSpec(name="title")],
        )
        spec2.client_id = "acme"
        spec2.schedule = Schedule(cadence="once")
        registry.save_task("acme", spec2)
        r = await loop.run_once()
        # spec2 is checked but doesn't fire
        assert r["checked"] >= 2
        assert r["fired"] == 0  # spec2 has "once", and spec's next is tomorrow

        # ---- 8. Fire creates a real job record ----
        jobs = registry.list_jobs("acme")
        assert len(jobs) == 2
        assert all(j["state"] == "queued" for j in jobs)
        assert all(j.get("source") == "scheduler" for j in jobs)

        # ---- 9. start()/stop() lifecycle ----
        loop2 = SchedulerLoop(
            registry, backend, clock=clock, poll_interval=0.05,
        )
        await loop2.start()
        assert loop2.is_started
        # Let it tick once
        await asyncio.sleep(0.15)
        await loop2.stop()
        assert not loop2.is_started

        print("SchedulerLoop OK.")

    asyncio.run(run())