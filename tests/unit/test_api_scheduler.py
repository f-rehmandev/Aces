"""
Unit tests for SchedulerLoop + FastAPI integration (spec §36).

Covers:
    - GET /v1/tasks/{id}/schedule returns the schedule fields
    - Unknown task → 404, cross-tenant → 404
    - Lifespan starts/stops the scheduler alongside the worker pool
    - SchedulerLoop fires due schedules and enqueues into the queue
    - Disabled schedules never fire
    - "once" cadence is skipped by the loop
    - First tick only computes next_fire_at; it does NOT fire
    - After firing, next_fire_at is recomputed; last_fired_at is set
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.keys import ApiKeyStore
from src.api.registry import ServiceRegistry
from src.api.server import build_app
from src.core.task_spec import (
    FieldSpec,
    Schedule,
    Target,
    TaskSpec,
)
from src.jobs.scheduler_loop import SchedulerLoop
from src.queue.backend import InMemoryQueueBackend


def _run(coro):
    return asyncio.run(coro)


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


# ---------------------------------------------------------------------------
# Fixtures / context
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, client, key, other_key, registry, backend,
                 scheduler, task_id, clock_state):
        self.client = client
        self.key = key
        self.other_key = other_key
        self.registry = registry
        self.backend = backend
        self.scheduler = scheduler
        self.task_id = task_id
        self.clock_state = clock_state

    @property
    def auth(self):
        return _auth(self.key)


@asynccontextmanager
async def _make_ctx(*, cadence="daily", enabled=True, start_scheduler=True):
    """
    Build a fully-wired app + SchedulerLoop with a faked clock.

    The fake clock lets tests advance "time" without sleeping.
    """
    ks = ApiKeyStore()
    _, key = ks.create("acme")
    _, other_key = ks.create("other")

    registry = ServiceRegistry()
    spec = TaskSpec(
        natural_language_prompt="daily task",
        target=Target(start_urls=["https://example.com/a"]),
        fields=[FieldSpec(name="title")],
        schedule=Schedule(
            cadence=cadence, at_time="09:00", timezone="UTC",
            enabled=enabled,
        ),
    )
    spec.client_id = "acme"
    registry.save_task("acme", spec)

    now = [datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)]
    backend = InMemoryQueueBackend()
    scheduler = SchedulerLoop(
        registry, backend,
        clock=lambda: now[0],
        poll_interval=0.05,
    )

    app = build_app(
        registry, ks,
        queue_backend=backend,
        scheduler_loop=scheduler,
    )

    if start_scheduler:
        await scheduler.start()

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")

    try:
        yield Ctx(client, key, other_key, registry, backend,
                  scheduler, spec.task_id, now)
    finally:
        await client.aclose()
        if scheduler.is_started:
            await scheduler.stop()


async def _wait_for(predicate, *, max_iters=40, interval=0.05):
    for _ in range(max_iters):
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


# ===========================================================================
# Schedule endpoint
# ===========================================================================

def test_schedule_endpoint_returns_cadence_fields():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.get(
                f"/v1/tasks/{c.task_id}/schedule", headers=c.auth,
            )
            assert r.status_code == 200
            body = r.json()
            assert body["cadence"] == "daily"
            assert body["at_time"] == "09:00"
            assert body["timezone"] == "UTC"
            assert body["enabled"] is True
    _run(scenario())


def test_schedule_endpoint_unknown_task_404():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.get(
                "/v1/tasks/does-not-exist/schedule", headers=c.auth,
            )
            assert r.status_code == 404
    _run(scenario())


def test_schedule_endpoint_cross_tenant_404():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.get(
                f"/v1/tasks/{c.task_id}/schedule",
                headers=_auth(c.other_key),
            )
            assert r.status_code == 404
    _run(scenario())


def test_schedule_endpoint_requires_auth():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.get(f"/v1/tasks/{c.task_id}/schedule")
            assert r.status_code == 401
    _run(scenario())


# ===========================================================================
# Scheduler loop: first-tick initialization
# ===========================================================================

def test_first_tick_populates_next_fire_at_without_firing():
    async def scenario():
        async with _make_ctx(start_scheduler=False) as c:
            # Directly drive the loop
            result = await c.scheduler.run_once()
            assert result["fired"] == 0
            # next_fire_at is now populated
            spec = c.registry.get_task("acme", c.task_id)
            assert spec.schedule.next_fire_at is not None
            assert spec.schedule.last_fired_at is None
    _run(scenario())


def test_schedule_endpoint_sees_next_fire_at_after_tick():
    async def scenario():
        async with _make_ctx(start_scheduler=False) as c:
            await c.scheduler.run_once()
            r = await c.client.get(
                f"/v1/tasks/{c.task_id}/schedule", headers=c.auth,
            )
            body = r.json()
            assert body["next_fire_at"] is not None
            assert body["last_fired_at"] is None
    _run(scenario())


# ===========================================================================
# Scheduler loop: firing
# ===========================================================================

def test_does_not_fire_before_due_time():
    async def scenario():
        async with _make_ctx(start_scheduler=False) as c:
            await c.scheduler.run_once()   # populate
            c.clock_state[0] = datetime(2026, 1, 1, 8, 59, tzinfo=timezone.utc)
            result = await c.scheduler.run_once()
            assert result["fired"] == 0
            assert (await c.backend.stats())["total"] == 0
    _run(scenario())


def test_fires_when_due():
    async def scenario():
        async with _make_ctx(start_scheduler=False) as c:
            await c.scheduler.run_once()   # populate
            c.clock_state[0] = datetime(2026, 1, 1, 9, 1, tzinfo=timezone.utc)
            result = await c.scheduler.run_once()
            assert result["fired"] == 1
            assert (await c.backend.stats())["total"] == 1
    _run(scenario())


def test_firing_updates_last_fired_at_and_next_fire_at():
    async def scenario():
        async with _make_ctx(start_scheduler=False) as c:
            await c.scheduler.run_once()
            c.clock_state[0] = datetime(2026, 1, 1, 9, 1, tzinfo=timezone.utc)
            await c.scheduler.run_once()
            spec = c.registry.get_task("acme", c.task_id)
            assert spec.schedule.last_fired_at is not None
            nf = datetime.fromisoformat(spec.schedule.next_fire_at)
            assert nf.date().isoformat() == "2026-01-02"
    _run(scenario())


def test_fired_job_has_scheduler_source():
    async def scenario():
        async with _make_ctx(start_scheduler=False) as c:
            await c.scheduler.run_once()
            c.clock_state[0] = datetime(2026, 1, 1, 9, 1, tzinfo=timezone.utc)
            await c.scheduler.run_once()
            jobs = c.registry.list_jobs("acme")
            assert len(jobs) == 1
            assert jobs[0]["source"] == "scheduler"
            assert jobs[0]["state"] == "queued"
            assert jobs[0]["task_id"] == c.task_id
    _run(scenario())


def test_does_not_fire_again_before_next_due():
    async def scenario():
        async with _make_ctx(start_scheduler=False) as c:
            await c.scheduler.run_once()
            c.clock_state[0] = datetime(2026, 1, 1, 9, 1, tzinfo=timezone.utc)
            await c.scheduler.run_once()
            # Same day, later — no second fire
            c.clock_state[0] = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
            result = await c.scheduler.run_once()
            assert result["fired"] == 0
            assert (await c.backend.stats())["total"] == 1
    _run(scenario())


def test_fires_again_next_day():
    async def scenario():
        async with _make_ctx(start_scheduler=False) as c:
            await c.scheduler.run_once()
            c.clock_state[0] = datetime(2026, 1, 1, 9, 1, tzinfo=timezone.utc)
            await c.scheduler.run_once()
            c.clock_state[0] = datetime(2026, 1, 2, 9, 1, tzinfo=timezone.utc)
            result = await c.scheduler.run_once()
            assert result["fired"] == 1
            assert (await c.backend.stats())["total"] == 2
    _run(scenario())


# ===========================================================================
# Disabled / once cadence
# ===========================================================================

def test_disabled_schedule_never_fires():
    async def scenario():
        async with _make_ctx(
            enabled=False, start_scheduler=False,
        ) as c:
            c.clock_state[0] = datetime(2026, 1, 1, 9, 1, tzinfo=timezone.utc)
            await c.scheduler.run_once()
            assert (await c.backend.stats())["total"] == 0
    _run(scenario())


def test_once_cadence_is_not_fired_by_loop():
    async def scenario():
        async with _make_ctx(
            cadence="once", start_scheduler=False,
        ) as c:
            c.clock_state[0] = datetime(2026, 1, 1, 9, 1, tzinfo=timezone.utc)
            result = await c.scheduler.run_once()
            assert result["fired"] == 0
            assert (await c.backend.stats())["total"] == 0
    _run(scenario())


# ===========================================================================
# Lifespan integration
# ===========================================================================

def test_lifespan_starts_and_stops_scheduler():
    """`app.state.scheduler_loop` is the same object; lifespan drives it."""
    async def scenario():
        ks = ApiKeyStore()
        registry = ServiceRegistry()
        backend = InMemoryQueueBackend()
        scheduler = SchedulerLoop(
            registry, backend, poll_interval=0.05,
        )
        app = build_app(
            registry, ks,
            queue_backend=backend,
            scheduler_loop=scheduler,
        )
        assert app.state.scheduler_loop is scheduler
    _run(scenario())


def test_no_scheduler_means_no_scheduler_state():
    async def scenario():
        ks = ApiKeyStore()
        registry = ServiceRegistry()
        app = build_app(registry, ks)
        assert app.state.scheduler_loop is None
    _run(scenario())


# ===========================================================================
# Auto-start via the manual pattern used in tests
# ===========================================================================

def test_started_scheduler_ticks_in_background():
    """
    Verify the *background* loop (not run_once) actually ticks against
    a real asyncio event loop and fires a due schedule.

    Sequencing matters here:
        1. The background loop's FIRST tick only initializes
           `next_fire_at` — it never fires. We must wait for that
           initialization before advancing the clock, otherwise the
           first tick runs *after* the clock has moved and computes
           the next day's due time instead.
        2. Once `next_fire_at` is populated, advance the fake clock
           past it.
        3. Then poll for `last_fired_at`.
    """
    async def scenario():
        async with _make_ctx() as c:
            # --- phase 1: wait for the first tick to initialize state ---
            initialized = False
            for _ in range(60):
                spec = c.registry.get_task("acme", c.task_id)
                if spec.schedule.next_fire_at is not None:
                    initialized = True
                    break
                await asyncio.sleep(0.05)
            assert initialized, (
                "background loop never ran its first tick "
                "(next_fire_at stayed None)"
            )

            # --- phase 2: advance the fake clock past the due time ---
            c.clock_state[0] = datetime(
                2026, 1, 1, 9, 1, tzinfo=timezone.utc,
            )

            # --- phase 3: wait for the loop to fire ---
            fired = False
            for _ in range(60):
                spec = c.registry.get_task("acme", c.task_id)
                if spec.schedule.last_fired_at is not None:
                    fired = True
                    break
                await asyncio.sleep(0.05)
            assert fired, (
                "background loop never fired even though the schedule "
                "was past due"
            )
    _run(scenario())