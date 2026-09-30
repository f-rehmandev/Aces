"""
Unit tests for the async job pipeline (§35 + §37).

When a `queue_backend` is supplied to `build_app`, POST /run enqueues
the job and returns immediately with 202 `queued`; a WorkerPool then
consumes it and flips the job record's state.

httpx.ASGITransport does not run ASGI lifespans, so each test starts
the pool manually and stops it in teardown. This is the canonical
pattern for driving a lifespan-dependent ASGI app under ASGITransport
and matches the manual verification used when the wiring was added.

Covers:
    - 202 / queued response, job record persisted, payload forwarded
    - Worker transitions job → completed; result retrievable
    - Multiple concurrent jobs all complete
    - Sync path still works when no queue is supplied (backward compat)
    - Auth + tenant scoping on both the run and job-view endpoints
    - Enqueue failure → 500 + job marked failed
    - Body whitelist still applied on the async path
    - GET /v1/jobs/{id} works with or without a running worker
"""
import asyncio
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.keys import ApiKeyStore
from src.api.registry import ServiceRegistry
from src.api.server import build_app
from src.core.task_spec import FieldSpec, Target, TaskSpec
from src.jobs.executor import JobExecutor
from src.queue.backend import InMemoryQueueBackend


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeScraper:
    async def fetch_html(self, url, timeout=None):
        return "<html>SENTINEL</html>"

    async def fetch_screenshot(self, url, timeout=None):
        return b"png"


class FakeExtractor:
    def __init__(self, items=None):
        self.items = items if items is not None else [
            {"title": "A", "price": "$1"},
            {"title": "B", "price": "$2"},
        ]

    def extract_list(self, html, instruction):
        return list(self.items)

    def extract_from_image(self, img, instr):
        return []


class ExplodingBackend(InMemoryQueueBackend):
    """Enqueue always fails — exercises the 500 path."""
    async def enqueue(self, job):
        raise RuntimeError("simulated enqueue failure")


# ---------------------------------------------------------------------------
# Context: builds + tears down the whole app+pool for one scenario
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, client, key, other_key, registry, backend, pool, task_id):
        self.client = client
        self.key = key
        self.other_key = other_key
        self.registry = registry
        self.backend = backend
        self.pool = pool
        self.task_id = task_id

    @property
    def auth(self):
        return {"Authorization": f"Bearer {self.key}"}

    def auth_for(self, key: str):
        return {"Authorization": f"Bearer {key}"}


@asynccontextmanager
async def _make_ctx(
    *,
    start_pool: bool = True,
    backend=None,
    items=None,
    queue_backend: bool = True,
):
    ks = ApiKeyStore()
    _, key = ks.create("acme")
    _, other_key = ks.create("other")

    registry = ServiceRegistry()
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.client_id = "acme"
    registry.save_task("acme", spec)

    if queue_backend:
        backend = backend or InMemoryQueueBackend()
    else:
        backend = None

    executor = JobExecutor(registry, FakeScraper(), FakeExtractor(items))

    app = build_app(
        registry, ks,
        executor=executor,
        queue_backend=backend,
    )
    pool = app.state.worker_pool
    if pool is not None and start_pool:
        await pool.start()

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")

    ctx = Ctx(client, key, other_key, registry, backend, pool, spec.task_id)
    try:
        yield ctx
    finally:
        await client.aclose()
        if pool is not None and pool.is_started:
            await pool.stop()


def _run(coro):
    return asyncio.run(coro)


async def _wait_for_state(
    client, job_id, key, *,
    want=("completed", "failed"),
    max_iters: int = 60,
    interval: float = 0.05,
):
    auth = {"Authorization": f"Bearer {key}"}
    last = None
    for _ in range(max_iters):
        await asyncio.sleep(interval)
        r = await client.get(f"/v1/jobs/{job_id}", headers=auth)
        last = r.json().get("state")
        if last in want:
            return last
    return last


# ===========================================================================
# Basic async path
# ===========================================================================

def test_run_returns_202_with_queued_state():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.post(f"/v1/tasks/{c.task_id}/run", headers=c.auth)
            assert r.status_code == 202
            body = r.json()
            assert body["state"] == "queued"
            assert body["task_id"] == c.task_id
            assert body["client_id"] == "acme"
            assert "job_id" in body
    _run(scenario())


def test_job_record_exists_right_after_enqueue():
    """Before the worker picks it up, the job record exists in `queued`."""
    async def scenario():
        async with _make_ctx(start_pool=False) as c:
            r = await c.client.post(f"/v1/tasks/{c.task_id}/run", headers=c.auth)
            job_id = r.json()["job_id"]
            job = c.registry.get_job("acme", job_id)
            assert job is not None
            assert job["state"] == "queued"
    _run(scenario())


def test_job_payload_reaches_backend():
    """The enqueued payload carries task_id + client_id + job_id."""
    async def scenario():
        async with _make_ctx(start_pool=False) as c:
            r = await c.client.post(f"/v1/tasks/{c.task_id}/run", headers=c.auth)
            job_id = r.json()["job_id"]
            job = await c.backend.get(job_id)
            assert job is not None
            assert job.payload["task_id"] == c.task_id
            assert job.payload["client_id"] == "acme"
            assert job.payload["job_id"] == job_id
    _run(scenario())


# ===========================================================================
# Worker drains the queue
# ===========================================================================

def test_worker_transitions_job_to_completed():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.post(f"/v1/tasks/{c.task_id}/run", headers=c.auth)
            job_id = r.json()["job_id"]
            state = await _wait_for_state(c.client, job_id, c.key)
            assert state == "completed", f"final state: {state}"
    _run(scenario())


def test_completed_job_result_is_retrievable():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.post(f"/v1/tasks/{c.task_id}/run", headers=c.auth)
            job_id = r.json()["job_id"]
            await _wait_for_state(c.client, job_id, c.key)
            r = await c.client.get(f"/v1/jobs/{job_id}/result", headers=c.auth)
            assert r.status_code == 200
            assert r.json()["result"]["records_count"] == 2
    _run(scenario())


def test_job_status_view_hides_result():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.post(f"/v1/tasks/{c.task_id}/run", headers=c.auth)
            job_id = r.json()["job_id"]
            await _wait_for_state(c.client, job_id, c.key)
            r = await c.client.get(f"/v1/jobs/{job_id}", headers=c.auth)
            body = r.json()
            assert body["state"] == "completed"
            assert "result" not in body
    _run(scenario())


# ===========================================================================
# Concurrency
# ===========================================================================

def test_multiple_jobs_all_complete():
    async def scenario():
        async with _make_ctx() as c:
            job_ids = []
            for _ in range(3):
                r = await c.client.post(
                    f"/v1/tasks/{c.task_id}/run", headers=c.auth,
                )
                job_ids.append(r.json()["job_id"])
            for jid in job_ids:
                state = await _wait_for_state(c.client, jid, c.key)
                assert state == "completed", f"{jid} → {state}"
            stats = await c.backend.stats()
            assert stats["by_state"].get("completed", 0) == 3
    _run(scenario())


# ===========================================================================
# Sync path still works (backward compat)
# ===========================================================================

def test_sync_path_still_works_when_no_queue_supplied():
    async def scenario():
        async with _make_ctx(queue_backend=False) as c:
            assert c.pool is None
            r = await c.client.post(
                f"/v1/tasks/{c.task_id}/run", headers=c.auth,
            )
            # Sync path returns 200 with `completed` immediately
            assert r.status_code == 200
            assert r.json()["state"] == "completed"
    _run(scenario())


# ===========================================================================
# Auth + tenant scoping
# ===========================================================================

def test_async_run_requires_auth():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.post(f"/v1/tasks/{c.task_id}/run")
            assert r.status_code == 401
    _run(scenario())


def test_async_run_unknown_task_returns_404():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.post(
                "/v1/tasks/does-not-exist/run", headers=c.auth,
            )
            assert r.status_code == 404
    _run(scenario())


def test_async_run_other_tenant_task_returns_404():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.post(
                f"/v1/tasks/{c.task_id}/run",
                headers=c.auth_for(c.other_key),
            )
            assert r.status_code == 404
    _run(scenario())


def test_async_job_cross_tenant_404():
    async def scenario():
        async with _make_ctx() as c:
            r = await c.client.post(
                f"/v1/tasks/{c.task_id}/run", headers=c.auth,
            )
            job_id = r.json()["job_id"]
            r = await c.client.get(
                f"/v1/jobs/{job_id}",
                headers=c.auth_for(c.other_key),
            )
            assert r.status_code == 404
    _run(scenario())


# ===========================================================================
# Failure paths
# ===========================================================================

def test_enqueue_failure_returns_500():
    async def scenario():
        backend = ExplodingBackend()
        async with _make_ctx(backend=backend) as c:
            r = await c.client.post(
                f"/v1/tasks/{c.task_id}/run", headers=c.auth,
            )
            assert r.status_code == 500
            assert r.json()["error"]["code"] == "enqueue_failed"
    _run(scenario())


def test_enqueue_failure_marks_job_failed():
    async def scenario():
        backend = ExplodingBackend()
        async with _make_ctx(backend=backend) as c:
            r = await c.client.post(
                f"/v1/tasks/{c.task_id}/run", headers=c.auth,
            )
            # The job record was created then marked failed
            jobs = c.registry.list_jobs("acme")
            assert len(jobs) == 1
            assert jobs[0]["state"] == "failed"
            assert "enqueue failed" in jobs[0]["error"]
    _run(scenario())


def test_body_whitelist_still_applied():
    """Only whitelisted body keys reach the enqueued payload."""
    async def scenario():
        async with _make_ctx(start_pool=False) as c:
            r = await c.client.post(
                f"/v1/tasks/{c.task_id}/run",
                headers=c.auth,
                json={
                    "default_currency": "USD",
                    "evil_key": "should not appear",
                },
            )
            job_id = r.json()["job_id"]
            job = await c.backend.get(job_id)
            assert job.payload["default_currency"] == "USD"
            assert "evil_key" not in job.payload
    _run(scenario())


# ===========================================================================
# WorkerPool lifecycle is independent of job reads
# ===========================================================================

def test_get_job_works_without_running_pool():
    """GET /v1/jobs/{id} works whether or not the pool is running."""
    async def scenario():
        async with _make_ctx(start_pool=False) as c:
            r = await c.client.post(
                f"/v1/tasks/{c.task_id}/run", headers=c.auth,
            )
            job_id = r.json()["job_id"]
            # No worker running → job is still queued
            r = await c.client.get(f"/v1/jobs/{job_id}", headers=c.auth)
            assert r.status_code == 200
            assert r.json()["state"] == "queued"
    _run(scenario())