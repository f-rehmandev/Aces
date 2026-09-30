"""
Unit tests for the batch lifecycle controls (spec §14.3C):

    POST /v1/batches/{id}/pause
    POST /v1/batches/{id}/resume
    POST /v1/batches/{id}/cancel

Covers:
    - auth: missing/invalid key → 401
    - batch_store not configured → 503
    - unknown batch → 404, cross-tenant → 404
    - valid transitions update the persisted state
    - idempotent pause/cancel return changed=False
    - invalid transitions (resume not-paused, pause/cancel terminal) → 409
    - store read/write failures surface as 500, not crashes
    - read/write raise paths never leak the raw exception type to a 5xx
      without a code (they carry `batch_read_failed` / `batch_save_failed`)
"""
import asyncio
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.keys import ApiKeyStore
from src.api.registry import ServiceRegistry
from src.api.server import build_app
from src.jobs.batch import Batch, BatchState


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeBatchStore:
    """In-memory batch store. Sync API — matches the real BatchStore shape."""

    def __init__(self):
        self._store: dict[tuple[str, str], Batch] = {}
        self.raise_on_read = False
        self.raise_on_write = False

    def get(self, client_id: str, batch_id: str):
        if self.raise_on_read:
            raise RuntimeError("simulated read failure")
        return self._store.get((client_id, batch_id))

    def save(self, batch: Batch) -> None:
        if self.raise_on_write:
            raise RuntimeError("simulated write failure")
        self._store[(batch.client_id, batch.batch_id)] = batch


def _make_batch(
    client_id: str = "acme",
    state: BatchState = BatchState.RUNNING,
) -> Batch:
    b = Batch.create(
        task_id="t-1",
        task_version=1,
        client_id=client_id,
        urls=[
            "https://93.184.216.34/a",
            "https://93.184.216.35/b",
        ],
        chunk_size=1,
    )
    b.state = state
    return b


def _run(coro):
    return asyncio.run(coro)


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@asynccontextmanager
async def _make_ctx(*, with_batch_store: bool = True):
    ks = ApiKeyStore()
    _, key = ks.create("acme")
    _, other_key = ks.create("other")

    registry = ServiceRegistry()
    store = FakeBatchStore() if with_batch_store else None

    app = build_app(
        registry, ks,
        batch_store=store,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")

    try:
        yield client, key, other_key, store
    finally:
        await client.aclose()


# ===========================================================================
# Auth
# ===========================================================================

def test_pause_requires_auth():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", BatchState.RUNNING)
            store.save(b)
            r = await c.post(f"/v1/batches/{b.batch_id}/pause")
            assert r.status_code == 401
            assert r.json()["error"]["code"] == "unauthorized"
    _run(scenario())


def test_resume_rejects_bad_key():
    async def scenario():
        async with _make_ctx() as (c, _, _, store):
            b = _make_batch("acme", BatchState.PAUSED)
            store.save(b)
            r = await c.post(
                f"/v1/batches/{b.batch_id}/resume",
                headers=_auth("aces_wrong"),
            )
            assert r.status_code == 401
    _run(scenario())


# ===========================================================================
# Store not configured
# ===========================================================================

def test_batch_control_returns_503_when_store_missing():
    async def scenario():
        async with _make_ctx(with_batch_store=False) as (c, key, _, _):
            r = await c.post(
                "/v1/batches/anything/pause",
                headers=_auth(key),
            )
            assert r.status_code == 503
            assert r.json()["error"]["code"] == "batch_unavailable"
    _run(scenario())


# ===========================================================================
# 404 paths
# ===========================================================================

def test_pause_unknown_batch_404():
    async def scenario():
        async with _make_ctx() as (c, key, _, _):
            r = await c.post(
                "/v1/batches/does-not-exist/pause",
                headers=_auth(key),
            )
            assert r.status_code == 404
            assert r.json()["error"]["code"] == "not_found"
    _run(scenario())


def test_pause_cross_tenant_404():
    async def scenario():
        async with _make_ctx() as (c, _, other_key, store):
            b = _make_batch("acme", BatchState.RUNNING)
            store.save(b)
            r = await c.post(
                f"/v1/batches/{b.batch_id}/pause",
                headers=_auth(other_key),
            )
            assert r.status_code == 404
    _run(scenario())


# ===========================================================================
# Happy path: pause
# ===========================================================================

def test_pause_running_batch():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", BatchState.RUNNING)
            store.save(b)

            r = await c.post(
                f"/v1/batches/{b.batch_id}/pause",
                headers=_auth(key),
            )
            assert r.status_code == 200
            body = r.json()
            assert body["changed"] is True
            assert body["batch"]["state"] == "paused"

            persisted = store.get("acme", b.batch_id)
            assert persisted.state == BatchState.PAUSED
    _run(scenario())


def test_pause_queued_batch():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", BatchState.QUEUED)
            store.save(b)
            r = await c.post(
                f"/v1/batches/{b.batch_id}/pause",
                headers=_auth(key),
            )
            assert r.status_code == 200
            assert store.get("acme", b.batch_id).state == BatchState.PAUSED
    _run(scenario())


def test_pause_is_idempotent():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", BatchState.PAUSED)
            store.save(b)

            r = await c.post(
                f"/v1/batches/{b.batch_id}/pause",
                headers=_auth(key),
            )
            assert r.status_code == 200
            assert r.json()["changed"] is False
            assert store.get("acme", b.batch_id).state == BatchState.PAUSED
    _run(scenario())


# ===========================================================================
# 409 paths: pause
# ===========================================================================

@pytest.mark.parametrize("terminal", [
    BatchState.COMPLETED,
    BatchState.FAILED,
    BatchState.CANCELLED,
])
def test_pause_terminal_returns_409(terminal):
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", terminal)
            store.save(b)
            r = await c.post(
                f"/v1/batches/{b.batch_id}/pause",
                headers=_auth(key),
            )
            assert r.status_code == 409
            assert r.json()["error"]["code"] == "invalid_state"
    _run(scenario())


# ===========================================================================
# Happy path: resume
# ===========================================================================

def test_resume_paused_batch():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", BatchState.PAUSED)
            store.save(b)

            r = await c.post(
                f"/v1/batches/{b.batch_id}/resume",
                headers=_auth(key),
            )
            assert r.status_code == 200
            body = r.json()
            assert body["changed"] is True
            assert body["batch"]["state"] == "queued"
            assert store.get("acme", b.batch_id).state == BatchState.QUEUED
    _run(scenario())


# ===========================================================================
# 409 paths: resume
# ===========================================================================

@pytest.mark.parametrize("not_paused", [
    BatchState.QUEUED,
    BatchState.RUNNING,
    BatchState.COMPLETED,
    BatchState.FAILED,
    BatchState.CANCELLED,
])
def test_resume_non_paused_returns_409(not_paused):
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", not_paused)
            store.save(b)
            r = await c.post(
                f"/v1/batches/{b.batch_id}/resume",
                headers=_auth(key),
            )
            assert r.status_code == 409
            assert r.json()["error"]["code"] == "invalid_state"
    _run(scenario())


# ===========================================================================
# Happy path: cancel
# ===========================================================================

def test_cancel_running_batch():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", BatchState.RUNNING)
            store.save(b)

            r = await c.post(
                f"/v1/batches/{b.batch_id}/cancel",
                headers=_auth(key),
            )
            assert r.status_code == 200
            body = r.json()
            assert body["changed"] is True
            assert body["batch"]["state"] == "cancelled"
            assert store.get("acme", b.batch_id).state == BatchState.CANCELLED
    _run(scenario())


def test_cancel_paused_batch():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", BatchState.PAUSED)
            store.save(b)
            r = await c.post(
                f"/v1/batches/{b.batch_id}/cancel",
                headers=_auth(key),
            )
            assert r.status_code == 200
            assert store.get("acme", b.batch_id).state == BatchState.CANCELLED
    _run(scenario())


def test_cancel_is_idempotent():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", BatchState.CANCELLED)
            store.save(b)
            r = await c.post(
                f"/v1/batches/{b.batch_id}/cancel",
                headers=_auth(key),
            )
            assert r.status_code == 200
            assert r.json()["changed"] is False
    _run(scenario())


# ===========================================================================
# 409 paths: cancel
# ===========================================================================

@pytest.mark.parametrize("uncancellable", [
    BatchState.COMPLETED,
    BatchState.FAILED,
])
def test_cancel_uncancellable_returns_409(uncancellable):
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", uncancellable)
            store.save(b)
            r = await c.post(
                f"/v1/batches/{b.batch_id}/cancel",
                headers=_auth(key),
            )
            assert r.status_code == 409
            assert r.json()["error"]["code"] == "invalid_state"
    _run(scenario())


# ===========================================================================
# Store failures
# ===========================================================================

def test_store_read_failure_returns_500():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            store.raise_on_read = True
            r = await c.post(
                "/v1/batches/anything/pause",
                headers=_auth(key),
            )
            assert r.status_code == 500
            assert r.json()["error"]["code"] == "batch_read_failed"
    _run(scenario())


def test_store_write_failure_returns_500():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch("acme", BatchState.RUNNING)
            store.save(b)
            store.raise_on_write = True
            r = await c.post(
                f"/v1/batches/{b.batch_id}/pause",
                headers=_auth(key),
            )
            assert r.status_code == 500
            assert r.json()["error"]["code"] == "batch_save_failed"
    _run(scenario())