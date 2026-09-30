"""
Regression tests for GET /v1/batches/{id}.

This endpoint was silently broken: `_handle_batch_get` had only a
docstring as its body, and its real implementation was orphaned as
unreachable dead code after `_handle_batch_control`'s return statement.
The route returned `200 null` instead of the batch object.

Covers:
    - auth required
    - store not configured -> 503
    - unknown batch -> 404
    - cross-tenant batch -> 404
    - happy path returns the batch object
    - store read failure -> 500 (not a crash)
"""
import asyncio
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.keys import ApiKeyStore
from src.api.registry import ServiceRegistry
from src.api.server import build_app
from src.jobs.batch import Batch, BatchState


class FakeBatchStore:
    def __init__(self):
        self._store: dict[tuple[str, str], Batch] = {}
        self.raise_on_read = False

    def get(self, client_id: str, batch_id: str):
        if self.raise_on_read:
            raise RuntimeError("simulated read failure")
        return self._store.get((client_id, batch_id))

    def save(self, batch: Batch) -> None:
        self._store[(batch.client_id, batch.batch_id)] = batch


def _make_batch(client_id: str = "acme") -> Batch:
    return Batch.create(
        task_id="t-1",
        task_version=1,
        client_id=client_id,
        urls=["https://93.184.216.34/a", "https://93.184.216.35/b"],
        chunk_size=1,
    )


def _run(coro):
    return asyncio.run(coro)


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@asynccontextmanager
async def _make_ctx(*, with_store: bool = True):
    ks = ApiKeyStore()
    _, key = ks.create("acme")
    _, other_key = ks.create("other")

    registry = ServiceRegistry()
    store = FakeBatchStore() if with_store else None

    app = build_app(registry, ks, batch_store=store)
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")

    try:
        yield client, key, other_key, store
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Auth / config
# ---------------------------------------------------------------------------

def test_get_batch_requires_auth():
    async def scenario():
        async with _make_ctx() as (c, _, _, store):
            b = _make_batch()
            store.save(b)
            r = await c.get(f"/v1/batches/{b.batch_id}")
            assert r.status_code == 401
            assert r.json()["error"]["code"] == "unauthorized"
    _run(scenario())


def test_get_batch_503_when_store_missing():
    async def scenario():
        async with _make_ctx(with_store=False) as (c, key, _, _):
            r = await c.get("/v1/batches/anything", headers=_auth(key))
            assert r.status_code == 503
            assert r.json()["error"]["code"] == "batch_unavailable"
    _run(scenario())


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def test_get_batch_unknown_404():
    async def scenario():
        async with _make_ctx() as (c, key, _, _):
            r = await c.get("/v1/batches/does-not-exist", headers=_auth(key))
            assert r.status_code == 404
            assert r.json()["error"]["code"] == "not_found"
    _run(scenario())


def test_get_batch_cross_tenant_404():
    async def scenario():
        async with _make_ctx() as (c, _, other_key, store):
            b = _make_batch("acme")
            store.save(b)
            r = await c.get(
                f"/v1/batches/{b.batch_id}",
                headers=_auth(other_key),
            )
            assert r.status_code == 404
    _run(scenario())


# ---------------------------------------------------------------------------
# Happy path — regression guard for the orphaned-body bug
# ---------------------------------------------------------------------------

def test_get_batch_returns_batch_object_not_null():
    """
    Before the fix, this route returned `200 null` because the
    handler's body was orphaned. This test asserts the actual shape.
    """
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch()
            store.save(b)

            r = await c.get(f"/v1/batches/{b.batch_id}", headers=_auth(key))
            assert r.status_code == 200
            body = r.json()

            assert body["batch_id"] == b.batch_id
            assert "batch" in body
            assert body["batch"]["task_id"] == "t-1"
            assert body["batch"]["task_version"] == 1
            assert body["batch"]["client_id"] == "acme"
            assert body["batch"]["state"] == BatchState.QUEUED.value
            # Two start URLs / chunk_size=1 -> two chunks
            assert len(body["batch"]["chunks"]) == 2
    _run(scenario())


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_get_batch_store_read_failure_500():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            store.raise_on_read = True
            r = await c.get("/v1/batches/whatever", headers=_auth(key))
            assert r.status_code == 500
            assert r.json()["error"]["code"] == "batch_read_failed"
    _run(scenario())