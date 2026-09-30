"""
Tests for GET /v1/batches/{id}/result.

The endpoint concatenates every succeeded chunk's records in chunk
order, includes a per-chunk manifest describing what was and wasn't
merged, and exposes a `complete` flag plus a `?require_complete=1`
strict mode.

Covers:
    - auth required
    - store not configured -> 503
    - unknown batch -> 404
    - cross-tenant -> 404
    - empty (all-queued) batch -> partial, complete=false
    - single succeeded chunk
    - multiple succeeded chunks concatenated in chunk order
    - failed chunk excluded from records but listed in manifest
    - pending chunk excluded from records but listed in manifest
    - ?require_complete=1 returns 409 when incomplete
    - ?require_complete=1 returns 200 when complete
    - store read failure -> 500
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


def _run(coro):
    return asyncio.run(coro)


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _make_batch(n_chunks: int = 2, client_id: str = "acme") -> Batch:
    return Batch.create(
        task_id="t-1",
        task_version=1,
        client_id=client_id,
        urls=[f"https://93.184.216.{34 + i}/p" for i in range(n_chunks)],
        chunk_size=1,
    )


def _complete_chunk(batch: Batch, index: int, records: list[dict]) -> None:
    """Move one chunk through QUEUED -> RUNNING -> SUCCEEDED."""
    if batch.state == BatchState.QUEUED:
        batch.state = BatchState.RUNNING
    chunk = batch.chunks[index]
    chunk.mark_running()
    chunk.mark_succeeded(records_count=len(records), records=records)
    batch._refresh_counts()


def _fail_chunk(batch: Batch, index: int, error: str) -> None:
    """Move one chunk through QUEUED -> RUNNING -> FAILED (no retry)."""
    if batch.state == BatchState.QUEUED:
        batch.state = BatchState.RUNNING
    chunk = batch.chunks[index]
    chunk.max_attempts = 1   # so mark_failed goes straight to FAILED
    chunk.mark_running()
    chunk.mark_failed(error)
    batch._refresh_counts()


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


# ===========================================================================
# Auth / config
# ===========================================================================

def test_result_requires_auth():
    async def scenario():
        async with _make_ctx() as (c, _, _, store):
            b = _make_batch()
            store.save(b)
            r = await c.get(f"/v1/batches/{b.batch_id}/result")
            assert r.status_code == 401
            assert r.json()["error"]["code"] == "unauthorized"
    _run(scenario())


def test_result_503_when_store_missing():
    async def scenario():
        async with _make_ctx(with_store=False) as (c, key, _, _):
            r = await c.get(
                "/v1/batches/anything/result", headers=_auth(key),
            )
            assert r.status_code == 503
            assert r.json()["error"]["code"] == "batch_unavailable"
    _run(scenario())


# ===========================================================================
# Lookup
# ===========================================================================

def test_result_unknown_batch_404():
    async def scenario():
        async with _make_ctx() as (c, key, _, _):
            r = await c.get(
                "/v1/batches/does-not-exist/result", headers=_auth(key),
            )
            assert r.status_code == 404
            assert r.json()["error"]["code"] == "not_found"
    _run(scenario())


def test_result_cross_tenant_404():
    async def scenario():
        async with _make_ctx() as (c, _, other_key, store):
            b = _make_batch(client_id="acme")
            store.save(b)
            r = await c.get(
                f"/v1/batches/{b.batch_id}/result",
                headers=_auth(other_key),
            )
            assert r.status_code == 404
    _run(scenario())


# ===========================================================================
# Merge
# ===========================================================================

def test_result_empty_batch_is_partial():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch(n_chunks=2)
            store.save(b)
            r = await c.get(
                f"/v1/batches/{b.batch_id}/result", headers=_auth(key),
            )
            assert r.status_code == 200
            body = r.json()
            assert body["complete"] is False
            assert body["records_count"] == 0
            assert body["records"] == []
            assert body["pending_chunks"] == 2
            assert body["completed_chunks"] == 0
            assert body["failed_chunks"] == 0
            assert len(body["chunk_manifest"]) == 2
    _run(scenario())


def test_result_single_succeeded_chunk():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch(n_chunks=1)
            _complete_chunk(b, 0, [{"a": 1}, {"a": 2}])
            store.save(b)
            r = await c.get(
                f"/v1/batches/{b.batch_id}/result", headers=_auth(key),
            )
            body = r.json()
            assert body["complete"] is True
            assert body["records_count"] == 2
            assert body["records"] == [{"a": 1}, {"a": 2}]
            assert body["completed_chunks"] == 1
    _run(scenario())


def test_result_multiple_chunks_concatenated_in_order():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch(n_chunks=3)
            _complete_chunk(b, 0, [{"i": 0}])
            _complete_chunk(b, 1, [{"i": 1}])
            _complete_chunk(b, 2, [{"i": 2}])
            store.save(b)
            r = await c.get(
                f"/v1/batches/{b.batch_id}/result", headers=_auth(key),
            )
            body = r.json()
            assert body["records_count"] == 3
            assert body["records"] == [{"i": 0}, {"i": 1}, {"i": 2}]
            assert body["complete"] is True
    _run(scenario())


def test_result_failed_chunk_excluded_from_records_but_in_manifest():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch(n_chunks=2)
            _complete_chunk(b, 0, [{"ok": True}])
            _fail_chunk(b, 1, "boom")
            store.save(b)

            r = await c.get(
                f"/v1/batches/{b.batch_id}/result", headers=_auth(key),
            )
            body = r.json()
            assert body["records"] == [{"ok": True}]
            assert body["records_count"] == 1
            assert body["failed_chunks"] == 1
            assert body["complete"] is False

            manifest = {m["index"]: m for m in body["chunk_manifest"]}
            assert manifest[1]["state"] == "failed"
            assert manifest[1]["error"] == "boom"
    _run(scenario())


def test_result_pending_chunk_in_manifest_only():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch(n_chunks=2)
            _complete_chunk(b, 0, [{"i": 0}])
            # chunk 1 left QUEUED
            store.save(b)
            r = await c.get(
                f"/v1/batches/{b.batch_id}/result", headers=_auth(key),
            )
            body = r.json()
            assert body["records"] == [{"i": 0}]
            assert body["pending_chunks"] == 1
            assert body["complete"] is False
            states = [m["state"] for m in body["chunk_manifest"]]
            assert "queued" in states
    _run(scenario())


# ===========================================================================
# require_complete
# ===========================================================================

def test_result_require_complete_409_when_incomplete():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch(n_chunks=2)
            store.save(b)
            r = await c.get(
                f"/v1/batches/{b.batch_id}/result?require_complete=1",
                headers=_auth(key),
            )
            assert r.status_code == 409
            assert r.json()["error"]["code"] == "batch_not_complete"
    _run(scenario())


def test_result_require_complete_200_when_complete():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch(n_chunks=1)
            _complete_chunk(b, 0, [{"a": 1}])
            store.save(b)
            r = await c.get(
                f"/v1/batches/{b.batch_id}/result?require_complete=1",
                headers=_auth(key),
            )
            assert r.status_code == 200
            assert r.json()["complete"] is True
    _run(scenario())


def test_result_require_complete_accepts_true_string():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            b = _make_batch()
            store.save(b)
            r = await c.get(
                f"/v1/batches/{b.batch_id}/result?require_complete=true",
                headers=_auth(key),
            )
            assert r.status_code == 409
    _run(scenario())


# ===========================================================================
# Failure
# ===========================================================================

def test_result_store_read_failure_500():
    async def scenario():
        async with _make_ctx() as (c, key, _, store):
            store.raise_on_read = True
            r = await c.get(
                "/v1/batches/whatever/result", headers=_auth(key),
            )
            assert r.status_code == 500
            assert r.json()["error"]["code"] == "batch_read_failed"
    _run(scenario())