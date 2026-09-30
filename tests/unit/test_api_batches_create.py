"""
HTTP-level tests for POST /v1/batches (spec §14.3C + §45.5).

Covers:
    - auth required
    - batch coordinator must be configured → 503
    - body must be a JSON object → 400
    - task_id required and non-empty string → 400
    - task_version required, int >= 1, not bool → 400
    - urls required, non-empty list of non-empty strings → 400
    - chunk_size int >= 1 → 400
    - max_attempts int >= 1 → 400
    - input_manifest list of strings → 400
    - task lookup scoped by client → 404
    - task version must exist → 404
    - happy path returns 201 with batch_id and batch
    - happy path persists the batch and enqueues one durable job per chunk
"""
import asyncio
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.keys import ApiKeyStore
from src.api.registry import ServiceRegistry
from src.api.server import build_app
from src.core.task_spec import FieldSpec, Target, TaskSpec
from src.jobs.batch_coordinator import BatchCoordinator
from src.queue.backend import InMemoryQueueBackend


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeBatchStore:
    """In-memory batch store matching the real BatchStore save/get shape."""

    def __init__(self):
        self._store: dict[tuple[str, str], object] = {}

    def save(self, batch):
        self._store[(batch.client_id, batch.batch_id)] = batch

    def get(self, client_id: str, batch_id: str):
        return self._store.get((client_id, batch_id))


def _run(coro):
    return asyncio.run(coro)


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _make_task(client_id: str = "acme") -> TaskSpec:
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.client_id = client_id
    return spec


@asynccontextmanager
async def _make_ctx(*, with_coordinator: bool = True):
    ks = ApiKeyStore()
    _, key = ks.create("acme")
    _, other_key = ks.create("other")

    registry = ServiceRegistry()
    spec = _make_task("acme")
    registry.save_task("acme", spec)

    store = FakeBatchStore()
    queue = InMemoryQueueBackend()
    coordinator = (
        BatchCoordinator(store, queue) if with_coordinator else None
    )

    app = build_app(
        registry, ks,
        batch_store=store,
        batch_coordinator=coordinator,
        queue_backend=queue,
        executor=None,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")

    try:
        yield client, key, other_key, registry, store, queue, spec.task_id
    finally:
        await client.aclose()


def _valid_body(task_id: str, **overrides) -> dict:
    return {
        "task_id": task_id,
        "task_version": 1,
        "urls": ["https://93.184.216.34/a"],
        **overrides,
    }


# ===========================================================================
# Auth / config
# ===========================================================================

def test_create_requires_auth():
    async def scenario():
        async with _make_ctx() as (c, _, _, _, _, _, tid):
            r = await c.post("/v1/batches", json=_valid_body(tid))
            assert r.status_code == 401
            assert r.json()["error"]["code"] == "unauthorized"
    _run(scenario())


def test_create_returns_503_when_coordinator_missing():
    async def scenario():
        async with _make_ctx(with_coordinator=False) as (
            c, key, _, _, _, _, tid
        ):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json=_valid_body(tid),
            )
            assert r.status_code == 503
            assert r.json()["error"]["code"] == "batch_unavailable"
    _run(scenario())


# ===========================================================================
# Body validation
# ===========================================================================

def test_create_rejects_missing_body():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, _):
            r = await c.post("/v1/batches", headers=_auth(key))
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "invalid_request"
    _run(scenario())


def test_create_rejects_missing_task_id():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, _):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={"task_version": 1, "urls": ["https://x/a"]},
            )
            assert r.status_code == 400
            assert "task_id" in r.json()["error"]["message"]
    _run(scenario())


def test_create_rejects_blank_task_id():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, _):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={"task_id": "   ", "task_version": 1,
                      "urls": ["https://x/a"]},
            )
            assert r.status_code == 400
    _run(scenario())


def test_create_rejects_missing_task_version():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={"task_id": tid, "urls": ["https://x/a"]},
            )
            assert r.status_code == 400
            assert "task_version" in r.json()["error"]["message"]
    _run(scenario())


@pytest.mark.parametrize("bad_version", [0, -1, "1", True, False, 1.5])
def test_create_rejects_bad_task_version(bad_version):
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={"task_id": tid, "task_version": bad_version,
                      "urls": ["https://x/a"]},
            )
            assert r.status_code == 400
    _run(scenario())


def test_create_rejects_missing_urls():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={"task_id": tid, "task_version": 1},
            )
            assert r.status_code == 400
            assert "urls" in r.json()["error"]["message"]
    _run(scenario())


def test_create_rejects_empty_urls():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={"task_id": tid, "task_version": 1, "urls": []},
            )
            assert r.status_code == 400
    _run(scenario())


def test_create_rejects_blank_url_strings():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={"task_id": tid, "task_version": 1,
                      "urls": ["https://x/a", "   "]},
            )
            assert r.status_code == 400
    _run(scenario())


def test_create_rejects_non_string_url_items():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={"task_id": tid, "task_version": 1, "urls": [123]},
            )
            assert r.status_code == 400
    _run(scenario())


@pytest.mark.parametrize("bad_chunk", [0, -1, "100", True, 1.5])
def test_create_rejects_bad_chunk_size(bad_chunk):
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json=_valid_body(tid, chunk_size=bad_chunk),
            )
            assert r.status_code == 400
    _run(scenario())


@pytest.mark.parametrize("bad_attempts", [0, -1, "3", True, 1.5])
def test_create_rejects_bad_max_attempts(bad_attempts):
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json=_valid_body(tid, max_attempts=bad_attempts),
            )
            assert r.status_code == 400
    _run(scenario())


def test_create_rejects_bad_input_manifest():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json=_valid_body(tid, input_manifest=[123]),
            )
            assert r.status_code == 400
    _run(scenario())


# ===========================================================================
# Task lookup
# ===========================================================================

def test_create_returns_404_for_unknown_task():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, _):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json=_valid_body("does-not-exist"),
            )
            assert r.status_code == 404
            assert r.json()["error"]["code"] == "not_found"
    _run(scenario())


def test_create_returns_404_for_unknown_version():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={"task_id": tid, "task_version": 99,
                      "urls": ["https://x/a"]},
            )
            assert r.status_code == 404
    _run(scenario())


def test_create_cross_tenant_task_invisible():
    async def scenario():
        async with _make_ctx() as (c, _, other_key, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(other_key),
                json=_valid_body(tid),
            )
            assert r.status_code == 404
    _run(scenario())


# ===========================================================================
# Happy path
# ===========================================================================

def test_create_success_returns_201():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json=_valid_body(tid),
            )
            assert r.status_code == 201, r.text
            body = r.json()
            assert "batch_id" in body
            assert body["batch"]["task_id"] == tid
            assert body["batch"]["task_version"] == 1
            assert body["batch"]["state"] == "queued"
    _run(scenario())


def test_create_persists_batch_to_store():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, store, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json=_valid_body(tid),
            )
            batch_id = r.json()["batch_id"]
            persisted = store.get("acme", batch_id)
            assert persisted is not None
            assert persisted.task_id == tid
    _run(scenario())


def test_create_enqueues_one_job_per_chunk():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, _, queue, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={
                    "task_id": tid, "task_version": 1,
                    "urls": [f"https://x.com/{i}" for i in range(5)],
                    "chunk_size": 2,
                },
            )
            assert r.status_code == 201
            stats = await queue.stats()
            # 5 URLs / chunk_size 2 → 3 chunks
            assert stats["total"] == 3
    _run(scenario())


def test_create_forwards_chunk_config():
    async def scenario():
        async with _make_ctx() as (c, key, _, _, store, _, tid):
            r = await c.post(
                "/v1/batches", headers=_auth(key),
                json={
                    "task_id": tid, "task_version": 1,
                    "urls": ["https://x/a", "https://x/b", "https://x/c"],
                    "chunk_size": 1,
                    "max_attempts": 7,
                },
            )
            assert r.status_code == 201
            batch_id = r.json()["batch_id"]
            b = store.get("acme", batch_id)
            assert b.chunk_size == 1
            assert all(c.max_attempts == 7 for c in b.chunks)
    _run(scenario())