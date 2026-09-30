"""
Tests for batch chunk record persistence (spec §14.3C).

Before this change:
    BatchChunk only stored `records_count`. The actual records produced
    by the pipeline were discarded, so a batch could report "500 records"
    but there was no way to retrieve them.

After this change:
    `BatchChunk.records` carries the actual records. They are serialized
    into `chunk.to_dict()["records"]`, which `BatchStore` writes into the
    `batch_chunks.data` JSONB column. Backward compatible: old rows
    without the key default to [].

Covers:
    - field default
    - mark_succeeded with records
    - mark_succeeded without records leaves existing list intact
    - to_dict / from_dict round trip
    - backward compat: from_dict of a legacy chunk with no `records` key
    - end-to-end through BatchStore (in-memory fake)
    - queue handler persists records for a chunk
"""
import asyncio
from contextlib import asynccontextmanager

import pytest

from src.jobs.batch import Batch, BatchState, BatchChunk, ChunkState
from src.jobs.executor import make_queue_handler
from src.storage.batch_store import BatchStore


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# BatchChunk model
# ===========================================================================

def test_chunk_records_defaults_to_empty_list():
    c = BatchChunk()
    assert c.records == []
    assert c.records_count == 0


def test_chunk_mark_succeeded_stores_records():
    c = BatchChunk(state=ChunkState.QUEUED)
    c.mark_running()
    c.mark_succeeded(
        records_count=2,
        records=[{"title": "A"}, {"title": "B"}],
    )
    assert c.state == ChunkState.SUCCEEDED
    assert c.records_count == 2
    assert c.records == [{"title": "A"}, {"title": "B"}]


def test_chunk_mark_succeeded_with_only_count_leaves_records_empty():
    c = BatchChunk(state=ChunkState.QUEUED)
    c.mark_running()
    c.mark_succeeded(records_count=5)
    assert c.records_count == 5
    assert c.records == []


def test_chunk_mark_succeeded_with_none_preserves_existing_records():
    """
    Passing records=None must not blow away records that were already
    set (idempotent replay protection).
    """
    c = BatchChunk(state=ChunkState.QUEUED)
    c.mark_running()
    c.mark_succeeded(records_count=1, records=[{"title": "A"}])

    # Simulate a second call where the caller only has the count.
    # Note: mark_succeeded raises if state != RUNNING, so we reset for
    # the test to exercise the None-preservation branch.
    c.state = ChunkState.RUNNING
    c.mark_succeeded(records_count=1, records=None)
    assert c.records == [{"title": "A"}]


def test_chunk_to_dict_includes_records():
    c = BatchChunk(state=ChunkState.QUEUED)
    c.mark_running()
    c.mark_succeeded(records_count=2, records=[{"a": 1}, {"a": 2}])
    d = c.to_dict()
    assert "records" in d
    assert d["records"] == [{"a": 1}, {"a": 2}]


def test_chunk_round_trip_preserves_records():
    c = BatchChunk(state=ChunkState.QUEUED)
    c.mark_running()
    c.mark_succeeded(
        records_count=1,
        records=[{"title": "X", "price": "$9"}],
    )
    c2 = BatchChunk.from_dict(c.to_dict())
    assert c2.records == [{"title": "X", "price": "$9"}]
    assert c2.records_count == 1


def test_chunk_from_dict_backward_compat_missing_records():
    """Old rows saved before this change won't have `records` key."""
    legacy = {
        "chunk_id": "c-1",
        "batch_id": "b-1",
        "index": 0,
        "urls": ["https://x/a"],
        "state": "succeeded",
        "attempts": 1,
        "max_attempts": 3,
        "records_count": 4,
        "error": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:01+00:00",
    }
    c = BatchChunk.from_dict(legacy)
    assert c.records == []
    assert c.records_count == 4


# ===========================================================================
# End-to-end through BatchStore
# ===========================================================================

@asynccontextmanager
async def _in_memory_store():
    """
    Real BatchStore wired to in-memory fakes that mirror the Supabase
    JSONB shape: each row carries a `data` column with the serialized
    chunk/batch, and reads pull `data` back out.
    """
    batches: dict[tuple[str, str], dict] = {}
    chunks: dict[str, dict] = {}

    def fetch_batch(client_id: str, batch_id: str):
        return batches.get((client_id, batch_id))

    def fetch_chunks(batch_id: str):
        return [
            row for row in chunks.values()
            if row["batch_id"] == batch_id
        ]

    def write_batch(payload: dict) -> None:
        batches[(payload["client_id"], payload["batch_id"])] = {
            "data": dict(payload),
        }

    def write_chunk(batch_id: str, payload: dict) -> None:
        chunks[payload["chunk_id"]] = {
            "batch_id": batch_id,
            "data": dict(payload),
        }

    def delete(client_id: str, batch_id: str) -> None:
        batches.pop((client_id, batch_id), None)
        for cid in [k for k, v in chunks.items()
                    if v["batch_id"] == batch_id]:
            chunks.pop(cid, None)

    yield BatchStore(
        fetch_batch_fn=fetch_batch,
        fetch_chunks_fn=fetch_chunks,
        write_batch_fn=write_batch,
        write_chunk_fn=write_chunk,
        delete_fn=delete,
    )


def test_batch_store_persists_and_recovers_records():
    async def scenario():
        async with _in_memory_store() as store:
            b = Batch.create(
                task_id="t-1",
                task_version=1,
                client_id="acme",
                urls=["https://x/a", "https://x/b"],
                chunk_size=1,
            )
            # Simulate: chunk 0 completed with 3 records.
            c0 = b.start_next_chunk()
            c0.mark_succeeded(
                records_count=3,
                records=[
                    {"title": "A", "price": "$1"},
                    {"title": "B", "price": "$2"},
                    {"title": "C", "price": "$3"},
                ],
            )
            store.save(b)

            loaded = store.get("acme", b.batch_id)
            assert loaded is not None
            assert len(loaded.chunks) == 2
            assert loaded.chunks[0].records_count == 3
            assert loaded.chunks[0].records == [
                {"title": "A", "price": "$1"},
                {"title": "B", "price": "$2"},
                {"title": "C", "price": "$3"},
            ]
            # Second chunk still queued, records empty.
            assert loaded.chunks[1].records == []

    _run(scenario())


# ===========================================================================
# Queue handler
# ===========================================================================

class _StubExecutor:
    """Executor that reports a fixed record list via run_and_return_full."""

    def __init__(self, records):
        self._records = records
        self.full_calls = 0
        self.execute_calls = 0

    async def execute(self, payload):
        self.execute_calls += 1
        return {
            "task_id": payload.get("task_id", ""),
            "client_id": payload.get("client_id", ""),
            "records_count": len(self._records),
            "quality_passed": True,
            "quality_score": 1.0,
            "confidence_mean": 0.9,
            "warnings": [],
        }

    async def run_and_return_full(self, payload):
        self.full_calls += 1
        from src.pipeline_runner import PipelineResult
        return PipelineResult(
            task_id=payload.get("task_id", ""),
            records=list(self._records),
            quality_passed=True,
            quality_score=1.0,
            publication_decision=None,
            change_set_summary={"new": 0, "modified": 0, "removed": 0,
                                 "unchanged": 0, "total": 0},
            workbook=None,
            receipt_signature=None,
        )


class _StubRegistry:
    """Registry that just stashes job records."""

    def __init__(self):
        self.jobs: dict[tuple[str, str], dict] = {}

    def save_job(self, client_id, job_id, job):
        self.jobs[(client_id, job_id)] = job

    def get_job(self, client_id, job_id):
        return self.jobs.get((client_id, job_id))


def test_queue_handler_persists_chunk_records():
    async def scenario():
        async with _in_memory_store() as store:
            b = Batch.create(
                task_id="t-1",
                task_version=1,
                client_id="acme",
                urls=["https://x/a", "https://x/b"],
                chunk_size=1,
            )
            store.save(b)

            records = [{"title": "A"}, {"title": "B"}]
            executor = _StubExecutor(records)
            registry = _StubRegistry()

            handler = make_queue_handler(
                executor, registry, batch_store=store,
            )

            payload = {
                "task_id": "t-1",
                "client_id": "acme",
                "batch_id": b.batch_id,
                "chunk_id": b.chunks[0].chunk_id,
                "urls": list(b.chunks[0].urls),
            }

            result = await handler(payload)

            # The handler used the full-records path for batch chunks.
            assert executor.full_calls == 1
            assert executor.execute_calls == 0
            assert result["records_count"] == 2

            # Records are now durable.
            loaded = store.get("acme", b.batch_id)
            assert loaded.chunks[0].state == ChunkState.SUCCEEDED
            assert loaded.chunks[0].records == records

    _run(scenario())


def test_queue_handler_single_job_still_uses_execute():
    """Non-batch jobs must not use the heavy full-records path."""
    async def scenario():
        executor = _StubExecutor([{"x": 1}])
        registry = _StubRegistry()

        handler = make_queue_handler(executor, registry)

        await handler({
            "task_id": "t-1",
            "client_id": "acme",
            "job_id": "j-1",
        })

        assert executor.execute_calls == 1
        assert executor.full_calls == 0

    _run(scenario())