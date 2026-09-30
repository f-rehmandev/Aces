import asyncio

import pytest

from src.api.registry import ServiceRegistry
from src.jobs.batch import Batch
from src.jobs.executor import JobExecutor, make_queue_handler


def _run(coro):
    return asyncio.run(coro)


class FakeBatchStore:
    def __init__(self):
        self.batches = {}
        self.saves = []

    async def save(self, batch):
        self.batches[(batch.client_id, batch.batch_id)] = batch
        self.saves.append(batch)
        return True

    async def get(self, client_id, batch_id):
        return self.batches.get((client_id, batch_id))


class FakeExecutor:
    """
    Minimal executor stub that implements BOTH production entry points:

        execute(payload)              -> lightweight dict (non-batch jobs)
        run_and_return_full(payload)  -> PipelineResult (batch chunks)

    The real JobExecutor exposes both. Task 3 changed `make_queue_handler`
    to call `run_and_return_full` for batch chunks so the actual records
    get persisted into the chunk. Any fake must therefore implement both,
    or the batch path explodes with AttributeError.
    """

    def __init__(self, result=None, error=None):
        self.result = result or {
            "task_id": "task-1",
            "client_id": "acme",
            "records_count": 7,
        }
        self.error = error
        self.calls = []

    async def execute(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return dict(self.result)

    async def run_and_return_full(self, payload):
        """
        Batch-chunk path. The handler reads:
            len(full.records), full.quality_passed, full.quality_score,
            full.confidence_mean, full.warnings

        We build a real PipelineResult so the interface matches
        production exactly. The number of records is derived from
        self.result["records_count"] so the existing test assertions
        on `result["records_count"]` still hold.
        """
        self.calls.append(payload)
        if self.error is not None:
            raise self.error

        from src.pipeline_runner import PipelineResult

        n = int(self.result.get("records_count", 0))
        return PipelineResult(
            task_id=payload.get("task_id", ""),
            records=[{"idx": i} for i in range(n)],
            quality_passed=True,
            quality_score=1.0,
            publication_decision=None,
            change_set_summary={
                "new": 0, "modified": 0, "removed": 0,
                "unchanged": 0, "total": 0,
            },
            workbook=None,
            receipt_signature=None,
            warnings=[],
        )


class FakeRegistry(ServiceRegistry):
    def __init__(self):
        self.jobs = {}

    def get_job(self, client_id, job_id):
        return self.jobs.get((client_id, job_id))


def _make_batch():
    return Batch.create(
        client_id="acme",
        task_id="task-1",
        task_version=1,
        urls=[
            "https://example.com/a",
            "https://example.com/b",
        ],
        chunk_size=2,
        max_attempts=3,
    )


def test_batch_worker_marks_chunk_running_then_succeeded():
    registry = FakeRegistry()
    batch_store = FakeBatchStore()
    batch = _make_batch()

    _run(batch_store.save(batch))

    executor = FakeExecutor()
    handler = make_queue_handler(
        executor,
        registry,
        batch_store=batch_store,
    )

    result = _run(handler({
        "task_id": "task-1",
        "client_id": "acme",
        "batch_id": batch.batch_id,
        "chunk_id": batch.chunks[0].chunk_id,
        "urls": list(batch.chunks[0].urls),
    }))

    loaded = _run(
        batch_store.get(
            "acme",
            batch.batch_id,
        )
    )

    chunk = loaded.chunks[0]

    assert result["records_count"] == 7
    assert chunk.state.value == "succeeded"
    assert chunk.records_count == 7
    assert loaded.completed_count == 1
    assert len(executor.calls) == 1


def test_batch_worker_failure_persists_retryable_chunk_state():
    registry = FakeRegistry()
    batch_store = FakeBatchStore()
    batch = _make_batch()

    _run(batch_store.save(batch))

    executor = FakeExecutor(
        error=RuntimeError("boom"),
    )

    handler = make_queue_handler(
        executor,
        registry,
        batch_store=batch_store,
    )

    with pytest.raises(RuntimeError, match="boom"):
        _run(handler({
            "task_id": "task-1",
            "client_id": "acme",
            "batch_id": batch.batch_id,
            "chunk_id": batch.chunks[0].chunk_id,
            "urls": list(batch.chunks[0].urls),
        }))

    loaded = _run(
        batch_store.get(
            "acme",
            batch.batch_id,
        )
    )

    chunk = loaded.chunks[0]

    assert chunk.state.value == "queued"
    assert chunk.attempts == 1
    assert loaded.retry_count == 1
    assert "RuntimeError: boom" in chunk.error


def test_completed_chunk_is_not_executed_again():
    registry = FakeRegistry()
    batch_store = FakeBatchStore()
    batch = _make_batch()

    batch.start_next_chunk()
    batch.chunks[0].mark_succeeded(
        records_count=3,
    )
    batch._refresh_counts()

    _run(batch_store.save(batch))

    executor = FakeExecutor()
    handler = make_queue_handler(
        executor,
        registry,
        batch_store=batch_store,
    )

    result = _run(handler({
        "task_id": "task-1",
        "client_id": "acme",
        "batch_id": batch.batch_id,
        "chunk_id": batch.chunks[0].chunk_id,
        "urls": list(batch.chunks[0].urls),
    }))

    assert result["skipped"] is True
    assert result["state"] == "succeeded"
    assert executor.calls == []


def test_non_batch_job_still_works():
    registry = FakeRegistry()
    executor = FakeExecutor()

    handler = make_queue_handler(
        executor,
        registry,
    )

    result = _run(handler({
        "task_id": "task-1",
        "client_id": "acme",
    }))

    assert result["records_count"] == 7
    assert len(executor.calls) == 1