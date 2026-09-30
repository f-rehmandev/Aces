import asyncio

from src.jobs.batch import BatchState, ChunkState
from src.jobs.batch_coordinator import (
    BatchCoordinator,
    BatchCoordinatorError,
)
from src.queue.backend import InMemoryQueueBackend


class FakeBatchStore:
    def __init__(self):
        self.saved = []

    def save(self, batch):
        self.saved.append(batch)


def run(coro):
    return asyncio.run(coro)


def test_submit_persists_batch_and_enqueues_every_chunk():
    store = FakeBatchStore()
    queue = InMemoryQueueBackend()

    coordinator = BatchCoordinator(store, queue)

    batch = run(
        coordinator.submit(
            client_id="acme",
            task_id="task-42",
            task_version=7,
            urls=[
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ],
            chunk_size=2,
            max_attempts=3,
            input_manifest=["urls.csv"],
        )
    )

    assert batch.state == BatchState.QUEUED
    assert batch.task_id == "task-42"
    assert batch.task_version == 7
    assert batch.total_urls == 3
    assert batch.unique_urls == 3
    assert len(batch.chunks) == 2

    assert len(store.saved) == 1
    assert store.saved[0].batch_id == batch.batch_id

    stats = run(queue.stats())
    assert stats["total"] == 2


def test_chunk_queue_payload_contains_frozen_task_version_and_exact_urls():
    store = FakeBatchStore()
    queue = InMemoryQueueBackend()

    coordinator = BatchCoordinator(store, queue)

    batch = run(
        coordinator.submit(
            client_id="acme",
            task_id="task-99",
            task_version=12,
            urls=[
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ],
            chunk_size=2,
        )
    )

    first_job = run(queue.get(batch.chunks[0].chunk_id))
    second_job = run(queue.get(batch.chunks[1].chunk_id))

    assert first_job is not None
    assert second_job is not None

    assert first_job.payload["batch_id"] == batch.batch_id
    assert first_job.payload["chunk_id"] == batch.chunks[0].chunk_id
    assert first_job.payload["task_id"] == "task-99"
    assert first_job.payload["task_version"] == 12
    assert first_job.payload["client_id"] == "acme"
    assert first_job.payload["urls"] == [
        "https://example.com/a",
        "https://example.com/b",
    ]

    assert second_job.payload["urls"] == [
        "https://example.com/c",
    ]


def test_repeated_enqueue_is_idempotent():
    store = FakeBatchStore()
    queue = InMemoryQueueBackend()

    coordinator = BatchCoordinator(store, queue)

    batch = run(
        coordinator.submit(
            client_id="acme",
            task_id="task-1",
            task_version=2,
            urls=[
                "https://example.com/a",
                "https://example.com/b",
            ],
            chunk_size=1,
        )
    )

    first_ids = run(coordinator.enqueue_batch(batch))
    second_ids = run(coordinator.enqueue_batch(batch))

    assert second_ids == first_ids

    stats = run(queue.stats())
    assert stats["total"] == 2


def test_non_queued_chunks_are_not_reenqueued():
    store = FakeBatchStore()
    queue = InMemoryQueueBackend()

    coordinator = BatchCoordinator(store, queue)

    batch = run(
        coordinator.submit(
            client_id="acme",
            task_id="task-1",
            task_version=1,
            urls=[
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ],
            chunk_size=1,
        )
    )

    batch.start_next_chunk()
    batch.chunks[0].mark_succeeded()

    ids = run(coordinator.enqueue_batch(batch))

    assert len(ids) == 2

    stats = run(queue.stats())
    assert stats["total"] == 3


def test_queue_failure_is_wrapped():
    class ExplodingQueue:
        async def enqueue(self, job):
            raise RuntimeError("queue unavailable")

    store = FakeBatchStore()
    coordinator = BatchCoordinator(
        store,
        ExplodingQueue(),
    )

    try:
        run(
            coordinator.submit(
                client_id="acme",
                task_id="task-1",
                task_version=1,
                urls=["https://example.com/a"],
                chunk_size=1,
            )
        )
        raise AssertionError("expected BatchCoordinatorError")
    except BatchCoordinatorError as e:
        assert "failed to submit batch" in str(e)


def test_successful_chunk_still_has_chunk_state_unchanged_until_worker_starts():
    store = FakeBatchStore()
    queue = InMemoryQueueBackend()

    coordinator = BatchCoordinator(store, queue)

    batch = run(
        coordinator.submit(
            client_id="acme",
            task_id="task-1",
            task_version=1,
            urls=["https://example.com/a"],
            chunk_size=1,
        )
    )

    assert batch.chunks[0].state == ChunkState.QUEUED