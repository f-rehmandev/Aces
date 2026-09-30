import pytest

from src.jobs.batch import (
    Batch,
    BatchState,
    ChunkState,
    chunk_urls,
)


def test_chunk_urls_splits_deterministically():
    urls = [f"https://example.com/{i}" for i in range(5)]

    assert chunk_urls(urls, chunk_size=2) == [
        [
            "https://example.com/0",
            "https://example.com/1",
        ],
        [
            "https://example.com/2",
            "https://example.com/3",
        ],
        [
            "https://example.com/4",
        ],
    ]


def test_batch_create_deduplicates_and_preserves_order():
    batch = Batch.create(
        task_id="task-1",
        task_version=3,
        client_id="acme",
        urls=[
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/a",
            "https://example.com/c",
        ],
        chunk_size=2,
    )

    assert batch.task_id == "task-1"
    assert batch.task_version == 3
    assert batch.total_urls == 4
    assert batch.unique_urls == 3
    assert batch.input_manifest == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]

    assert batch.total_chunks == 2
    assert batch.chunks[0].urls == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert batch.chunks[1].urls == [
        "https://example.com/c",
    ]


def test_chunk_retry_until_exhausted():
    batch = Batch.create(
        task_id="task-1",
        task_version=1,
        client_id="acme",
        urls=["https://example.com/a"],
        chunk_size=1,
        max_attempts=2,
    )

    chunk = batch.start_next_chunk()

    assert chunk is not None
    assert chunk.state == ChunkState.RUNNING
    assert chunk.attempts == 1

    batch.fail_chunk(chunk.chunk_id, error="temporary")

    assert chunk.state == ChunkState.QUEUED
    assert batch.retry_count == 1

    chunk2 = batch.start_next_chunk()

    assert chunk2 is chunk
    assert chunk2.attempts == 2

    batch.fail_chunk(chunk2.chunk_id, error="permanent")

    assert chunk2.state == ChunkState.FAILED
    assert batch.state == BatchState.FAILED


def test_batch_complete():
    batch = Batch.create(
        task_id="task-1",
        task_version=2,
        client_id="acme",
        urls=["a", "b", "c"],
        chunk_size=2,
    )

    first = batch.start_next_chunk()
    assert first is not None

    batch.complete_chunk(first.chunk_id, records_count=5)

    assert batch.state == BatchState.RUNNING
    assert batch.completed_count == 1
    assert batch.progress_fraction == 0.5

    second = batch.start_next_chunk()
    assert second is not None

    batch.complete_chunk(second.chunk_id, records_count=2)

    assert batch.state == BatchState.COMPLETED
    assert batch.completed_count == 2
    assert batch.progress_fraction == 1.0


def test_pause_and_resume():
    batch = Batch.create(
        task_id="task-1",
        task_version=1,
        client_id="acme",
        urls=["a", "b", "c"],
        chunk_size=1,
    )

    first = batch.start_next_chunk()
    assert first is not None
    assert first.state == ChunkState.RUNNING

    batch.pause()

    assert batch.state == BatchState.PAUSED
    assert first.state == ChunkState.PAUSED
    assert batch.paused_count == 1

    assert batch.start_next_chunk() is None

    batch.resume()

    assert batch.state == BatchState.QUEUED
    assert first.state == ChunkState.QUEUED

    resumed = batch.start_next_chunk()

    assert resumed is first
    assert resumed.state == ChunkState.RUNNING


def test_cancel_prevents_future_work():
    batch = Batch.create(
        task_id="task-1",
        task_version=1,
        client_id="acme",
        urls=["a", "b"],
        chunk_size=1,
    )

    first = batch.start_next_chunk()
    assert first is not None

    batch.cancel()

    assert batch.state == BatchState.CANCELLED
    assert first.state == ChunkState.CANCELLED
    assert batch.start_next_chunk() is None


def test_batch_round_trip_preserves_version_and_chunks():
    original = Batch.create(
        task_id="task-42",
        task_version=7,
        client_id="acme",
        urls=["a", "b", "c"],
        chunk_size=2,
    )

    data = original.to_dict()
    restored = Batch.from_dict(data)

    assert restored.batch_id == original.batch_id
    assert restored.task_id == "task-42"
    assert restored.task_version == 7
    assert restored.client_id == "acme"
    assert restored.input_manifest == original.input_manifest
    assert [c.urls for c in restored.chunks] == [
        ["a", "b"],
        ["c"],
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_id": "", "task_version": 1, "client_id": "acme", "urls": ["a"]},
        {"task_id": "t", "task_version": 0, "client_id": "acme", "urls": ["a"]},
        {"task_id": "t", "task_version": 1, "client_id": "", "urls": ["a"]},
        {
            "task_id": "t",
            "task_version": 1,
            "client_id": "acme",
            "urls": ["a"],
            "chunk_size": 0,
        },
    ],
)
def test_batch_create_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Batch.create(**kwargs)