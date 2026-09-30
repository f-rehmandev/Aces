from src.jobs.batch import Batch, BatchState, ChunkState
from src.storage.batch_store import BatchStore, BatchStoreError


def make_batch():
    batch = Batch.create(
        client_id="acme",
        task_id="task-1",
        task_version=3,
        urls=[
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/a",
        ],
        chunk_size=1,
        max_attempts=2,
    )
    batch.input_manifest = ["test.csv"]
    return batch


def test_save_and_load_round_trip():
    batches = {}
    chunks = {}

    def fetch_batch(client_id, batch_id):
        row = batches.get((client_id, batch_id))
        return row

    def fetch_chunks(batch_id):
        return [
            row
            for row in chunks.values()
            if row["batch_id"] == batch_id
        ]

    def write_batch(payload):
        batches[(payload["client_id"], payload["batch_id"])] = {
            "data": payload
        }

    def write_chunk(batch_id, payload):
        chunks[payload["chunk_id"]] = {
            "batch_id": batch_id,
            "data": payload,
        }

    store = BatchStore(
        fetch_batch_fn=fetch_batch,
        fetch_chunks_fn=fetch_chunks,
        write_batch_fn=write_batch,
        write_chunk_fn=write_chunk,
    )

    batch = make_batch()
    store.save(batch)

    loaded = store.get("acme", batch.batch_id)

    assert loaded is not None
    assert loaded.batch_id == batch.batch_id
    assert loaded.task_id == "task-1"
    assert loaded.task_version == 3
    assert loaded.input_manifest == ["test.csv"]
    assert len(loaded.chunks) == 2
    assert loaded.chunks[0].state == ChunkState.QUEUED


def test_save_preserves_state_and_chunk_changes():
    batches = {}
    chunks = {}

    def fetch_batch(client_id, batch_id):
        return batches.get((client_id, batch_id))

    def fetch_chunks(batch_id):
        return [
            row
            for row in chunks.values()
            if row["batch_id"] == batch_id
        ]

    def write_batch(payload):
        batches[(payload["client_id"], payload["batch_id"])] = {
            "data": payload
        }

    def write_chunk(batch_id, payload):
        chunks[payload["chunk_id"]] = {
            "batch_id": batch_id,
            "data": payload,
        }

    store = BatchStore(
        fetch_batch_fn=fetch_batch,
        fetch_chunks_fn=fetch_chunks,
        write_batch_fn=write_batch,
        write_chunk_fn=write_chunk,
    )

    batch = make_batch()
    batch.start_next_chunk()
    batch.pause()

    store.save(batch)

    loaded = store.get("acme", batch.batch_id)

    assert loaded is not None
    assert loaded.state == BatchState.PAUSED
    assert loaded.paused_count == 1
    assert loaded.chunks[0].state == ChunkState.PAUSED


def test_client_isolation():
    batches = {}

    def fetch_batch(client_id, batch_id):
        return batches.get((client_id, batch_id))

    def fetch_chunks(batch_id):
        return []

    def write_batch(payload):
        batches[(payload["client_id"], payload["batch_id"])] = {
            "data": payload
        }

    store = BatchStore(
        fetch_batch_fn=fetch_batch,
        fetch_chunks_fn=fetch_chunks,
        write_batch_fn=write_batch,
        write_chunk_fn=lambda batch_id, payload: None,
    )

    batch = make_batch()
    store.save(batch)

    assert store.get("other-client", batch.batch_id) is None
    assert store.get("acme", batch.batch_id) is not None


def test_missing_batch_returns_none():
    store = BatchStore(
        fetch_batch_fn=lambda client_id, batch_id: None,
        fetch_chunks_fn=lambda batch_id: [],
        write_batch_fn=lambda payload: None,
        write_chunk_fn=lambda batch_id, payload: None,
    )

    assert store.get("acme", "missing") is None


def test_save_failure_is_wrapped():
    batch = make_batch()

    store = BatchStore(
        fetch_batch_fn=lambda client_id, batch_id: None,
        fetch_chunks_fn=lambda batch_id: [],
        write_batch_fn=lambda payload: (_ for _ in ()).throw(
            RuntimeError("db down")
        ),
        write_chunk_fn=lambda batch_id, payload: None,
    )

    try:
        store.save(batch)
        raise AssertionError("expected BatchStoreError")
    except BatchStoreError:
        pass


def test_load_failure_is_wrapped():
    store = BatchStore(
        fetch_batch_fn=lambda client_id, batch_id: (_ for _ in ()).throw(
            RuntimeError("db down")
        ),
        fetch_chunks_fn=lambda batch_id: [],
        write_batch_fn=lambda payload: None,
        write_chunk_fn=lambda batch_id, payload: None,
    )

    try:
        store.get("acme", "batch-1")
        raise AssertionError("expected BatchStoreError")
    except BatchStoreError:
        pass


def test_delete():
    batches = {}

    def fetch_batch(client_id, batch_id):
        return batches.get((client_id, batch_id))

    def write_batch(payload):
        batches[(payload["client_id"], payload["batch_id"])] = {
            "data": payload
        }

    def delete_batch(client_id, batch_id):
        batches.pop((client_id, batch_id), None)

    store = BatchStore(
        fetch_batch_fn=fetch_batch,
        fetch_chunks_fn=lambda batch_id: [],
        write_batch_fn=write_batch,
        write_chunk_fn=lambda batch_id, payload: None,
        delete_fn=delete_batch,
    )

    batch = make_batch()
    store.save(batch)

    assert store.get("acme", batch.batch_id) is not None

    store.delete("acme", batch.batch_id)

    assert store.get("acme", batch.batch_id) is None