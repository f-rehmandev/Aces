from src.jobs.batch import Batch, BatchState, ChunkState
from src.storage.batch_store import build_supabase_store


CLIENT_ID = "__aces_batch_smoke__"


def main():
    store = build_supabase_store()

    batch = Batch.create(
        client_id=CLIENT_ID,
        task_id="__batch_smoke_task__",
        task_version=7,
        urls=[
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/a",
        ],
        chunk_size=1,
        max_attempts=2,
    )

    batch.input_manifest = ["smoke.csv"]

    print(f"Created batch: {batch.batch_id}")

    try:
        # 1. Save initial batch.
        store.save(batch)

        loaded = store.get(CLIENT_ID, batch.batch_id)

        assert loaded is not None
        assert loaded.batch_id == batch.batch_id
        assert loaded.task_id == "__batch_smoke_task__"
        assert loaded.task_version == 7
        assert loaded.total_urls == 3
        assert loaded.unique_urls == 2
        assert len(loaded.chunks) == 2

        print("Initial save/load: PASS")

        # 2. Persist a running chunk.
        chunk = loaded.start_next_chunk()

        assert chunk is not None
        assert chunk.state == ChunkState.RUNNING

        store.save(loaded)

        loaded2 = store.get(CLIENT_ID, batch.batch_id)

        assert loaded2 is not None
        assert loaded2.state == BatchState.RUNNING
        assert loaded2.chunks[0].state == ChunkState.RUNNING

        print("Chunk state persistence: PASS")

        # 3. Persist a successful chunk.
        loaded2.chunks[0].mark_succeeded()
        store.save(loaded2)

        loaded3 = store.get(CLIENT_ID, batch.batch_id)

        assert loaded3 is not None
        assert loaded3.chunks[0].state == ChunkState.SUCCEEDED
        assert loaded3.completed_count == 1

        print("Completed chunk persistence: PASS")

        # 4. Verify tenant isolation.
        assert store.get(
            "__different_client__",
            batch.batch_id,
        ) is None

        print("Client isolation: PASS")

        print("LIVE BATCH SMOKE: PASS")

    finally:
        # Remove the smoke-test batch. FK cascade removes its chunks.
        store.delete(CLIENT_ID, batch.batch_id)

        assert store.get(CLIENT_ID, batch.batch_id) is None

        print("Cleanup: PASS")


if __name__ == "__main__":
    main()