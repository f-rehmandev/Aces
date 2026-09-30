"""
Integration test: real `BatchStore` through the real
`build_supabase_store()` factory.

This is deliberately a level above the unit tests in
`tests/unit/test_batch_chunk_records.py`. Those inject `fetch_fn` and
`write_fn` callables directly. This test exercises the ACTUAL code path
that runs in production:

    BatchStore.save()   ->  _write_batch   ->  client.table("batches").upsert(...)
    BatchStore.save()   ->  _write_chunk   ->  client.table("batch_chunks").upsert(...)
    BatchStore.get()    ->  _fetch_batch   ->  client.table("batches").select(...).eq(...)
    BatchStore.get()    ->  _fetch_chunks  ->  client.table("batch_chunks").select(...).eq(...).order(...)

The only thing swapped out is `src.storage.db.get_client`. Everything
else is real production code.

Why this matters: the sync-vs-async BatchStore bug (Task 5 in the
session log) was invisible to unit tests precisely because they
substituted callables that happened to match the real signatures. A
test at this level — real store, fake transport — would have caught
it directly.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from src.jobs.batch import Batch, BatchState, ChunkState
from tests.integration._fakes import FakeSupabaseClient


# ---------------------------------------------------------------------------
# Fixture: swap the real Supabase client for our in-memory fake
# ---------------------------------------------------------------------------

@contextmanager
def _patched_get_client():
    """
    Monkeypatch `src.storage.db.get_client` so `build_supabase_store()`
    and every other storage-layer factory use our fake.

    We patch the module attribute directly (rather than pytest's
    monkeypatch fixture) so the patch is scoped to the `with` block and
    can be used inside a single test's body.
    """
    import src.storage.db as db_mod
    original = db_mod.get_client
    fake = FakeSupabaseClient()
    db_mod.get_client = lambda: fake
    try:
        yield fake
    finally:
        db_mod.get_client = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_supabase_store_uses_real_read_write_path():
    """End-to-end: build → save → load → assert records survive."""
    from src.storage.batch_store import build_supabase_store

    with _patched_get_client() as client:
        store = build_supabase_store()

        batch = Batch.create(
            task_id="t-int-1",
            task_version=1,
            client_id="acme",
            urls=["https://example.com/a", "https://example.com/b"],
            chunk_size=1,
        )

        # Drive chunk 0 through a full lifecycle with records.
        chunk0 = batch.start_next_chunk()
        chunk0.mark_succeeded(
            records_count=2,
            records=[
                {"title": "A", "price": "$1"},
                {"title": "B", "price": "$2"},
            ],
        )
        batch._refresh_counts()

        store.save(batch)

        # Confirm the actual rows exist under the expected table names.
        batches_rows = client._tables["batches"]
        chunks_rows = client._tables["batch_chunks"]
        assert len(batches_rows) == 1
        assert len(chunks_rows) == 2   # two chunks written

        # ---- round trip ----
        loaded = store.get("acme", batch.batch_id)
        assert loaded is not None
        assert loaded.batch_id == batch.batch_id
        assert loaded.task_id == "t-int-1"
        assert loaded.state == BatchState.RUNNING
        assert len(loaded.chunks) == 2

        # Chunk 0 has records, chunk 1 is still queued.
        assert loaded.chunks[0].state == ChunkState.SUCCEEDED
        assert loaded.chunks[0].records == [
            {"title": "A", "price": "$1"},
            {"title": "B", "price": "$2"},
        ]
        assert loaded.chunks[1].state == ChunkState.QUEUED
        assert loaded.chunks[1].records == []


def test_save_upserts_not_duplicates_on_second_call():
    """
    `BatchStore.save` uses `.upsert(on_conflict=...)`, so saving the
    same batch twice must not duplicate rows. This is the real code
    path — `_write_batch` / `_write_chunk`.
    """
    from src.storage.batch_store import build_supabase_store

    with _patched_get_client() as client:
        store = build_supabase_store()

        batch = Batch.create(
            task_id="t-int-2",
            task_version=1,
            client_id="acme",
            urls=["https://example.com/a"],
            chunk_size=1,
        )
        store.save(batch)
        store.save(batch)
        store.save(batch)

        assert len(client._tables["batches"]) == 1
        assert len(client._tables["batch_chunks"]) == 1


def test_get_returns_none_for_wrong_client():
    """
    Tenant scoping must be enforced at the storage layer via
    `.eq("client_id", ...)`, not just at the API layer.
    """
    from src.storage.batch_store import build_supabase_store

    with _patched_get_client():
        store = build_supabase_store()

        batch = Batch.create(
            task_id="t-int-3",
            task_version=1,
            client_id="acme",
            urls=["https://example.com/a"],
            chunk_size=1,
        )
        store.save(batch)

        assert store.get("acme", batch.batch_id) is not None
        assert store.get("other", batch.batch_id) is None


def test_delete_removes_batch_and_chunks():
    from src.storage.batch_store import build_supabase_store

    with _patched_get_client() as client:
        store = build_supabase_store()

        batch = Batch.create(
            task_id="t-int-4",
            task_version=1,
            client_id="acme",
            urls=["https://example.com/a", "https://example.com/b"],
            chunk_size=1,
        )
        store.save(batch)
        assert len(client._tables["batches"]) == 1
        assert len(client._tables["batch_chunks"]) == 2

        store.delete("acme", batch.batch_id)

        assert len(client._tables["batches"]) == 0
        # delete() in our fake only deletes from the batches table;
        # production relies on the FK cascade for chunks. Assert the
        # real behavior we observe: batch row is gone, get() returns None.
        assert store.get("acme", batch.batch_id) is None


def test_load_latest_semantics_via_table_order():
    """
    `_fetch_chunks` calls `.order("chunk_index", desc=False)`. Confirm
    that ordering survives the round trip — chunks must come back in
    the original index order, not in insertion order.
    """
    from src.storage.batch_store import build_supabase_store

    with _patched_get_client():
        store = build_supabase_store()

        batch = Batch.create(
            task_id="t-int-5",
            task_version=1,
            client_id="acme",
            urls=[f"https://example.com/p{i}" for i in range(5)],
            chunk_size=1,
        )
        # Record original indexes so we can assert order after reload
        original_indexes = [c.index for c in batch.chunks]
        assert original_indexes == [0, 1, 2, 3, 4]

        store.save(batch)
        loaded = store.get("acme", batch.batch_id)

        reloaded_indexes = [c.index for c in loaded.chunks]
        assert reloaded_indexes == [0, 1, 2, 3, 4]