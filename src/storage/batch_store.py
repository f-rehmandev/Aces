"""
Persistent Batch store — spec §14.3C.

The in-memory Batch model remains the source of state-machine behavior.
This store handles serialization/persistence and reconstructs a Batch
from one durable batch row plus its durable chunk rows.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.jobs.batch import Batch


logger = logging.getLogger("batch_store")


class BatchStoreError(RuntimeError):
    """Raised when a batch persistence operation fails."""


class BatchStore:
    def __init__(
        self,
        fetch_batch_fn: Callable[[str, str], Optional[dict]],
        fetch_chunks_fn: Callable[[str], list[dict]],
        write_batch_fn: Callable[[dict], None],
        write_chunk_fn: Callable[[str, dict], None],
        delete_fn: Optional[Callable[[str, str], None]] = None,
    ):
        self._fetch_batch = fetch_batch_fn
        self._fetch_chunks = fetch_chunks_fn
        self._write_batch = write_batch_fn
        self._write_chunk = write_chunk_fn
        self._delete = delete_fn

    def save(self, batch: Batch) -> None:
        """Persist the current batch state and every chunk snapshot."""
        try:
            batch_dict = batch.to_dict()
            batch_id = batch.batch_id

            batch_payload = dict(batch_dict)
            batch_payload["chunks"] = []

            self._write_batch(batch_payload)

            for chunk in batch.chunks:
                self._write_chunk(batch_id, chunk.to_dict())

        except Exception as e:
            logger.warning(
                "batch save failed for %r: %s: %s",
                getattr(batch, "batch_id", ""),
                type(e).__name__,
                e,
            )
            raise BatchStoreError("failed to persist batch") from e

    def get(
        self,
        client_id: str,
        batch_id: str,
    ) -> Optional[Batch]:
        """Load one batch with its durable chunks."""
        if not client_id or not batch_id:
            return None

        try:
            raw = self._fetch_batch(client_id, batch_id)

            if not raw:
                return None

            payload = dict(raw.get("data") or raw)

            chunks = self._fetch_chunks(batch_id) or []

            payload["chunks"] = [
                dict(row.get("data") or row)
                for row in chunks
            ]

            return Batch.from_dict(payload)

        except Exception as e:
            logger.warning(
                "batch load failed for %r: %s: %s",
                batch_id,
                type(e).__name__,
                e,
            )
            raise BatchStoreError("failed to load batch") from e

    def delete(
        self,
        client_id: str,
        batch_id: str,
    ) -> None:
        """Delete one batch and its chunks."""
        if self._delete is None:
            return

        try:
            self._delete(client_id, batch_id)

        except Exception as e:
            logger.warning(
                "batch delete failed for %r: %s: %s",
                batch_id,
                type(e).__name__,
                e,
            )
            raise BatchStoreError("failed to delete batch") from e


def build_supabase_store() -> BatchStore:
    """Build a BatchStore backed by `batches` and `batch_chunks`."""

    from src.storage.db import get_client

    def fetch_batch(
        client_id: str,
        batch_id: str,
    ) -> Optional[dict]:
        client = get_client()

        resp = (
            client.table("batches")
            .select("*")
            .eq("client_id", client_id)
            .eq("batch_id", batch_id)
            .limit(1)
            .execute()
        )

        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None

    def fetch_chunks(batch_id: str) -> list[dict]:
        client = get_client()

        resp = (
            client.table("batch_chunks")
            .select("*")
            .eq("batch_id", batch_id)
            .order("chunk_index", desc=False)
            .execute()
        )

        return list(getattr(resp, "data", None) or [])

    def write_batch(batch_dict: dict) -> None:
        client = get_client()

        row = {
            "batch_id": batch_dict["batch_id"],
            "client_id": batch_dict["client_id"],
            "task_id": batch_dict["task_id"],
            "task_version": int(batch_dict["task_version"]),
            "state": batch_dict["state"],
            "total_urls": int(batch_dict["total_urls"]),
            "unique_urls": int(batch_dict["unique_urls"]),
            "completed_count": int(batch_dict["completed_count"]),
            "failed_count": int(batch_dict["failed_count"]),
            "skipped_count": int(batch_dict["skipped_count"]),
            "paused_count": int(batch_dict["paused_count"]),
            "retry_count": int(batch_dict["retry_count"]),
            "chunk_size": int(batch_dict["chunk_size"]),
            "input_manifest": batch_dict.get("input_manifest") or {},
            "created_at": batch_dict.get("created_at"),
            "updated_at": batch_dict.get("updated_at"),
            "data": batch_dict,
        }

        client.table("batches").upsert(
            row,
            on_conflict="batch_id",
        ).execute()

    def write_chunk(
        batch_id: str,
        chunk_dict: dict,
    ) -> None:
        client = get_client()

        row = {
            "chunk_id": chunk_dict["chunk_id"],
            "batch_id": batch_id,
            "chunk_index": int(chunk_dict["index"]),
            "state": chunk_dict["state"],
            "attempts": int(chunk_dict["attempts"]),
            "max_attempts": int(chunk_dict["max_attempts"]),
            "records_count": int(chunk_dict["records_count"]),
            "error": chunk_dict.get("error", ""),
            "created_at": chunk_dict.get("created_at"),
            "started_at": chunk_dict.get("started_at"),
            "completed_at": chunk_dict.get("completed_at"),
            "data": chunk_dict,
        }

        client.table("batch_chunks").upsert(
            row,
            on_conflict="chunk_id",
        ).execute()

    def delete(
        client_id: str,
        batch_id: str,
    ) -> None:
        client = get_client()

        client.table("batches").delete().eq(
            "client_id",
            client_id,
        ).eq(
            "batch_id",
            batch_id,
        ).execute()

    return BatchStore(
        fetch_batch_fn=fetch_batch,
        fetch_chunks_fn=fetch_chunks,
        write_batch_fn=write_batch,
        write_chunk_fn=write_chunk,
        delete_fn=delete,
    )