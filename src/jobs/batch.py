"""
Batch processing primitives — ACES spec §14.3C.

A Batch is a first-class object for large URL inputs.

Responsibilities:
    - deduplicate input URLs conservatively;
    - split URLs into configuration-driven chunks;
    - preserve task/version/client identity;
    - track per-chunk lifecycle;
    - support pause/resume/retry/cancel;
    - expose deterministic progress information.

This module intentionally performs no database or queue I/O.
Persistence and queue integration are separate layers.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class BatchState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChunkState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


_TERMINAL_BATCH_STATES = {
    BatchState.COMPLETED,
    BatchState.FAILED,
    BatchState.CANCELLED,
}


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

@dataclass
class BatchChunk:
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    batch_id: str = ""
    index: int = 0

    urls: list[str] = field(default_factory=list)

    state: ChunkState = ChunkState.QUEUED
    attempts: int = 0
    max_attempts: int = 3

    records_count: int = 0
    records: list[dict] = field(default_factory=list)
    error: str = ""

    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)

    def mark_running(self) -> None:
        if self.state not in {ChunkState.QUEUED, ChunkState.FAILED}:
            raise ValueError(
                f"chunk {self.chunk_id!r} cannot start from {self.state.value!r}"
            )

        if self.attempts >= self.max_attempts:
            raise ValueError(
                f"chunk {self.chunk_id!r} exhausted max_attempts"
            )

        self.attempts += 1
        self.state = ChunkState.RUNNING
        self.error = ""
        self.updated_at = _utc_iso()

    def mark_succeeded(
        self,
        records_count: int = 0,
        records: list[dict] | None = None,
    ) -> None:
        """
        Mark this chunk as successfully processed.

        `records_count` is authoritative for the counter. `records` is the
        actual payload, persisted so `BatchStore.get()` can reconstruct the
        merged dataset later. Passing `None` leaves any previously-set
        `records` intact (idempotent replay protection).

        Callers that only care about the count (e.g. backfill of old rows)
        can pass `records_count` alone.
        """
        if self.state != ChunkState.RUNNING:
            raise ValueError(
                f"chunk {self.chunk_id!r} is not running"
            )

        self.records_count = max(0, int(records_count))
        if records is not None:
            self.records = list(records)
        self.state = ChunkState.SUCCEEDED
        self.error = ""
        self.updated_at = _utc_iso()

    def mark_failed(self, error: str) -> None:
        if self.state != ChunkState.RUNNING:
            raise ValueError(
                f"chunk {self.chunk_id!r} is not running"
            )

        self.error = str(error or "chunk failed")
        self.state = (
            ChunkState.QUEUED
            if self.attempts < self.max_attempts
            else ChunkState.FAILED
        )
        self.updated_at = _utc_iso()

    def pause(self) -> None:
        if self.state in {
            ChunkState.SUCCEEDED,
            ChunkState.CANCELLED,
        }:
            return

        self.state = ChunkState.PAUSED
        self.updated_at = _utc_iso()

    def resume(self) -> None:
        if self.state == ChunkState.PAUSED:
            self.state = ChunkState.QUEUED
            self.updated_at = _utc_iso()

    def cancel(self) -> None:
        if self.state != ChunkState.SUCCEEDED:
            self.state = ChunkState.CANCELLED
            self.updated_at = _utc_iso()

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "batch_id": self.batch_id,
            "index": self.index,
            "urls": list(self.urls),
            "state": self.state.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "records_count": self.records_count,
            "records": list(self.records),
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchChunk":
        d = dict(data)
        d["state"] = ChunkState(d.get("state", ChunkState.QUEUED.value))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    task_version: int = 1
    client_id: str = "default"

    input_manifest: list[str] = field(default_factory=list)

    chunk_size: int = 100
    chunks: list[BatchChunk] = field(default_factory=list)

    state: BatchState = BatchState.QUEUED

    total_urls: int = 0
    unique_urls: int = 0

    completed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    paused_count: int = 0
    retry_count: int = 0

    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        task_version: int,
        client_id: str,
        urls: Iterable[str],
        chunk_size: int = 100,
        max_attempts: int = 3,
    ) -> "Batch":
        if not task_id:
            raise ValueError("task_id is required")

        if not client_id:
            raise ValueError("client_id is required")

        if int(task_version) < 1:
            raise ValueError("task_version must be >= 1")

        chunk_size = int(chunk_size)
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        max_attempts = int(max_attempts)
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        raw_urls = [str(url).strip() for url in urls if str(url).strip()]

        # Preserve first occurrence order while removing duplicates.
        unique_urls: list[str] = []
        seen: set[str] = set()

        for url in raw_urls:
            if url in seen:
                continue
            seen.add(url)
            unique_urls.append(url)

        batch = cls(
            task_id=task_id,
            task_version=int(task_version),
            client_id=client_id,
            input_manifest=list(unique_urls),
            chunk_size=chunk_size,
            total_urls=len(raw_urls),
            unique_urls=len(unique_urls),
        )

        batch_id = batch.batch_id

        for index in range(0, len(unique_urls), chunk_size):
            chunk_urls = unique_urls[index:index + chunk_size]
            chunk = BatchChunk(
                batch_id=batch_id,
                index=len(batch.chunks),
                urls=chunk_urls,
                max_attempts=max_attempts,
            )
            batch.chunks.append(chunk)

        batch._refresh_counts()
        return batch

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def pause(self) -> None:
        if self.state in _TERMINAL_BATCH_STATES:
            return

        self.state = BatchState.PAUSED

        # Only an actively running chunk needs its own paused state.
        # Queued chunks remain queued and are prevented from starting
        # while the batch itself is paused.
        for chunk in self.chunks:
            if chunk.state == ChunkState.RUNNING:
                chunk.pause()

        self._refresh_counts()

    def resume(self) -> None:
        if self.state != BatchState.PAUSED:
            return

        self.state = BatchState.QUEUED

        for chunk in self.chunks:
            if chunk.state == ChunkState.PAUSED:
                chunk.resume()

        self._refresh_counts()

    def cancel(self) -> None:
        if self.state in _TERMINAL_BATCH_STATES:
            return

        self.state = BatchState.CANCELLED

        for chunk in self.chunks:
            if chunk.state not in {
                ChunkState.SUCCEEDED,
                ChunkState.CANCELLED,
            }:
                chunk.cancel()

        self._refresh_counts()

    # ------------------------------------------------------------------
    # Chunk lifecycle
    # ------------------------------------------------------------------

    def start_next_chunk(self) -> BatchChunk | None:
        if self.state in {
            BatchState.PAUSED,
            BatchState.CANCELLED,
            BatchState.COMPLETED,
        }:
            return None

        if self.state == BatchState.QUEUED:
            self.state = BatchState.RUNNING

        for chunk in self.chunks:
            if chunk.state == ChunkState.QUEUED:
                chunk.mark_running()
                self._refresh_counts()
                return chunk

        self._refresh_counts()
        return None

    def complete_chunk(
        self,
        chunk_id: str,
        *,
        records_count: int = 0,
    ) -> BatchChunk:
        chunk = self._get_chunk(chunk_id)
        chunk.mark_succeeded(records_count)
        self._refresh_counts()
        return chunk

    def fail_chunk(
        self,
        chunk_id: str,
        *,
        error: str,
    ) -> BatchChunk:
        chunk = self._get_chunk(chunk_id)

        chunk.mark_failed(error)

        if chunk.state == ChunkState.QUEUED:
            # The failed attempt remains retryable.
            self.retry_count += 1

        self._refresh_counts()
        return chunk

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_BATCH_STATES

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def completed_chunks(self) -> int:
        return sum(
            1 for chunk in self.chunks
            if chunk.state == ChunkState.SUCCEEDED
        )

    @property
    def progress_fraction(self) -> float:
        if not self.chunks:
            return 1.0

        return self.completed_chunks / len(self.chunks)

    def _refresh_counts(self) -> None:
        self.completed_count = sum(
            1 for chunk in self.chunks
            if chunk.state == ChunkState.SUCCEEDED
        )

        self.failed_count = sum(
            1 for chunk in self.chunks
            if chunk.state == ChunkState.FAILED
        )

        self.paused_count = sum(
            1 for chunk in self.chunks
            if chunk.state == ChunkState.PAUSED
        )

        self.skipped_count = 0

        if self.state == BatchState.RUNNING:
            if self.chunks and all(
                chunk.state == ChunkState.SUCCEEDED
                for chunk in self.chunks
            ):
                self.state = BatchState.COMPLETED
            elif self.failed_count and not any(
                chunk.state in {
                    ChunkState.QUEUED,
                    ChunkState.RUNNING,
                }
                for chunk in self.chunks
            ):
                self.state = BatchState.FAILED

        self.updated_at = _utc_iso()

    def _get_chunk(self, chunk_id: str) -> BatchChunk:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise KeyError(f"unknown chunk_id: {chunk_id!r}")

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "client_id": self.client_id,
            "input_manifest": list(self.input_manifest),
            "chunk_size": self.chunk_size,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "state": self.state.value,
            "total_urls": self.total_urls,
            "unique_urls": self.unique_urls,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "paused_count": self.paused_count,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Batch":
        d = dict(data)
        d["state"] = BatchState(d.get("state", BatchState.QUEUED.value))
        d["chunks"] = [
            BatchChunk.from_dict(item)
            for item in d.get("chunks", [])
        ]
        known = set(cls.__dataclass_fields__)
        batch = cls(**{k: v for k, v in d.items() if k in known})
        batch._refresh_counts()
        return batch


def chunk_urls(
    urls: Iterable[str],
    *,
    chunk_size: int = 100,
) -> list[list[str]]:
    """
    Pure helper for callers that only need deterministic chunking.
    """
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be >= 1")

    normalized = [str(url).strip() for url in urls if str(url).strip()]

    return [
        normalized[index:index + int(chunk_size)]
        for index in range(0, len(normalized), int(chunk_size))
    ]