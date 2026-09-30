"""
Batch submission coordinator — spec §14.3C.

Responsibilities:
    1. Create/persist a first-class Batch.
    2. Submit one durable queue job per chunk.
    3. Preserve the frozen task version on every chunk job.
    4. Preserve the exact chunk URL list on every queue payload.
    5. Use queue idempotency so retrying submission does not duplicate
       a chunk job.

This module intentionally does NOT execute the chunk. Chunk-aware
execution is a separate integration step.
"""

from __future__ import annotations

from src.jobs.batch import Batch, BatchChunk, ChunkState
from src.queue.types import Priority, QueuedJob


class BatchCoordinatorError(RuntimeError):
    """Raised when batch submission cannot be completed."""


class BatchCoordinator:
    def __init__(self, batch_store, queue_backend):
        self.batch_store = batch_store
        self.queue_backend = queue_backend

    async def submit(
        self,
        *,
        client_id: str,
        task_id: str,
        task_version: int,
        urls: list[str],
        chunk_size: int,
        max_attempts: int = 3,
        input_manifest: list[str] | None = None,
    ) -> Batch:
        """
        Create a durable Batch and enqueue every chunk.

        The Batch remains QUEUED until chunk execution begins.
        """
        batch = Batch.create(
            client_id=client_id,
            task_id=task_id,
            task_version=task_version,
            urls=urls,
            chunk_size=chunk_size,
            max_attempts=max_attempts,
        )

        batch.input_manifest = list(input_manifest or [])

        try:
            self.batch_store.save(batch)
            await self.enqueue_batch(batch)
        except Exception as e:
            raise BatchCoordinatorError(
                f"failed to submit batch {batch.batch_id!r}: "
                f"{type(e).__name__}: {e}"
            ) from e

        return batch

    async def enqueue_batch(self, batch: Batch) -> list[str]:
        """
        Enqueue all currently QUEUED chunks.

        Returns the queue job IDs returned by QueueBackend.
        """
        job_ids: list[str] = []

        for chunk in batch.chunks:
            if chunk.state != ChunkState.QUEUED:
                continue

            job = QueuedJob(
                job_id=chunk.chunk_id,
                idempotency_key=(
                    f"batch:{batch.batch_id}:chunk:{chunk.chunk_id}"
                ),
                priority=Priority.P4_BULK,
                payload={
                    "batch_id": batch.batch_id,
                    "chunk_id": chunk.chunk_id,
                    "task_id": batch.task_id,
                    "task_version": batch.task_version,
                    "client_id": batch.client_id,
                    "urls": list(chunk.urls),
                },
                client_id=batch.client_id,
                label=f"batch:{batch.batch_id}:chunk:{chunk.index}",
                max_attempts=chunk.max_attempts,
            )

            try:
                queue_job_id = await self.queue_backend.enqueue(job)
            except Exception as e:
                raise BatchCoordinatorError(
                    f"failed to enqueue chunk {chunk.chunk_id!r}: "
                    f"{type(e).__name__}: {e}"
                ) from e

            job_ids.append(queue_job_id)

        return job_ids