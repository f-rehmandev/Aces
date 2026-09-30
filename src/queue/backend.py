"""
Queue backend — spec §37.

Defines the `QueueBackend` protocol and provides an in-memory
implementation. A Supabase-backed implementation lives in
`supabase_backend.py` and satisfies the same protocol.

Semantics (all methods are async):

    enqueue(job)                      -> str (job_id)
        Insert a new job. If `job.idempotency_key` matches an existing
        non-terminal job, return THAT job's id instead of creating a
        duplicate.

    lease_next(worker_id, lease_seconds) -> QueuedJob | None
        Atomically claim the highest-priority QUEUED job (or an
        expired-lease job that was recovered back to QUEUED).
        Returns None when nothing is ready.
        Ordering: priority ASC, then created_at ASC.

    heartbeat(job_id, worker_id, lease_seconds) -> bool
        Extend the lease on a job the caller currently holds.
        Returns False if the caller no longer holds the lease.

    complete(job_id, result)          -> bool
        Mark a job COMPLETED. Idempotent for terminal states.

    fail(job_id, error, retryable)    -> JobState
        Mark a job FAILED. If retryable and attempts < max_attempts,
        the job returns to QUEUED. Otherwise it moves to DEAD_LETTER.
        Returns the resulting state.

    recover_expired_leases()          -> int
        Sweep the queue for LEASED/RUNNING jobs whose leases have
        expired (worker crash) and reset them to QUEUED with an
        incremented attempt counter. Returns the number recovered.
        Any job exceeding max_attempts while recovering goes to
        DEAD_LETTER.

    get(job_id)                       -> QueuedJob | None
    stats()                           -> dict
    dead_letters(limit)               -> list[QueuedJob]
    requeue_dead_letter(job_id)       -> bool
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from src.queue.types import (
    JobState,
    Priority,
    QueuedJob,
    is_terminal,
)


logger = logging.getLogger("queue.backend")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _future_iso(seconds: int) -> str:
    return (_utc_now() + timedelta(seconds=seconds)).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class QueueBackend(Protocol):
    async def enqueue(self, job: QueuedJob) -> str: ...
    async def lease_next(
        self, worker_id: str, lease_seconds: int,
    ) -> Optional[QueuedJob]: ...
    async def heartbeat(
        self, job_id: str, worker_id: str, lease_seconds: int,
    ) -> bool: ...
    async def complete(
        self, job_id: str, result: Optional[dict] = None,
    ) -> bool: ...
    async def fail(
        self, job_id: str, error: str, retryable: bool = True,
    ) -> JobState: ...
    async def recover_expired_leases(self) -> int: ...
    async def get(self, job_id: str) -> Optional[QueuedJob]: ...
    async def stats(self) -> dict: ...
    async def dead_letters(self, limit: int = 100) -> list[QueuedJob]: ...
    async def requeue_dead_letter(self, job_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------

class InMemoryQueueBackend:
    """
    Single-process queue. Suitable for tests and for local dev.

    Not durable across process restarts — use the Supabase-backed
    implementation for that. Concurrency is via asyncio.Lock, which
    is correct because Python asyncio is single-threaded.
    """

    def __init__(self):
        self._jobs: dict[str, QueuedJob] = {}
        self._by_idempotency: dict[str, str] = {}   # key -> job_id
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    async def enqueue(self, job: QueuedJob) -> str:
        async with self._lock:
            # Idempotency: only dedupe against non-terminal jobs
            if job.idempotency_key:
                existing_id = self._by_idempotency.get(job.idempotency_key)
                if existing_id:
                    existing = self._jobs.get(existing_id)
                    if existing and not is_terminal(existing.state):
                        logger.debug(
                            f"enqueue: idempotency hit for "
                            f"{job.idempotency_key!r} -> {existing_id}"
                        )
                        return existing_id

            job.state = JobState.QUEUED
            job.updated_at = _utc_now().isoformat(timespec="milliseconds")
            self._jobs[job.job_id] = job
            if job.idempotency_key:
                self._by_idempotency[job.idempotency_key] = job.job_id
            return job.job_id

    async def lease_next(
        self, worker_id: str, lease_seconds: int,
    ) -> Optional[QueuedJob]:
        async with self._lock:
            # First: promote any expired leases back to QUEUED
            self._recover_expired_locked()

            candidates = [
                j for j in self._jobs.values()
                if j.state == JobState.QUEUED
            ]
            if not candidates:
                return None

            # Priority ASC (P0 first), then created_at ASC (FIFO)
            candidates.sort(key=lambda j: (int(j.priority), j.created_at))
            job = candidates[0]

            job.state = JobState.LEASED
            job.leased_by = worker_id
            job.leased_at = _utc_now().isoformat(timespec="milliseconds")
            job.lease_expires_at = _future_iso(lease_seconds)
            job.updated_at = job.leased_at
            return job

    async def heartbeat(
        self, job_id: str, worker_id: str, lease_seconds: int,
    ) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.leased_by != worker_id:
                return False
            if job.state not in (JobState.LEASED, JobState.RUNNING):
                return False
            job.lease_expires_at = _future_iso(lease_seconds)
            job.updated_at = _utc_now().isoformat(timespec="milliseconds")
            return True

    async def complete(
        self, job_id: str, result: Optional[dict] = None,
    ) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.state == JobState.COMPLETED:
                return True   # idempotent
            if is_terminal(job.state):
                return False  # can't complete a dead-lettered job
            job.state = JobState.COMPLETED
            job.leased_by = ""
            job.lease_expires_at = ""
            job.updated_at = _utc_now().isoformat(timespec="milliseconds")
            if result is not None:
                # Don't mutate the payload in place — a handler or
                # caller may still hold a reference to the original
                # dict. Replace it with a fresh copy that adds `_result`.
                new_payload = dict(job.payload) if isinstance(job.payload, dict) else {}
                new_payload["_result"] = result
                job.payload = new_payload
            return True

    async def fail(
        self, job_id: str, error: str, retryable: bool = True,
    ) -> JobState:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return JobState.DEAD_LETTER
            if is_terminal(job.state):
                return job.state

            job.last_error = error
            job.attempts += 1
            job.leased_by = ""
            job.lease_expires_at = ""

            if retryable and job.attempts < job.max_attempts:
                job.state = JobState.FAILED
                # Immediately re-queue for the next lease pass.
                # A production system might delay this; the spec says
                # retryable failures re-queue.
                job.state = JobState.QUEUED
            else:
                job.state = JobState.DEAD_LETTER

            job.updated_at = _utc_now().isoformat(timespec="milliseconds")
            return job.state

    async def recover_expired_leases(self) -> int:
        async with self._lock:
            return self._recover_expired_locked()

    async def requeue_dead_letter(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != JobState.DEAD_LETTER:
                return False
            job.state = JobState.QUEUED
            job.attempts = 0
            job.last_error = ""
            job.updated_at = _utc_now().isoformat(timespec="milliseconds")
            return True

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    async def get(self, job_id: str) -> Optional[QueuedJob]:
        return self._jobs.get(job_id)

    async def stats(self) -> dict:
        counts: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        for j in self._jobs.values():
            counts[j.state.value] = counts.get(j.state.value, 0) + 1
            by_priority[j.priority.name] = by_priority.get(j.priority.name, 0) + 1
        return {
            "total": len(self._jobs),
            "by_state": counts,
            "by_priority": by_priority,
        }

    async def dead_letters(self, limit: int = 100) -> list[QueuedJob]:
        out = [
            j for j in self._jobs.values()
            if j.state == JobState.DEAD_LETTER
        ]
        out.sort(key=lambda j: j.updated_at, reverse=True)
        return out[:limit]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _recover_expired_locked(self) -> int:
        now = _utc_now()
        recovered = 0
        for job in self._jobs.values():
            if not job.is_expired(now):
                continue
            job.attempts += 1
            job.leased_by = ""
            job.leased_at = ""
            job.lease_expires_at = ""
            if job.attempts >= job.max_attempts:
                job.state = JobState.DEAD_LETTER
                job.last_error = "lease expired before completion"
            else:
                job.state = JobState.QUEUED
                job.last_error = "lease expired; re-queued"
            job.updated_at = _utc_now().isoformat(timespec="milliseconds")
            recovered += 1
        if recovered:
            logger.info(f"recovered {recovered} expired lease(s)")
        return recovered


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def run():
        q = InMemoryQueueBackend()

        # Empty
        assert (await q.lease_next("w-1", 60)) is None
        s = await q.stats()
        assert s["total"] == 0

        # Enqueue + lease
        j1 = QueuedJob(label="first", priority=Priority.P2_SCHEDULED)
        jid1 = await q.enqueue(j1)
        assert jid1 == j1.job_id

        leased = await q.lease_next("w-1", 60)
        assert leased is not None
        assert leased.job_id == jid1
        assert leased.state == JobState.LEASED
        assert leased.leased_by == "w-1"
        assert leased.lease_expires_at

        # No more to lease
        assert (await q.lease_next("w-2", 60)) is None

        # Heartbeat from the holder
        assert await q.heartbeat(jid1, "w-1", 120)

        # Heartbeat from a different worker (rejected)
        assert not await q.heartbeat(jid1, "w-2", 120)

        # Complete
        assert await q.complete(jid1, result={"records": 42})
        got = await q.get(jid1)
        assert got.state == JobState.COMPLETED
        assert got.payload["_result"] == {"records": 42}

        # Cannot lease a completed job
        assert (await q.lease_next("w-1", 60)) is None

        # --- Idempotency ---
        j2a = QueuedJob(idempotency_key="key-A", label="dup")
        j2b = QueuedJob(idempotency_key="key-A", label="dup")
        id_a = await q.enqueue(j2a)
        id_b = await q.enqueue(j2b)
        assert id_a == id_b, "same key must collapse to one job"

        # --- Priority ordering ---
        q2 = InMemoryQueueBackend()
        bulk = QueuedJob(label="bulk", priority=Priority.P4_BULK)
        interactive = QueuedJob(label="interactive", priority=Priority.P0_INTERACTIVE)
        client = QueuedJob(label="client", priority=Priority.P1_CLIENT)
        await q2.enqueue(bulk)
        await q2.enqueue(client)
        await q2.enqueue(interactive)

        first = await q2.lease_next("w-1", 60)
        assert first.label == "interactive", first.label
        second = await q2.lease_next("w-1", 60)
        assert second.label == "client"
        third = await q2.lease_next("w-1", 60)
        assert third.label == "bulk"

        # --- Retryable failure re-queues ---
        q3 = InMemoryQueueBackend()
        jr = QueuedJob(max_attempts=3, label="retry-me")
        await q3.enqueue(jr)
        await q3.lease_next("w-1", 60)
        state = await q3.fail(jr.job_id, "transient error", retryable=True)
        assert state == JobState.QUEUED
        got = await q3.get(jr.job_id)
        assert got.attempts == 1
        assert got.last_error == "transient error"

        # --- Exhausted retryable -> dead letter ---
        for _ in range(5):
            await q3.lease_next("w-1", 60)
            await q3.fail(jr.job_id, "still broken", retryable=True)
        got = await q3.get(jr.job_id)
        assert got.state == JobState.DEAD_LETTER, got.state
        assert got.attempts >= 3

        # --- Non-retryable goes straight to dead-letter ---
        q4 = InMemoryQueueBackend()
        jn = QueuedJob(max_attempts=5, label="non-retry")
        await q4.enqueue(jn)
        await q4.lease_next("w-1", 60)
        state = await q4.fail(jn.job_id, "permanent", retryable=False)
        assert state == JobState.DEAD_LETTER

        # --- Dead letter listing + requeue ---
        letters = await q4.dead_letters()
        assert len(letters) == 1
        assert await q4.requeue_dead_letter(letters[0].job_id)
        got = await q4.get(letters[0].job_id)
        assert got.state == JobState.QUEUED
        assert got.attempts == 0

        # --- Lease expiry recovery ---
        q5 = InMemoryQueueBackend()
        jl = QueuedJob(max_attempts=3, label="expiring")
        await q5.enqueue(jl)
        leased = await q5.lease_next("w-1", lease_seconds=0)
        # lease_seconds=0 means it expires immediately
        assert leased.state == JobState.LEASED

        recovered = await q5.recover_expired_leases()
        assert recovered == 1
        got = await q5.get(jl.job_id)
        assert got.state == JobState.QUEUED
        assert got.attempts == 1

        # Exhaust it: recover again after another lease with 0s
        await q5.lease_next("w-1", lease_seconds=0)
        await q5.recover_expired_leases()
        await q5.lease_next("w-1", lease_seconds=0)
        await q5.recover_expired_leases()
        got = await q5.get(jl.job_id)
        assert got.state == JobState.DEAD_LETTER, got.state

        # --- Stats shape ---
        s = await q5.stats()
        assert "total" in s
        assert "by_state" in s
        assert "by_priority" in s
        assert s["by_state"].get("dead_letter", 0) >= 1

        print("InMemoryQueueBackend OK.")

    asyncio.run(run())