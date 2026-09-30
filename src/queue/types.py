"""
Durable queue types — spec §37.

A job submitted to the queue becomes a `QueuedJob` with:
  - a unique job_id
  - an idempotency key (dedupes duplicate submissions)
  - a priority (P0 interactive → P4 bulk backfill)
  - a lease (expires if the worker dies)
  - attempts (for retry/dead-letter policy)
  - a payload (opaque to the queue — the caller decides its shape)

Jobs move through a small state machine:

    QUEUED → LEASED → RUNNING → COMPLETED
                    ├→ FAILED (retryable)
                    │     ├→ QUEUED (retry)
                    │     └→ DEAD_LETTER (exhausted)
                    └→ DEAD_LETTER (non-retryable)

This module defines the data. The queue backend (in-memory + Supabase)
lives in `backend.py`. The orchestrator (lease manager, worker loop,
dead-letter router) lives in `runner.py`.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

class Priority(int, Enum):
    """
    Lower number = higher priority. Values are explicit so the numeric
    ordering matches the spec table (§37.6).

        P0 interactive validation
        P1 client production job
        P2 scheduled monitoring
        P3 healing / experiments
        P4 bulk backfill / low-priority work
    """
    P0_INTERACTIVE = 0
    P1_CLIENT = 1
    P2_SCHEDULED = 2
    P3_HEALING = 3
    P4_BULK = 4


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class JobState(str, Enum):
    QUEUED = "queued"
    LEASED = "leased"            # a worker has picked it up but not started
    RUNNING = "running"          # actively processing
    COMPLETED = "completed"
    FAILED = "failed"            # retryable, will be re-queued or dead-lettered
    DEAD_LETTER = "dead_letter"  # exhausted retries or non-retryable


TERMINAL_STATES = {JobState.COMPLETED, JobState.DEAD_LETTER}


def is_terminal(state: JobState) -> bool:
    return state in TERMINAL_STATES


# ---------------------------------------------------------------------------
# QueuedJob
# ---------------------------------------------------------------------------

@dataclass
class QueuedJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Idempotency: two submissions with the same key collapse into one job.
    idempotency_key: str = ""

    priority: Priority = Priority.P2_SCHEDULED
    state: JobState = JobState.QUEUED

    # Opaque payload — the queue never inspects this.
    payload: dict = field(default_factory=dict)

    # Ownership (tenant scoping — spec §43)
    client_id: str = "default"
    # A short label for debugging / UI.
    label: str = ""

    # Retry bookkeeping
    attempts: int = 0
    max_attempts: int = 3
    last_error: str = ""

    # Lease bookkeeping
    leased_by: str = ""              # worker_id
    leased_at: str = ""
    lease_expires_at: str = ""

    # Timestamps
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    # ------------------------------------------------------------------
    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """True if the lease has expired (worker crashed or hung)."""
        if self.state not in (JobState.LEASED, JobState.RUNNING):
            return False
        if not self.lease_expires_at:
            return False
        now = now or datetime.now(timezone.utc)
        try:
            exp = datetime.fromisoformat(self.lease_expires_at)
        except (ValueError, TypeError):
            return False
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return now >= exp

    def to_dict(self) -> dict:
        d = asdict(self)
        d["priority"] = int(self.priority)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "QueuedJob":
        d = dict(data)
        try:
            d["priority"] = Priority(int(d.get("priority", 2)))
        except (ValueError, TypeError):
            d["priority"] = Priority.P2_SCHEDULED
        try:
            d["state"] = JobState(d.get("state", "queued"))
        except ValueError:
            d["state"] = JobState.QUEUED
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Defaults
    j = QueuedJob()
    assert j.state == JobState.QUEUED
    assert j.priority == Priority.P2_SCHEDULED
    assert j.attempts == 0
    assert not j.is_expired()

    # Round trip
    j2 = QueuedJob(
        idempotency_key="abc",
        priority=Priority.P0_INTERACTIVE,
        payload={"url": "https://x.com"},
        client_id="acme",
        label="test",
    )
    d = j2.to_dict()
    assert d["priority"] == 0
    assert d["state"] == "queued"
    assert d["client_id"] == "acme"

    j3 = QueuedJob.from_dict(d)
    assert j3.idempotency_key == "abc"
    assert j3.priority == Priority.P0_INTERACTIVE
    assert j3.payload == {"url": "https://x.com"}
    assert j3.client_id == "acme"

    # Lease expiry
    j4 = QueuedJob(
        state=JobState.LEASED,
        lease_expires_at="2020-01-01T00:00:00+00:00",
    )
    assert j4.is_expired()

    j5 = QueuedJob(
        state=JobState.LEASED,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    assert not j5.is_expired()

    # Terminal states
    assert is_terminal(JobState.COMPLETED)
    assert is_terminal(JobState.DEAD_LETTER)
    assert not is_terminal(JobState.QUEUED)
    assert not is_terminal(JobState.RUNNING)

    # Priority ordering
    assert Priority.P0_INTERACTIVE < Priority.P4_BULK

    print("Queue types OK.")