"""Unit tests for the durable queue (spec §37)."""
import asyncio

import pytest

from src.queue.backend import InMemoryQueueBackend, QueueBackend
from src.queue.supabase_backend import _row_to_job, _now_iso, _future_iso
from src.queue.types import (
    JobState,
    Priority,
    QueuedJob,
    is_terminal,
    TERMINAL_STATES,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def test_job_defaults():
    j = QueuedJob()
    assert j.state == JobState.QUEUED
    assert j.priority == Priority.P2_SCHEDULED
    assert j.attempts == 0
    assert j.max_attempts == 3
    assert not j.is_expired()


def test_job_roundtrip():
    j = QueuedJob(
        idempotency_key="k",
        priority=Priority.P0_INTERACTIVE,
        payload={"x": 1},
        client_id="acme",
        label="test",
    )
    d = j.to_dict()
    assert d["priority"] == 0
    assert d["state"] == "queued"

    j2 = QueuedJob.from_dict(d)
    assert j2.priority == Priority.P0_INTERACTIVE
    assert j2.payload == {"x": 1}
    assert j2.client_id == "acme"


def test_job_from_dict_bad_priority():
    j = QueuedJob.from_dict({"priority": "bogus"})
    assert j.priority == Priority.P2_SCHEDULED


def test_job_from_dict_bad_state():
    j = QueuedJob.from_dict({"state": "nonsense"})
    assert j.state == JobState.QUEUED


def test_job_lease_expiry_past():
    j = QueuedJob(
        state=JobState.LEASED,
        lease_expires_at="2020-01-01T00:00:00+00:00",
    )
    assert j.is_expired()


def test_job_lease_expiry_future():
    j = QueuedJob(
        state=JobState.LEASED,
        lease_expires_at="2099-01-01T00:00:00+00:00",
    )
    assert not j.is_expired()


def test_job_lease_expiry_ignores_queued():
    j = QueuedJob(lease_expires_at="2020-01-01T00:00:00+00:00")
    assert j.state == JobState.QUEUED
    assert not j.is_expired()   # only LEASED/RUNNING can expire


def test_priority_ordering():
    assert Priority.P0_INTERACTIVE < Priority.P1_CLIENT
    assert Priority.P1_CLIENT < Priority.P2_SCHEDULED
    assert Priority.P2_SCHEDULED < Priority.P3_HEALING
    assert Priority.P3_HEALING < Priority.P4_BULK


def test_terminal_states():
    assert is_terminal(JobState.COMPLETED)
    assert is_terminal(JobState.DEAD_LETTER)
    assert not is_terminal(JobState.QUEUED)
    assert not is_terminal(JobState.LEASED)
    assert not is_terminal(JobState.RUNNING)
    assert not is_terminal(JobState.FAILED)


# ---------------------------------------------------------------------------
# In-memory: basic flow
# ---------------------------------------------------------------------------

@pytest.fixture
def q():
    return InMemoryQueueBackend()


def test_empty_queue(q):
    assert asyncio.run(q.lease_next("w-1", 60)) is None
    s = asyncio.run(q.stats())
    assert s["total"] == 0


def test_enqueue_returns_job_id(q):
    j = QueuedJob(label="x")
    jid = asyncio.run(q.enqueue(j))
    assert jid == j.job_id


def test_enqueue_then_lease(q):
    j = QueuedJob(label="x")
    asyncio.run(q.enqueue(j))
    leased = asyncio.run(q.lease_next("w-1", 60))
    assert leased is not None
    assert leased.job_id == j.job_id
    assert leased.state == JobState.LEASED
    assert leased.leased_by == "w-1"
    assert leased.lease_expires_at


def test_lease_returns_none_when_empty(q):
    assert asyncio.run(q.lease_next("w-1", 60)) is None


def test_heartbeat_by_holder(q):
    j = QueuedJob()
    asyncio.run(q.enqueue(j))
    asyncio.run(q.lease_next("w-1", 60))
    assert asyncio.run(q.heartbeat(j.job_id, "w-1", 120))


def test_heartbeat_by_other_worker_rejected(q):
    j = QueuedJob()
    asyncio.run(q.enqueue(j))
    asyncio.run(q.lease_next("w-1", 60))
    assert not asyncio.run(q.heartbeat(j.job_id, "w-2", 120))


def test_complete_marks_job(q):
    j = QueuedJob()
    asyncio.run(q.enqueue(j))
    asyncio.run(q.lease_next("w-1", 60))
    assert asyncio.run(q.complete(j.job_id, result={"n": 1}))
    got = asyncio.run(q.get(j.job_id))
    assert got.state == JobState.COMPLETED
    assert got.payload["_result"] == {"n": 1}


def test_complete_is_idempotent(q):
    j = QueuedJob()
    asyncio.run(q.enqueue(j))
    asyncio.run(q.lease_next("w-1", 60))
    assert asyncio.run(q.complete(j.job_id))
    assert asyncio.run(q.complete(j.job_id))   # second call ok


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_idempotent_enqueue_returns_same_id(q):
    a = QueuedJob(idempotency_key="same-key")
    b = QueuedJob(idempotency_key="same-key")
    id_a = asyncio.run(q.enqueue(a))
    id_b = asyncio.run(q.enqueue(b))
    assert id_a == id_b


def test_different_keys_different_jobs(q):
    a = QueuedJob(idempotency_key="k1")
    b = QueuedJob(idempotency_key="k2")
    assert asyncio.run(q.enqueue(a)) != asyncio.run(q.enqueue(b))


def test_re_enqueue_allowed_after_terminal(q):
    a = QueuedJob(idempotency_key="k")
    id_a = asyncio.run(q.enqueue(a))
    asyncio.run(q.lease_next("w-1", 60))
    asyncio.run(q.complete(id_a))

    b = QueuedJob(idempotency_key="k")
    id_b = asyncio.run(q.enqueue(b))
    assert id_b != id_a


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

def test_priority_ordering_in_lease(q):
    bulk = QueuedJob(label="bulk", priority=Priority.P4_BULK)
    client = QueuedJob(label="client", priority=Priority.P1_CLIENT)
    interactive = QueuedJob(label="interactive", priority=Priority.P0_INTERACTIVE)
    for j in (bulk, client, interactive):
        asyncio.run(q.enqueue(j))

    assert asyncio.run(q.lease_next("w", 60)).label == "interactive"
    assert asyncio.run(q.lease_next("w", 60)).label == "client"
    assert asyncio.run(q.lease_next("w", 60)).label == "bulk"


def test_fifo_within_same_priority(q):
    a = QueuedJob(label="a", priority=Priority.P2_SCHEDULED)
    b = QueuedJob(label="b", priority=Priority.P2_SCHEDULED)
    c = QueuedJob(label="c", priority=Priority.P2_SCHEDULED)
    for j in (a, b, c):
        asyncio.run(q.enqueue(j))

    assert asyncio.run(q.lease_next("w", 60)).label == "a"
    assert asyncio.run(q.lease_next("w", 60)).label == "b"
    assert asyncio.run(q.lease_next("w", 60)).label == "c"


# ---------------------------------------------------------------------------
# Fail / retry / dead-letter
# ---------------------------------------------------------------------------

def test_retryable_failure_requeues(q):
    j = QueuedJob(max_attempts=3, label="retry-me")
    asyncio.run(q.enqueue(j))
    asyncio.run(q.lease_next("w-1", 60))
    state = asyncio.run(q.fail(j.job_id, "transient", retryable=True))
    assert state == JobState.QUEUED
    got = asyncio.run(q.get(j.job_id))
    assert got.attempts == 1
    assert got.last_error == "transient"


def test_exhausted_retryable_goes_to_dead_letter(q):
    j = QueuedJob(max_attempts=2, label="doomed")
    asyncio.run(q.enqueue(j))

    # Attempt 1: fail retryable → requeue
    asyncio.run(q.lease_next("w", 60))
    asyncio.run(q.fail(j.job_id, "still broken", retryable=True))

    # Attempt 2: fail retryable → attempts == max_attempts → DEAD_LETTER
    asyncio.run(q.lease_next("w", 60))
    state = asyncio.run(q.fail(j.job_id, "still broken", retryable=True))
    assert state == JobState.DEAD_LETTER


def test_non_retryable_immediate_dead_letter(q):
    j = QueuedJob(max_attempts=5, label="perm")
    asyncio.run(q.enqueue(j))
    asyncio.run(q.lease_next("w", 60))
    state = asyncio.run(q.fail(j.job_id, "permanent", retryable=False))
    assert state == JobState.DEAD_LETTER


def test_dead_letters_listing(q):
    j = QueuedJob(max_attempts=1)
    asyncio.run(q.enqueue(j))
    asyncio.run(q.lease_next("w", 60))
    asyncio.run(q.fail(j.job_id, "e", retryable=False))

    letters = asyncio.run(q.dead_letters())
    assert len(letters) == 1
    assert letters[0].job_id == j.job_id


def test_requeue_dead_letter(q):
    j = QueuedJob(max_attempts=1)
    asyncio.run(q.enqueue(j))
    asyncio.run(q.lease_next("w", 60))
    asyncio.run(q.fail(j.job_id, "e", retryable=False))

    assert asyncio.run(q.requeue_dead_letter(j.job_id))
    got = asyncio.run(q.get(j.job_id))
    assert got.state == JobState.QUEUED
    assert got.attempts == 0


def test_requeue_non_dead_letter_returns_false(q):
    j = QueuedJob()
    asyncio.run(q.enqueue(j))
    assert not asyncio.run(q.requeue_dead_letter(j.job_id))


# ---------------------------------------------------------------------------
# Lease expiry recovery
# ---------------------------------------------------------------------------

def test_expired_lease_recovers(q):
    j = QueuedJob(max_attempts=3, label="exp")
    asyncio.run(q.enqueue(j))
    asyncio.run(q.lease_next("w-1", lease_seconds=0))   # expires immediately

    n = asyncio.run(q.recover_expired_leases())
    assert n == 1
    got = asyncio.run(q.get(j.job_id))
    assert got.state == JobState.QUEUED
    assert got.attempts == 1
    assert "expired" in got.last_error


def test_expired_lease_exhausted_goes_to_dead_letter(q):
    j = QueuedJob(max_attempts=2)
    asyncio.run(q.enqueue(j))

    # Cycle 1: lease expires → recover → attempts=1 (still retryable)
    asyncio.run(q.lease_next("w", lease_seconds=0))
    asyncio.run(q.recover_expired_leases())
    got = asyncio.run(q.get(j.job_id))
    assert got.state == JobState.QUEUED

    # Cycle 2: lease expires → recover → attempts=2 >= max → dead-letter
    asyncio.run(q.lease_next("w", lease_seconds=0))
    asyncio.run(q.recover_expired_leases())
    got = asyncio.run(q.get(j.job_id))
    assert got.state == JobState.DEAD_LETTER


def test_recover_no_expired_returns_zero(q):
    j = QueuedJob()
    asyncio.run(q.enqueue(j))
    asyncio.run(q.lease_next("w", 60))   # not expired
    assert asyncio.run(q.recover_expired_leases()) == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats_shape(q):
    asyncio.run(q.enqueue(QueuedJob()))
    s = asyncio.run(q.stats())
    assert s["total"] == 1
    assert "by_state" in s
    assert "by_priority" in s
    assert s["by_state"].get("queued", 0) == 1


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def test_row_to_job_basic():
    row = {
        "job_id": "x",
        "idempotency_key": "k",
        "priority": 1,
        "state": "queued",
        "payload": {"a": 1},
        "client_id": "acme",
        "label": "lbl",
        "attempts": 2,
        "max_attempts": 5,
        "last_error": "",
        "leased_by": "",
    }
    j = _row_to_job(row)
    assert j.job_id == "x"
    assert j.priority == Priority.P1_CLIENT
    assert j.state == JobState.QUEUED
    assert j.payload == {"a": 1}
    assert j.attempts == 2


def test_row_to_job_handles_bad_priority():
    j = _row_to_job({"job_id": "x", "priority": "nonsense"})
    assert j.priority == Priority.P2_SCHEDULED


def test_row_to_job_handles_bad_state():
    j = _row_to_job({"job_id": "x", "state": "invalid"})
    assert j.state == JobState.QUEUED


def test_row_to_job_handles_missing_payload():
    j = _row_to_job({"job_id": "x"})
    assert j.payload == {}


def test_now_iso_format():
    s = _now_iso()
    # ISO 8601 with milliseconds
    assert "T" in s
    assert "." in s


def test_future_iso_in_future():
    from datetime import datetime, timezone
    s = _future_iso(60)
    dt = datetime.fromisoformat(s)
    assert dt > datetime.now(timezone.utc)