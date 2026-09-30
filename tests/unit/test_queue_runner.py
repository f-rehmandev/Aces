"""Unit tests for QueueWorker (spec §37.4)."""
import asyncio
import time

import pytest

from src.queue.backend import InMemoryQueueBackend
from src.queue.runner import QueueWorker, WorkerResult
from src.queue.types import JobState, Priority, QueuedJob


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def q():
    return InMemoryQueueBackend()


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Empty queue
# ---------------------------------------------------------------------------

def test_run_once_empty_returns_not_leased(q):
    async def handler(payload):
        return {}

    w = QueueWorker(q, handler, "w-1")
    r = _run(w.run_once())
    assert isinstance(r, WorkerResult)
    assert r.leased is False
    assert r.job_id == ""
    assert r.duration_ms == 0


def test_worker_result_to_dict():
    r = WorkerResult(leased=True, job_id="x", state="completed",
                     error="", duration_ms=42)
    d = r.to_dict()
    assert d["leased"] is True
    assert d["job_id"] == "x"
    assert d["duration_ms"] == 42


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_worker_processes_a_job(q):
    handled = []

    async def handler(payload):
        handled.append(payload)
        return {"records": 5}

    _run(q.enqueue(QueuedJob(payload={"url": "https://x.com"})))
    w = QueueWorker(q, handler, "w-1", lease_seconds=30)

    r = _run(w.run_once())
    assert r.leased is True
    assert r.state == JobState.COMPLETED.value
    assert len(handled) == 1
    assert handled[0] == {"url": "https://x.com"}


def test_worker_marks_job_completed(q):
    async def handler(payload):
        return {"records": 42}

    j = QueuedJob(payload={"a": 1})
    _run(q.enqueue(j))
    w = QueueWorker(q, handler, "w-1")

    _run(w.run_once())
    got = _run(q.get(j.job_id))
    assert got.state == JobState.COMPLETED
    assert got.payload["_result"] == {"records": 42}


def test_worker_passes_payload_verbatim(q):
    seen = {}

    async def handler(payload):
        seen.update(payload)
        return {}

    _run(q.enqueue(QueuedJob(payload={
        "url": "https://x.com",
        "nested": {"k": [1, 2, 3]},
    })))
    w = QueueWorker(q, handler, "w-1")
    _run(w.run_once())
    assert seen["url"] == "https://x.com"
    assert seen["nested"] == {"k": [1, 2, 3]}


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def test_handler_exception_marks_failed(q):
    async def handler(payload):
        raise RuntimeError("boom")

    j = QueuedJob(max_attempts=5)
    _run(q.enqueue(j))
    w = QueueWorker(q, handler, "w-1")

    r = _run(w.run_once())
    assert r.leased is True
    assert r.error.startswith("RuntimeError")
    got = _run(q.get(j.job_id))
    # retryable by default → re-queued
    assert got.state == JobState.QUEUED
    assert got.attempts == 1
    assert "boom" in got.last_error


def test_non_retryable_fn_goes_to_dead_letter(q):
    async def handler(payload):
        raise RuntimeError("permanent")

    def not_retryable(exc):
        return False

    j = QueuedJob(max_attempts=5)
    _run(q.enqueue(j))
    w = QueueWorker(q, handler, "w-1", retryable_fn=not_retryable)

    _run(w.run_once())
    got = _run(q.get(j.job_id))
    assert got.state == JobState.DEAD_LETTER


def test_custom_retryable_fn_classifies_by_exception():
    """
    Two queues, one for each exception class — avoids a subtle
    FIFO-ordering interaction where the second worker leases the
    first job again after it re-queued.
    """
    class PermanentError(Exception):
        pass

    class TransientError(Exception):
        pass

    def classify(exc):
        return isinstance(exc, TransientError)

    # --- Queue 1: transient → retryable → QUEUED ---
    q1 = InMemoryQueueBackend()

    async def h_transient(payload):
        raise TransientError("net glitch")

    j1 = QueuedJob(max_attempts=5)
    _run(q1.enqueue(j1))
    w1 = QueueWorker(q1, h_transient, "w-1", retryable_fn=classify)
    _run(w1.run_once())
    assert _run(q1.get(j1.job_id)).state == JobState.QUEUED
    assert _run(q1.get(j1.job_id)).attempts == 1

    # --- Queue 2: permanent → non-retryable → DEAD_LETTER ---
    q2 = InMemoryQueueBackend()

    async def h_permanent(payload):
        raise PermanentError("bad input")

    j2 = QueuedJob(max_attempts=5)
    _run(q2.enqueue(j2))
    w2 = QueueWorker(q2, h_permanent, "w-2", retryable_fn=classify)
    _run(w2.run_once())
    assert _run(q2.get(j2.job_id)).state == JobState.DEAD_LETTER

# ---------------------------------------------------------------------------
# run_forever
# ---------------------------------------------------------------------------

def test_run_forever_processes_all_then_stops(q):
    handled = []

    async def handler(payload):
        handled.append(payload["i"])
        return {}

    for i in range(5):
        _run(q.enqueue(QueuedJob(payload={"i": i})))

    w = QueueWorker(q, handler, "w-1", poll_interval=0.05)
    processed = _run(w.run_forever(max_idle_polls=2))
    assert processed == 5
    assert sorted(handled) == [0, 1, 2, 3, 4]


def test_run_forever_respects_stop(q):
    calls = []

    async def handler(payload):
        calls.append(payload["i"])
        if len(calls) >= 3:
            w.stop()
        return {}

    for i in range(10):
        _run(q.enqueue(QueuedJob(payload={"i": i})))

    w = QueueWorker(q, handler, "w-1", poll_interval=0.05)
    processed = _run(w.run_forever())
    assert processed == 3
    assert len(calls) == 3
    # The remaining 7 jobs stay queued
    assert _run(q.stats())["by_state"].get("queued", 0) == 7


def test_run_forever_empty_queue_exits(q):
    async def handler(payload):
        return {}

    w = QueueWorker(q, handler, "w-1", poll_interval=0.05)
    processed = _run(w.run_forever(max_idle_polls=1))
    assert processed == 0


# ---------------------------------------------------------------------------
# Priority is honoured through the worker
# ---------------------------------------------------------------------------

def test_worker_processes_high_priority_first(q):
    order = []

    async def handler(payload):
        order.append(payload["label"])
        return {}

    # Enqueue in reverse priority order
    _run(q.enqueue(QueuedJob(label="bulk", priority=Priority.P4_BULK,
                              payload={"label": "bulk"})))
    _run(q.enqueue(QueuedJob(label="client", priority=Priority.P1_CLIENT,
                              payload={"label": "client"})))
    _run(q.enqueue(QueuedJob(label="interactive",
                              priority=Priority.P0_INTERACTIVE,
                              payload={"label": "interactive"})))

    w = QueueWorker(q, handler, "w-1", poll_interval=0.05)
    _run(w.run_forever(max_idle_polls=2))
    assert order == ["interactive", "client", "bulk"]


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def test_heartbeat_extends_lease_during_long_handler(q):
    """
    Use a tiny lease (0.6s) and a slow handler (1.5s). Without
    heartbeats the lease would expire mid-job; with heartbeats the
    job completes cleanly.
    """
    async def slow_handler(payload):
        await asyncio.sleep(1.5)
        return {"ok": True}

    j = QueuedJob(payload={"x": 1})
    _run(q.enqueue(j))
    w = QueueWorker(q, slow_handler, "w-1", lease_seconds=1)

    r = _run(w.run_once())
    assert r.state == JobState.COMPLETED.value
    got = _run(q.get(j.job_id))
    assert got.state == JobState.COMPLETED


def test_heartbeat_stops_when_handler_finishes(q):
    """
    After a handler finishes, we shouldn't see any 'lease lost'
    warnings or stray heartbeat calls. This is a behavioural check:
    the run_once() should return promptly after completion.
    """
    async def quick_handler(payload):
        return {}

    _run(q.enqueue(QueuedJob()))
    w = QueueWorker(q, quick_handler, "w-1", lease_seconds=30)
    started = time.monotonic()
    _run(w.run_once())
    elapsed = time.monotonic() - started
    # Should complete in well under the heartbeat interval (10s)
    assert elapsed < 5.0


# ---------------------------------------------------------------------------
# Multiple workers do not double-process
# ---------------------------------------------------------------------------

def test_two_workers_do_not_share_a_job(q):
    handled = []

    async def handler(payload):
        handled.append(payload["i"])
        await asyncio.sleep(0.05)   # let the other worker poll
        return {}

    for i in range(3):
        _run(q.enqueue(QueuedJob(payload={"i": i})))

    async def run_both():
        w1 = QueueWorker(q, handler, "w-1", poll_interval=0.02)
        w2 = QueueWorker(q, handler, "w-2", poll_interval=0.02)
        await asyncio.gather(
            w1.run_forever(max_idle_polls=3),
            w2.run_forever(max_idle_polls=3),
        )

    _run(run_both())
    # Every job handled exactly once
    assert sorted(handled) == [0, 1, 2]
    s = _run(q.stats())
    assert s["by_state"].get("completed", 0) == 3


# ---------------------------------------------------------------------------
# Idempotency is honoured by the worker loop
# ---------------------------------------------------------------------------

def test_worker_idempotency_via_backend(q):
    """
    Enqueue the same key twice. Only one job should reach the handler.
    """
    calls = []

    async def handler(payload):
        calls.append(payload)
        return {}

    _run(q.enqueue(QueuedJob(idempotency_key="K", payload={"first": True})))
    _run(q.enqueue(QueuedJob(idempotency_key="K", payload={"first": False})))

    w = QueueWorker(q, handler, "w-1", poll_interval=0.05)
    processed = _run(w.run_forever(max_idle_polls=2))
    assert processed == 1
    assert calls == [{"first": True}]