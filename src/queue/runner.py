"""
Queue worker loop — spec §37.4.

Ties a QueueBackend to a job handler. The worker:

    1. Leases the next ready job (backend handles priority + expiry)
    2. Runs a heartbeat task so the lease doesn't expire mid-job
    3. Invokes the handler with the job's payload
    4. Marks the job complete OR fails it (retryable or not)
    5. Repeats

Design:
    - Single coroutine. Concurrency comes from running multiple
      workers, not from threads within one.
    - `run_once()` does exactly one lease/execute/complete cycle.
      `run_forever()` loops until stopped or idle for too long.
    - Handler is injected — the worker doesn't know about Pipelines,
      TaskSpecs, or anything else. It just calls
      `await handler(payload) -> dict`.
    - Heartbeats run at 1/3 the lease duration to leave slack for
      clock skew.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from src.queue.backend import QueueBackend
from src.queue.types import JobState, QueuedJob


logger = logging.getLogger("queue.runner")


JobHandler = Callable[[dict], Awaitable[Optional[dict]]]
RetryableFn = Callable[[Exception], bool]


# ---------------------------------------------------------------------------
# Result of one lease/execute cycle
# ---------------------------------------------------------------------------

@dataclass
class WorkerResult:
    leased: bool
    job_id: str = ""
    state: str = ""
    error: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "leased": self.leased,
            "job_id": self.job_id,
            "state": self.state,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class QueueWorker:
    def __init__(
        self,
        backend: QueueBackend,
        handler: JobHandler,
        worker_id: str,
        lease_seconds: int = 60,
        poll_interval: float = 1.0,
        retryable_fn: Optional[RetryableFn] = None,
    ):
        self.backend = backend
        self.handler = handler
        self.worker_id = worker_id
        self.lease_seconds = int(lease_seconds)
        self.poll_interval = float(poll_interval)
        self._retryable_fn = retryable_fn or (lambda e: True)
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Signal the worker loop to exit after the current job."""
        self._stop.set()

    async def run_once(self) -> WorkerResult:
        """
        Lease at most one job and process it. Returns a WorkerResult
        describing what happened. Never raises — handler exceptions
        are routed into `fail()`.
        """
        job = await self.backend.lease_next(
            self.worker_id, self.lease_seconds,
        )
        if job is None:
            return WorkerResult(leased=False)

        started = time.monotonic()
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(job.job_id))
        try:
            try:
                # Pass a shallow copy so a handler that mutates its
                # payload argument cannot corrupt the job record.
                safe_payload = (
                    dict(job.payload) if isinstance(job.payload, dict) else {}
                )
                result = await self.handler(safe_payload)
            except Exception as e:
                retryable = bool(self._retryable_fn(e))
                state = await self.backend.fail(
                    job.job_id, f"{type(e).__name__}: {e}", retryable=retryable,
                )
                return WorkerResult(
                    leased=True,
                    job_id=job.job_id,
                    state=state.value,
                    error=f"{type(e).__name__}: {e}",
                    duration_ms=int((time.monotonic() - started) * 1000),
                )

            ok = await self.backend.complete(job.job_id, result=result)
            return WorkerResult(
                leased=True,
                job_id=job.job_id,
                state=JobState.COMPLETED.value if ok else "unknown",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass

    async def run_forever(self, max_idle_polls: int = 0) -> int:
        """
        Loop until stopped. `max_idle_polls > 0` limits how many
        consecutive empty polls we tolerate before exiting — useful
        for one-shot "drain the queue" runs and tests.
        Returns the number of jobs processed.
        """
        processed = 0
        idle_polls = 0

        while not self._stop.is_set():
            result = await self.run_once()
            if result.leased:
                processed += 1
                idle_polls = 0
                logger.info(
                    f"worker {self.worker_id}: job {result.job_id} "
                    f"→ {result.state} ({result.duration_ms}ms)"
                )
            else:
                idle_polls += 1
                if max_idle_polls and idle_polls >= max_idle_polls:
                    break
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.poll_interval,
                    )
                except asyncio.TimeoutError:
                    pass

        return processed

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------
    async def _heartbeat_loop(self, job_id: str) -> None:
        """
        Re-extend the lease every lease_seconds/3 while the handler
        runs. Silently stops if the backend says we no longer hold it.
        """
        interval = max(1.0, self.lease_seconds / 3.0)
        try:
            while True:
                await asyncio.sleep(interval)
                ok = await self.backend.heartbeat(
                    job_id, self.worker_id, self.lease_seconds,
                )
                if not ok:
                    logger.warning(
                        f"worker {self.worker_id}: lost lease on {job_id}"
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                f"worker {self.worker_id}: heartbeat error on {job_id}: {e}"
            )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def run():
        from src.queue.backend import InMemoryQueueBackend
        from src.queue.types import Priority, QueuedJob

        q = InMemoryQueueBackend()
        handled: list[dict] = []

        async def handler(payload: dict):
            handled.append(payload)
            if payload.get("fail"):
                raise RuntimeError("handler asked to fail")
            return {"records": payload.get("n", 0) * 2}

        # Enqueue three jobs
        for i in range(3):
            await q.enqueue(QueuedJob(
                label=f"j{i}", payload={"n": i},
                priority=Priority.P2_SCHEDULED,
            ))

        # One-shot drain: worker exits after 2 idle polls
        w = QueueWorker(q, handler, "w-1", lease_seconds=30, poll_interval=0.1)
        processed = await w.run_forever(max_idle_polls=2)
        assert processed == 3, processed
        assert len(handled) == 3

        # All completed
        s = await q.stats()
        assert s["by_state"].get("completed", 0) == 3

        # --- Failure handling ---
        q2 = InMemoryQueueBackend()
        await q2.enqueue(QueuedJob(
            label="boom", payload={"fail": True}, max_attempts=1,
        ))

        async def h2(payload):
            raise ValueError("nope")

        w2 = QueueWorker(q2, h2, "w-2", lease_seconds=30, poll_interval=0.1)
        await w2.run_once()
        # With max_attempts=1, a failure goes straight to dead-letter
        letters = await q2.dead_letters()
        assert len(letters) == 1
        assert "ValueError: nope" in letters[0].last_error

        # --- Graceful stop ---
        q3 = InMemoryQueueBackend()
        for i in range(10):
            await q3.enqueue(QueuedJob(label=f"g{i}", payload={"n": i}))

        calls: list[int] = []
        async def h3(payload):
            calls.append(payload["n"])
            if len(calls) >= 3:
                w3.stop()
            return {}

        w3 = QueueWorker(q3, h3, "w-3", lease_seconds=30, poll_interval=0.05)
        processed = await w3.run_forever()
        assert processed == 3, processed
        assert len(calls) == 3

        print("QueueWorker OK.")

    asyncio.run(run())