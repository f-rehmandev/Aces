"""
Worker pool lifecycle — spec §37.4.

Manages a set of QueueWorkers running on the asyncio event loop of the
hosting process (typically FastAPI's). Provides:

    await pool.start()   — spawn N worker tasks
    await pool.stop()    — signal stop and await graceful drain
    pool.stats()         — snapshot of worker state

Design notes:

    - Each worker is a `QueueWorker` running `run_forever()` in its own
      asyncio task. All workers share the same `QueueBackend` — the
      backend is what makes them cooperate (leases prevent double-work).
    - `start()` is idempotent: calling it twice is a no-op.
    - `stop()` cancels worker tasks cleanly. In-flight jobs are NOT
      interrupted — `QueueWorker.run_forever()` only exits between
      leases, so the current job finishes first. The queue lease then
      expires naturally if it was held (or completes cleanly).
    - Never raises on startup failure of an individual worker. The
      pool is best-effort: if a worker task dies, its siblings keep
      going and the backend's lease expiry re-queues any orphaned job.
    - The pool does not own the backend. Callers construct it, hand it
      to the pool, and keep it if they want to enqueue directly.

Not implemented here (deferred):
    - Per-tenant concurrency limits — the backend's priority queue
      handles ordering, but capping "at most N jobs per client" is a
      policy layer on top. Roadmap item.
    - Dynamic scaling — worker count is fixed at construction. Fine
      for single-process deployments.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.queue.backend import QueueBackend
from src.queue.runner import JobHandler, QueueWorker


logger = logging.getLogger("queue.lifecycle")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PoolStats:
    workers: int = 0
    running: int = 0
    tasks_alive: int = 0
    started: bool = False

    def to_dict(self) -> dict:
        return {
            "workers": self.workers,
            "running": self.running,
            "tasks_alive": self.tasks_alive,
            "started": self.started,
        }


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

class WorkerPool:
    """
    Manages N `QueueWorker` tasks on the current event loop.

    Usage in a FastAPI lifespan:

        pool = WorkerPool(backend, handler, worker_count=2)
        await pool.start()
        try:
            yield
        finally:
            await pool.stop()
    """

    def __init__(
        self,
        backend: QueueBackend,
        handler: JobHandler,
        *,
        worker_count: int = 2,
        lease_seconds: int = 60,
        poll_interval: float = 1.0,
        retryable_fn=None,
    ):
        if worker_count < 0:
            raise ValueError("worker_count must be >= 0")
        self.backend = backend
        self.handler = handler
        self.worker_count = int(worker_count)
        self.lease_seconds = int(lease_seconds)
        self.poll_interval = float(poll_interval)
        self.retryable_fn = retryable_fn

        self._workers: list[QueueWorker] = []
        self._tasks: list[asyncio.Task] = []
        self._started = False
        self._stopping = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """
        Spawn the workers. Idempotent.
        """
        if self._started:
            return
        self._started = True
        self._stopping = False

        for i in range(self.worker_count):
            worker_id = f"pool-{id(self):x}-w{i}"
            worker = QueueWorker(
                backend=self.backend,
                handler=self.handler,
                worker_id=worker_id,
                lease_seconds=self.lease_seconds,
                poll_interval=self.poll_interval,
                retryable_fn=self.retryable_fn,
            )
            self._workers.append(worker)
            task = asyncio.create_task(
                worker.run_forever(),
                name=f"aces-worker-{i}",
            )
            self._tasks.append(task)

        logger.info(f"WorkerPool started: {self.worker_count} worker(s)")

    async def stop(self) -> None:
        """
        Signal each worker to stop and wait for its current job to
        finish. In-flight jobs are never interrupted.
        """
        if not self._started or self._stopping:
            return
        self._stopping = True

        for w in self._workers:
            w.stop()

        # Wait for each task to exit. `run_forever` checks the stop flag
        # between leases, so this returns as soon as the current job
        # completes — with a short poll interval, that's seconds.
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        self._workers.clear()
        self._started = False
        self._stopping = False
        logger.info("WorkerPool stopped cleanly")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stats(self) -> PoolStats:
        alive = sum(1 for t in self._tasks if not t.done())
        return PoolStats(
            workers=self.worker_count,
            running=len(self._workers) if self._started else 0,
            tasks_alive=alive,
            started=self._started,
        )

    @property
    def is_started(self) -> bool:
        return self._started


# ---------------------------------------------------------------------------
# Smoke test — in-memory backend, no network, no subprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.queue.backend import InMemoryQueueBackend
    from src.queue.types import QueuedJob

    async def run():
        # ---- 1. Zero workers is a clean no-op ----
        backend0 = InMemoryQueueBackend()
        pool0 = WorkerPool(backend0, handler=_noop_handler, worker_count=0)
        await pool0.start()
        await pool0.stop()

        # ---- 2. Full lifecycle: enqueue → worker picks up → completes ----
        backend = InMemoryQueueBackend()
        handled: list[dict] = []

        async def handler(payload):
            handled.append(payload)
            return {"ok": True, "seen": payload.get("n", 0)}

        pool = WorkerPool(
            backend, handler,
            worker_count=2, poll_interval=0.05,
        )
        await pool.start()
        assert pool.is_started
        assert pool.stats().tasks_alive == 2

        # Enqueue 4 jobs, then poll until they're all done
        jobs = [QueuedJob(payload={"n": i}) for i in range(4)]
        for j in jobs:
            await backend.enqueue(j)

        # Give the workers a moment
        for _ in range(50):
            await asyncio.sleep(0.05)
            s = await backend.stats()
            if s["by_state"].get("completed", 0) == 4:
                break

        assert len(handled) == 4
        s = await backend.stats()
        assert s["by_state"].get("completed", 0) == 4, s

        # ---- 3. Job payloads reached the handler ----
        got_n = sorted(h["n"] for h in handled)
        assert got_n == [0, 1, 2, 3]

        # ---- 4. stop() drains gracefully ----
        await pool.stop()
        assert not pool.is_started
        assert pool.stats().tasks_alive == 0

        # ---- 5. start() is idempotent ----
        await pool.start()
        await pool.start()
        assert pool.stats().tasks_alive == 2
        await pool.stop()

        # ---- 6. stop() is idempotent ----
        await pool.stop()
        await pool.stop()

        # ---- 7. Enqueue before start → picked up after start ----
        backend2 = InMemoryQueueBackend()
        picked: list[dict] = []

        async def handler2(payload):
            picked.append(payload)
            return {}

        await backend2.enqueue(QueuedJob(payload={"early": True}))
        pool2 = WorkerPool(
            backend2, handler2, worker_count=1, poll_interval=0.05,
        )
        await pool2.start()
        for _ in range(40):
            await asyncio.sleep(0.05)
            if picked:
                break
        assert len(picked) == 1
        assert picked[0] == {"early": True}
        await pool2.stop()

        print("WorkerPool OK.")

    async def _noop_handler(payload):
        return {}

    asyncio.run(run())