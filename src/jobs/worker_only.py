"""
Worker-only process entrypoint — spec §37.4.

Runs a WorkerPool against the durable queue and exits on SIGTERM /
SIGINT. No HTTP server. Used by the `worker` mode of the container so
that API replicas and worker replicas can scale independently.

Only the queue handler runs here. Schedules are driven by the API
process (SchedulerLoop writes into the queue, workers consume from
it — one queue, one producer, many consumers).
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from src.api.registry import ServiceRegistry
from src.integrations.factory import build_production_registry
from src.jobs.executor import JobExecutor, make_queue_handler
from src.network.manager import build_production_manager
from src.queue.lifecycle import WorkerPool
from src.storage.batch_store import build_supabase_store


logger = logging.getLogger("jobs.worker_only")


def _env_int(name: str, default: int) -> int:
    v = (os.getenv(name) or "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


async def _run() -> None:
    logging.basicConfig(
        level=(os.getenv("ACES_LOG_LEVEL") or "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    worker_count = _env_int("ACES_WORKER_COUNT", 2)
    logger.info(f"worker-only mode starting ({worker_count} workers)")

    # --- queue ---
    supabase_url = (os.getenv("SUPABASE_URL") or "").strip()
    if supabase_url:
        try:
            from src.queue.supabase_backend import SupabaseQueueBackend
            backend = SupabaseQueueBackend()
            _ = backend._client()
            logger.info("queue: durable (Supabase)")
        except Exception as e:
            logger.warning(
                f"queue: falling back to in-memory "
                f"({type(e).__name__}: {e})"
            )
            from src.queue.backend import InMemoryQueueBackend
            backend = InMemoryQueueBackend()
    else:
        from src.queue.backend import InMemoryQueueBackend
        logger.info("queue: in-memory (SUPABASE_URL not set)")
        backend = InMemoryQueueBackend()

    # --- pipeline deps ---
    from src.scraper.engine import ScraperEngine
    from src.extractor.schema_extractor import DataExtractor

    scraper = ScraperEngine()
    extractor = DataExtractor()
    try:
        network_manager = build_production_manager(scraper)
    except Exception:
        from src.network.manager import NetworkManager
        network_manager = NetworkManager(scraper, None)

    try:
        connector_registry = build_production_registry()
    except Exception:
        connector_registry = None

    registry = ServiceRegistry()
    executor = JobExecutor(
        registry, scraper, extractor,
        runner_kwargs={
            "network_manager": network_manager,
            "connector_registry": connector_registry,
        },
    )

    batch_store = None
    if (os.getenv("SUPABASE_URL") or "").strip():
        try:
            batch_store = build_supabase_store()
            logger.info("batch store: durable (Supabase)")
        except Exception as e:
            logger.warning(
                f"batch store: unavailable ({type(e).__name__}: {e})"
            )

    handler = make_queue_handler(
        executor,
        registry,
        batch_store=batch_store,
    )

    pool = WorkerPool(backend, handler, worker_count=worker_count)

    # --- signals ---
    stop = asyncio.Event()

    def _signal_handler(*_):
        logger.info("worker-only received stop signal")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows: add_signal_handler not available; fall back
            signal.signal(sig, lambda *_: stop.set())

    await pool.start()
    logger.info("worker-only ready; awaiting jobs")

    try:
        await stop.wait()
    finally:
        await pool.stop()
        logger.info("worker-only stopped cleanly")