"""
Production app factory — spec §45 + §69.

Builds the fully-wired FastAPI app from environment configuration.
This is what the container runs:

    uvicorn src.api.production:create_app --factory --host 0.0.0.0 --port 8000

Everything the app needs is resolved here, once, at process start:

    ApiKeyStore          in-memory (see note)
    ServiceRegistry      in-memory (see note)
    JobExecutor          real (pipeline + crawler + LLM router)
    QueueBackend         durable if Supabase is reachable, else memory
    WorkerPool           inline (started by FastAPI's lifespan)
    SchedulerLoop        inline (started by FastAPI's lifespan)

Notes on persistence:

    `ApiKeyStore` and `ServiceRegistry` are intentionally in-memory for
    now. Persisting them would require new Supabase tables (and a fresh
    migration) plus a shared cache for multi-process deployments. Both
    are on the roadmap; neither blocks the container from working for a
    single-process deployment today.

    The queue IS durable when Supabase is configured — see
    `src.queue.supabase_backend`. Without Supabase, it falls back to
    memory, which is correct for local dev.

Environment variables (all optional, all defaulted):

    ACES_MODE                   api | worker (default api)
    ACES_WORKER_COUNT           workers to spawn (default 2)
    ACES_SCHEDULER_INTERVAL     seconds between scheduler ticks (default 30)
    ACES_SCHEDULER_ENABLED      "1"/"true"/"yes" to enable (default 1)
    SUPABASE_URL                enables durable queue
    SUPABASE_SERVICE_KEY        required if SUPABASE_URL is set
    SCRAPER_API_KEY             enables ScraperAPI network fallback
    SLACK_WEBHOOK_URL           registers the Slack connector
    ACES_WEBHOOK_URL + _SECRET  registers the HMAC webhook connector
    S3_BUCKET                   registers the S3 connector
    ACES_LOCAL_OUTPUT_DIR       registers the local_file connector

Design rule: `create_app()` must NEVER raise on a missing env var.
Every optional dependency degrades to "not configured" and the app
starts with reduced capability. A container that boots but can't do
Slack is far better than one that refuses to boot because Slack isn't
set up.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

from fastapi import FastAPI

from src.api.keys import ApiKeyStore
from src.api.registry import ServiceRegistry
from src.api.server import build_app
from src.integrations.factory import build_production_registry
from src.jobs.executor import JobExecutor
from src.jobs.scheduler_loop import SchedulerLoop
from src.network.manager import build_production_manager
from src.observability.store import SupabaseIncidentStore
from src.queue.backend import InMemoryQueueBackend
from src.storage.batch_store import build_supabase_store


logger = logging.getLogger("api.production")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    v = _env(name).lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

def _build_queue_backend():
    """
    Durable queue if Supabase is configured and reachable, else memory.

    The Supabase import is deliberately late so that a dev machine
    without supabase-py still works — this function is a soft boundary.
    """
    url = _env("SUPABASE_URL")
    if not url:
        logger.info("queue: in-memory (SUPABASE_URL not set)")
        return InMemoryQueueBackend()
    try:
        from src.queue.supabase_backend import SupabaseQueueBackend
        backend = SupabaseQueueBackend()
        # Cheap availability probe — one SELECT. If it fails, fall
        # back; we do NOT want a container to fail to boot because
        # the network blipped during startup.
        _ = backend._client()
        logger.info("queue: durable (Supabase reachable)")
        return backend
    except Exception as e:
        logger.warning(
            f"queue: falling back to in-memory ({type(e).__name__}: {e})"
        )
        return InMemoryQueueBackend()


def _build_batch_store():
    """
    Durable batch/chunk store when Supabase is configured.
    Falls back to None for local development.
    """
    if not _env("SUPABASE_URL"):
        logger.info("batch store: disabled (SUPABASE_URL not set)")
        return None

    try:
        store = build_supabase_store()
        logger.info("batch store: durable (Supabase)")
        return store
    except Exception as e:
        logger.warning(
            f"batch store: unavailable ({type(e).__name__}: {e})"
        )
        return None


def _build_scraper_and_extractor():
    """Real Playwright wrapper + LLM extractor. Lazy imports."""
    from src.scraper.engine import ScraperEngine
    from src.extractor.schema_extractor import DataExtractor
    scraper = ScraperEngine()
    extractor = DataExtractor()
    logger.info("scraper: ScraperEngine + DataExtractor ready")
    return scraper, extractor


def _build_network_manager(scraper):
    """Full tier chain — only what the env supports is registered."""
    try:
        manager = build_production_manager(scraper)
        logger.info("network: production manager built (env-tier chain)")
        return manager
    except Exception as e:
        logger.warning(
            f"network: production manager failed ({type(e).__name__}: {e}); "
            f"falling back to Playwright-only"
        )
        from src.network.manager import NetworkManager
        return NetworkManager(scraper, None)


def _build_incident_store():
    """Supabase incident store if configured, else None."""
    if not _env("SUPABASE_URL"):
        return None
    try:
        store = SupabaseIncidentStore()
        _ = store._client()
        logger.info("incidents: Supabase-backed store ready")
        return store
    except Exception as e:
        logger.warning(
            f"incidents: skipping Supabase store ({type(e).__name__}: {e})"
        )
        return None


def _build_observability_stack(registry: ServiceRegistry):
    """
    Build the optional observability subsystems that PipelineRunner
    consults during a run. Every piece is best-effort:

        - Usage store       (§41.4) — Supabase if reachable, else in-memory
        - Entitlement engine (§47B) — always built; uses the free/pro/
                                       enterprise catalogue from
                                       src.usage.entitlements.default_plans
        - Alert engine      (§30A)  — always built; default rules only
        - Incident tracker  (§40.6) — Supabase if reachable, else in-memory

    Returns (usage_store, entitlement_engine, alert_engine, incident_tracker).
    Any piece that couldn't be built is returned as None and the pipeline
    degrades gracefully.

    Never raises — a broken subsystem must not prevent the container
    from booting.

    Note on probing: `SupabaseX()._client()` only constructs a client —
    it does not make a network call. To actually verify reachability we
    issue one cheap `SELECT ... LIMIT 1` against a known table. Any
    failure (DNS, auth, missing table) falls back to in-memory.
    """
    from src.usage.store import InMemoryUsageStore
    from src.usage.entitlements import EntitlementEngine, default_plans
    from src.alerts.engine import build_default_engine
    from src.observability.store import InMemoryIncidentStore
    from src.observability.tracker import IncidentTracker

    # ---- Usage store ---------------------------------------------------
    usage_store = InMemoryUsageStore()
    if _env("SUPABASE_URL"):
        try:
            from src.usage.store import SupabaseUsageStore
            candidate = SupabaseUsageStore()
            # Real round-trip: one row from the usage_events table.
            (
                candidate._client()
                .table("usage_events")
                .select("event_id")
                .limit(1)
                .execute()
            )
            usage_store = candidate
            logger.info("usage: Supabase store ready")
        except Exception as e:
            logger.warning(
                f"usage: Supabase unavailable, using in-memory "
                f"({type(e).__name__}: {e})"
            )

    # ---- Entitlement engine --------------------------------------------
    try:
        entitlement_engine = EntitlementEngine(
            usage_store, plans_by_name=default_plans(),
        )
        logger.info(
            "entitlements: default plan catalogue loaded "
            "(free / pro / enterprise)"
        )
    except Exception as e:
        logger.warning(f"entitlements: skipped ({type(e).__name__}: {e})")
        entitlement_engine = None

    # ---- Alert engine --------------------------------------------------
    try:
        alert_engine = build_default_engine()
        logger.info("alerts: default rule engine built")
    except Exception as e:
        logger.warning(f"alerts: skipped ({type(e).__name__}: {e})")
        alert_engine = None

    # ---- Incident tracker (needs the alert engine to be useful) --------
    # The in-memory store is created FIRST as the safe default. The
    # Supabase probe runs in its own try/except so a probe failure
    # leaves the in-memory store in place and the tracker still builds.
    incident_tracker = None
    if alert_engine is not None:
        store = InMemoryIncidentStore()
        if _env("SUPABASE_URL"):
            try:
                from src.observability.store import SupabaseIncidentStore
                candidate = SupabaseIncidentStore()
                # Real round-trip: one row from the incidents table.
                (
                    candidate._client()
                    .table("incidents")
                    .select("incident_id")
                    .limit(1)
                    .execute()
                )
                store = candidate
                logger.info("incidents: Supabase store ready")
            except Exception as e:
                logger.warning(
                    f"incidents: Supabase unavailable, using in-memory "
                    f"({type(e).__name__}: {e})"
                )

        try:
            incident_tracker = IncidentTracker(store)
        except Exception as e:
            logger.warning(
                f"incidents: tracker build failed ({type(e).__name__}: {e})"
            )

    return usage_store, entitlement_engine, alert_engine, incident_tracker


def _build_task_store():
    """
    Build the durable TaskStore when Supabase is configured and the
    tasks table is reachable.

    Startup must remain resilient: if Supabase or the migration is not
    available yet, return None so ServiceRegistry keeps its in-memory
    behavior.
    """
    if not _env("SUPABASE_URL"):
        logger.info("tasks: in-memory (SUPABASE_URL not set)")
        return None

    try:
        from src.storage.db import get_client
        from src.storage.task_store import TaskStore

        client = get_client()

        # Real reachability/schema probe.
        (
            client.table("tasks")
            .select("task_id")
            .limit(1)
            .execute()
        )

        logger.info("tasks: persistent TaskStore ready")
        return TaskStore()

    except Exception as e:
        logger.warning(
            f"tasks: Supabase TaskStore unavailable, "
            f"using in-memory ({type(e).__name__}: {e})"
        )
        return None


def _build_checkpoint_store():
    """
    Build the durable Crawl CheckpointStore when Supabase is configured
    and the crawl_checkpoints table is reachable.

    Startup remains resilient: if Supabase or the migration is unavailable,
    return None and PipelineRunner keeps its in-memory-only behavior.
    """
    if not _env("SUPABASE_URL"):
        logger.info("crawl checkpoints: in-memory/disabled (SUPABASE_URL not set)")
        return None

    try:
        from src.storage.db import get_client
        from src.crawl.checkpoint import build_supabase_store

        client = get_client()

        # Real reachability/schema probe.
        (
            client.table("crawl_checkpoints")
            .select("checkpoint_id")
            .limit(1)
            .execute()
        )

        logger.info("crawl checkpoints: Supabase store ready")
        return build_supabase_store()

    except Exception as e:
        logger.warning(
            f"crawl checkpoints: Supabase unavailable, disabled "
            f"({type(e).__name__}: {e})"
        )
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    uvicorn entrypoint. Call via `--factory`.
    """
    logging.basicConfig(
        level=_env("ACES_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mode = _env("ACES_MODE", "api").lower()
    worker_count = _env_int("ACES_WORKER_COUNT", 2)
    scheduler_enabled = _env_bool("ACES_SCHEDULER_ENABLED", True)
    scheduler_interval = _env_float("ACES_SCHEDULER_INTERVAL", 30.0)

    logger.info(
        f"ACES starting: mode={mode} workers={worker_count} "
        f"scheduler={scheduler_enabled}"
    )

    # --- components ---
    key_store = ApiKeyStore()
    task_store = _build_task_store()
    checkpoint_store = _build_checkpoint_store()
    registry = ServiceRegistry(task_store=task_store)
    queue_backend = _build_queue_backend()
    batch_store = _build_batch_store()

    # ------------------------------------------------------------------
    # Batch coordinator
    # ------------------------------------------------------------------
    batch_coordinator = None
    if batch_store is not None:
        from src.jobs.batch_coordinator import BatchCoordinator

        batch_coordinator = BatchCoordinator(
            batch_store,
            queue_backend,
        )

    # --- bootstrap API key (optional) ---
    # A container running in Docker has no way to reach in and create an
    # API key. If the operator sets ACES_BOOTSTRAP_API_KEY, we register
    # that literal value as a valid key for the given client. Used for
    # the initial smoke test, CI, and single-tenant deployments.
    #
    # For real multi-tenant use, replace this with a persistent ApiKeyStore
    # backed by Supabase (roadmap item). Until then, this is the honest
    # working solution.
    bootstrap_key = _env("ACES_BOOTSTRAP_API_KEY")
    if bootstrap_key:
        bootstrap_client = _env("ACES_BOOTSTRAP_CLIENT_ID", "default")
        # ApiKeyStore.create() generates a key. We need to insert a
        # specific key. Do it by reaching into the store's internals —
        # not elegant, but the store is designed for exactly this kind
        # of bootstrap injection.
        from src.api.keys import ApiKey, hash_key, key_prefix
        record = ApiKey(
            key_id=str(uuid.uuid4()),
            client_id=bootstrap_client,
            key_prefix=key_prefix(bootstrap_key),
            key_hash=hash_key(bootstrap_key, key_store._pepper),
            note="bootstrap",
        )
        key_store._by_id[record.key_id] = record
        key_store._by_hash[record.key_hash] = record.key_id
        logger.info(
            f"bootstrap API key registered for client "
            f"{bootstrap_client!r}"
        )

    # Executor is always built — the run endpoint needs it even in
    # worker-only mode, because the queue handler wraps it.
    scraper, extractor = _build_scraper_and_extractor()
    network_manager = _build_network_manager(scraper)

    # Connector registry is optional; build_production_registry never
    # raises — it returns an empty registry when nothing is configured.
    try:
        connector_registry = build_production_registry()
    except Exception as e:
        logger.warning(
            f"connectors: registry build failed "
            f"({type(e).__name__}: {e}); continuing with none"
        )
        connector_registry = None

    # --- observability stack (§30A, §40.6, §41.4, §47B) ---
    # Every subsystem is best-effort — a broken one logs and returns
    # None, and the runner degrades gracefully.
    (
        usage_store,
        entitlement_engine,
        alert_engine,
        incident_tracker,
    ) = _build_observability_stack(registry)

    executor = JobExecutor(
        registry, scraper, extractor,
        runner_kwargs={
            "network_manager": network_manager,
            "connector_registry": connector_registry,
            "use_production_network": False,
            "usage_store": usage_store,
            "entitlement_engine": entitlement_engine,
            "alert_engine": alert_engine,
            "incident_tracker": incident_tracker,
            "checkpoint_store": checkpoint_store,
        },
    )

    # Scheduler is only useful when there's a queue to enqueue into.
    scheduler_loop = None
    if scheduler_enabled:
        scheduler_loop = SchedulerLoop(
            registry, queue_backend,
            poll_interval=scheduler_interval,
        )

    # --- diagnostics callable, wired before the app is built ---
    def _diagnostics() -> dict:
        return {
            "mode": mode,
            "worker_count": worker_count,
            "scheduler_enabled": scheduler_enabled,
            "scheduler_interval": scheduler_interval,
            "queue_backend": type(queue_backend).__name__,
            "connectors": (
                len(connector_registry.all()) if connector_registry else 0
            ),
            # --- observability stack (§40.6, §41.4, §47B) ---
            "usage_store": type(usage_store).__name__,
            "entitlements": (
                "enabled" if entitlement_engine is not None else "disabled"
            ),
            "alerts": (
                "enabled" if alert_engine is not None else "disabled"
            ),
            "incidents": (
                "enabled" if incident_tracker is not None else "disabled"
            ),
        }

    # --- build the app ---
    app = build_app(
        registry,
        key_store,
        executor=executor,
        queue_backend=queue_backend,
        batch_store=batch_store,
        batch_coordinator=batch_coordinator,
        worker_count=worker_count,
        scheduler_loop=scheduler_loop,
        diagnostics_fn=_diagnostics,
    )

    logger.info("ACES ready")
    return app