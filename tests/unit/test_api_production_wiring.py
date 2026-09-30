"""
Unit tests for production observability wiring (spec §40.6, §41.4, §47B).

Covers:
    - JobExecutor forwards `job_id` from the queue payload → PipelineRunner
    - `_build_observability_stack()` returns non-None pieces with no
      Supabase configured (in-memory fallbacks)
    - `_build_observability_stack()` returns a Supabase-backed usage
      store when SUPABASE_URL is set and the client probes successfully
    - A subsystem failure (mocked) is isolated — other subsystems still
      build
    - The stack is properly threaded through create_app()'s executor
      (tested via a lightweight app factory that doesn't boot Playwright)
"""
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.keys import ApiKeyStore
from src.api.registry import ServiceRegistry
from src.core.task_spec import FieldSpec, Target, TaskSpec
from src.jobs.executor import JobExecutor


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# Task 1: JobExecutor forwards job_id
# ===========================================================================

class _SpyRunner:
    """Records every constructor call so we can inspect kwargs."""
    instances: list["_SpyRunner"] = []

    def __init__(self, scraper, extractor, **kwargs):
        self.scraper = scraper
        self.extractor = extractor
        self.kwargs = kwargs
        self.runs: list[dict] = []
        _SpyRunner.instances.append(self)

    async def run(self, spec, **kwargs):
        self.runs.append({"spec": spec, **kwargs})
        # Return the minimum shape PipelineRunner.run returns.
        from src.pipeline_runner import PipelineResult
        return PipelineResult(
            task_id=spec.task_id,
            records=[],
            quality_passed=True,
            quality_score=1.0,
            publication_decision=None,
            change_set_summary={"new": 0, "modified": 0, "removed": 0,
                                 "unchanged": 0, "total": 0},
            workbook=None,
            receipt_signature=None,
        )


@pytest.fixture
def registry_with_task():
    reg = ServiceRegistry()
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.client_id = "acme"
    reg.save_task("acme", spec)
    return reg, spec.task_id

def test_executor_uses_frozen_version_and_exact_chunk_urls(
    monkeypatch, registry_with_task,
):
    reg, task_id = registry_with_task

    monkeypatch.setattr(
        "src.jobs.executor.PipelineRunner",
        _SpyRunner,
    )
    _SpyRunner.instances.clear()

    executor = JobExecutor(
        reg,
        scraper=object(),
        extractor=object(),
    )

    chunk_urls = [
        "https://example.com/products/1",
        "https://example.com/products/2",
    ]

    _run(executor.execute({
        "task_id": task_id,
        "client_id": "acme",
        "task_version": 1,
        "batch_id": "batch-1",
        "chunk_id": "chunk-1",
        "urls": chunk_urls,
    }))

    instance = _SpyRunner.instances[0]
    run_spec = instance.runs[0]["spec"]

    assert run_spec.target.start_urls == chunk_urls
    assert run_spec.target.start_urls is not chunk_urls

    stored_spec = reg.get_task(
        "acme",
        task_id,
        version=1,
    )
    assert stored_spec.target.start_urls == [
        "https://93.184.216.34/a",
    ]


def test_executor_rejects_invalid_chunk_urls(
    monkeypatch, registry_with_task,
):
    reg, task_id = registry_with_task

    monkeypatch.setattr(
        "src.jobs.executor.PipelineRunner",
        _SpyRunner,
    )

    executor = JobExecutor(
        reg,
        scraper=object(),
        extractor=object(),
    )

    with pytest.raises(ValueError, match="urls"):
        _run(executor.execute({
            "task_id": task_id,
            "client_id": "acme",
            "task_version": 1,
            "urls": [
                "https://example.com/a",
                "",
            ],
        }))


def test_executor_forwards_job_id_to_runner(monkeypatch, registry_with_task):
    """Regression: the queue payload's job_id must reach PipelineRunner."""
    reg, task_id = registry_with_task

    # Patch PipelineRunner where executor.py imports it.
    monkeypatch.setattr("src.jobs.executor.PipelineRunner", _SpyRunner)
    _SpyRunner.instances.clear()

    executor = JobExecutor(reg, scraper=object(), extractor=object())
    _run(executor.execute({
        "task_id": task_id,
        "client_id": "acme",
        "job_id": "job-42",
    }))

    assert len(_SpyRunner.instances) == 1
    inst = _SpyRunner.instances[0]
    assert inst.kwargs.get("job_id") == "job-42"
    assert inst.kwargs.get("client_id") == "acme"


def test_executor_forwards_job_id_in_run_and_return_full(
    monkeypatch, registry_with_task,
):
    reg, task_id = registry_with_task
    monkeypatch.setattr("src.jobs.executor.PipelineRunner", _SpyRunner)
    _SpyRunner.instances.clear()

    executor = JobExecutor(reg, scraper=object(), extractor=object())
    _run(executor.run_and_return_full({
        "task_id": task_id,
        "client_id": "acme",
        "job_id": "job-99",
    }))

    assert _SpyRunner.instances[0].kwargs.get("job_id") == "job-99"


def test_executor_empty_job_id_is_fine(monkeypatch, registry_with_task):
    """Missing job_id (sync caller with no queue) must still work."""
    reg, task_id = registry_with_task
    monkeypatch.setattr("src.jobs.executor.PipelineRunner", _SpyRunner)
    _SpyRunner.instances.clear()

    executor = JobExecutor(reg, scraper=object(), extractor=object())
    _run(executor.execute({"task_id": task_id, "client_id": "acme"}))

    assert _SpyRunner.instances[0].kwargs.get("job_id") == ""


def test_executor_passes_runner_kwargs_through(monkeypatch, registry_with_task):
    """Anything in runner_kwargs must survive to PipelineRunner."""
    reg, task_id = registry_with_task
    monkeypatch.setattr("src.jobs.executor.PipelineRunner", _SpyRunner)
    _SpyRunner.instances.clear()

    sentinel = object()
    executor = JobExecutor(
        reg, scraper=object(), extractor=object(),
        runner_kwargs={"usage_store": sentinel, "network_manager": "nm"},
    )
    _run(executor.execute({
        "task_id": task_id, "client_id": "acme", "job_id": "j-1",
    }))

    kwargs = _SpyRunner.instances[0].kwargs
    assert kwargs["usage_store"] is sentinel
    assert kwargs["network_manager"] == "nm"
    # job_id still wins over any absent value
    assert kwargs["job_id"] == "j-1"


# ===========================================================================
# Task 2: _build_observability_stack()
# ===========================================================================

def _import_builder():
    """
    Import the helper in isolation. Importing src.api.production is safe
    at module load — its heavy dependencies (Playwright, SeleniumBase)
    are all lazy.
    """
    from src.api.production import _build_observability_stack
    return _build_observability_stack


def test_stack_builds_with_no_supabase(monkeypatch):
    """No SUPABASE_URL → all in-memory fallbacks, all non-None."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    builder = _import_builder()

    usage_store, entitlements, alerts, incidents = builder(
        ServiceRegistry()
    )

    assert usage_store is not None
    assert type(usage_store).__name__ == "InMemoryUsageStore"

    assert entitlements is not None
    assert type(entitlements).__name__ == "EntitlementEngine"

    assert alerts is not None
    assert type(alerts).__name__ == "AlertEngine"

    assert incidents is not None
    assert type(incidents).__name__ == "IncidentTracker"


def test_stack_builds_with_supabase_env_but_unreachable(monkeypatch):
    """
    A SUPABASE_URL set but unreachable (e.g. bad network) must not
    crash the builder — it should fall back to in-memory.
    """
    monkeypatch.setenv("SUPABASE_URL", "https://nonexistent.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-key")
    builder = _import_builder()

    usage_store, entitlements, alerts, incidents = builder(
        ServiceRegistry()
    )
    # Fallback to in-memory because the probe failed.
    assert type(usage_store).__name__ == "InMemoryUsageStore"
    assert entitlements is not None
    assert alerts is not None
    assert incidents is not None


def test_stack_survives_alert_engine_failure(monkeypatch):
    """If build_default_engine() raises, the rest still builds."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)

    import src.alerts.engine as alerts_mod
    monkeypatch.setattr(
        alerts_mod, "build_default_engine",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    builder = _import_builder()
    usage_store, entitlements, alerts, incidents = builder(
        ServiceRegistry()
    )

    assert usage_store is not None
    assert entitlements is not None
    assert alerts is None              # failed, degraded to None
    assert incidents is None           # depends on alerts


def test_stack_survives_entitlement_failure(monkeypatch):
    """If the entitlement engine fails, the rest still builds."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)

    import src.usage.entitlements as ent_mod
    monkeypatch.setattr(
        ent_mod, "EntitlementEngine",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    builder = _import_builder()
    usage_store, entitlements, alerts, incidents = builder(
        ServiceRegistry()
    )

    assert usage_store is not None
    assert entitlements is None
    assert alerts is not None
    assert incidents is not None


# ===========================================================================
# Wire-through: the executor receives the observability kwargs
# ===========================================================================

def test_executor_receives_observability_kwargs_from_production(monkeypatch):
    """
    Build the observability stack the same way create_app() does, hand it
    to a JobExecutor, and verify PipelineRunner would receive each piece.
    """
    monkeypatch.delenv("SUPABASE_URL", raising=False)

    from src.usage.store import InMemoryUsageStore

    usage_store = InMemoryUsageStore()
    sentinel_ent = object()
    sentinel_alerts = object()
    sentinel_incidents = object()

    executor = JobExecutor(
        ServiceRegistry(), scraper=object(), extractor=object(),
        runner_kwargs={
            "usage_store": usage_store,
            "entitlement_engine": sentinel_ent,
            "alert_engine": sentinel_alerts,
            "incident_tracker": sentinel_incidents,
        },
    )

    # The kwargs must be preserved on the executor itself.
    rk = executor.runner_kwargs
    assert rk["usage_store"] is usage_store
    assert rk["entitlement_engine"] is sentinel_ent
    assert rk["alert_engine"] is sentinel_alerts
    assert rk["incident_tracker"] is sentinel_incidents


# ===========================================================================
# /_info endpoint exposes the new diagnostics
# ===========================================================================

def test_info_endpoint_reports_observability(monkeypatch):
    """
    A container booted with the observability stack must report it in
    /_info. We drive a real app through httpx but bypass the actual
    Playwright executor to keep the test fast and offline.
    """
    monkeypatch.delenv("SUPABASE_URL", raising=False)

    from src.api.server import build_app

    ks = ApiKeyStore()
    reg = ServiceRegistry()

    # Stand in for the observability dict the factory would produce.
    diagnostics = {
        "mode": "api",
        "worker_count": 1,
        "scheduler_enabled": False,
        "scheduler_interval": 30.0,
        "queue_backend": "InMemoryQueueBackend",
        "connectors": 0,
        "usage_store": "InMemoryUsageStore",
        "entitlements": "enabled",
        "alerts": "enabled",
        "incidents": "enabled",
    }

    app = build_app(
        reg, ks,
        diagnostics_fn=lambda: diagnostics,
    )

    async def scenario():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            return await client.get("/_info")

    r = _run(scenario())
    assert r.status_code == 200
    body = r.json()
    assert body["usage_store"] == "InMemoryUsageStore"
    assert body["entitlements"] == "enabled"
    assert body["alerts"] == "enabled"
    assert body["incidents"] == "enabled"

# ===========================================================================
# Durable task-store wiring
# ===========================================================================

def test_task_store_returns_none_without_supabase(monkeypatch):
    from src.api.production import _build_task_store

    monkeypatch.delenv("SUPABASE_URL", raising=False)

    assert _build_task_store() is None


def test_task_store_falls_back_when_probe_fails(monkeypatch):
    import src.storage.db as db_mod

    from src.api.production import _build_task_store

    class BrokenTable:
        def select(self, *_args):
            return self

        def limit(self, *_args):
            return self

        def execute(self):
            raise RuntimeError("database unavailable")

    class BrokenClient:
        def table(self, _name):
            return BrokenTable()

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(db_mod, "get_client", lambda: BrokenClient())

    assert _build_task_store() is None


def test_task_store_builds_when_probe_succeeds(monkeypatch):
    import src.storage.db as db_mod
    import src.storage.task_store as task_store_mod

    from src.api.production import _build_task_store

    class GoodTable:
        def select(self, *_args):
            return self

        def limit(self, *_args):
            return self

        def execute(self):
            return type("Response", (), {"data": []})()

    class GoodClient:
        def table(self, _name):
            return GoodTable()

    sentinel = object()

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(db_mod, "get_client", lambda: GoodClient())
    monkeypatch.setattr(task_store_mod, "TaskStore", lambda: sentinel)

    assert _build_task_store() is sentinel

def test_executor_uses_pinned_task_version(monkeypatch):
    class FakeTaskStore:
        def __init__(self, versioned_spec):
            self.versioned_spec = versioned_spec
            self.requested_version = None

        def get_task(self, client_id, task_id, version=None):
            self.requested_version = version
            return self.versioned_spec

    from src.core.task_spec import TaskSpec, Target

    pinned_spec = TaskSpec(
        natural_language_prompt="version 2 task",
        target=Target(
            start_urls=["https://version-2.example"],
        ),
    )
    pinned_spec.task_id = "task-version-test"
    pinned_spec.client_id = "acme"

    task_store = FakeTaskStore(pinned_spec)
    registry = ServiceRegistry(task_store=task_store)

    monkeypatch.setattr(
        "src.jobs.executor.PipelineRunner",
        _SpyRunner,
    )
    _SpyRunner.instances.clear()

    executor = JobExecutor(
        registry,
        scraper=object(),
        extractor=object(),
    )

    _run(
        executor.execute(
            {
                "task_id": "task-version-test",
                "client_id": "acme",
                "job_id": "job-version-2",
                "task_version": 2,
            }
        )
    )

    assert task_store.requested_version == 2
    assert len(_SpyRunner.instances) == 1
    assert _SpyRunner.instances[0].runs[0]["spec"] is pinned_spec

def test_build_checkpoint_store_without_supabase_returns_none(monkeypatch):
    from src.api.production import _build_checkpoint_store

    monkeypatch.delenv("SUPABASE_URL", raising=False)

    assert _build_checkpoint_store() is None


def test_build_checkpoint_store_with_broken_probe_returns_none(monkeypatch):
    from src.api.production import _build_checkpoint_store

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-key")

    class BrokenClient:
        def table(self, name):
            assert name == "crawl_checkpoints"
            raise RuntimeError("probe failed")

    monkeypatch.setattr(
        "src.storage.db.get_client",
        lambda: BrokenClient(),
    )

    assert _build_checkpoint_store() is None


def test_executor_receives_checkpoint_store():
    sentinel = object()

    executor = JobExecutor(
        ServiceRegistry(),
        scraper=object(),
        extractor=object(),
        runner_kwargs={"checkpoint_store": sentinel},
    )

    assert executor.runner_kwargs["checkpoint_store"] is sentinel