"""
Job executor — spec §35 + §37.

Bridges the queue to the pipeline. A queue worker calls
`JobExecutor.execute(payload)`; the executor resolves the TaskSpec,
runs PipelineRunner end-to-end, records the outcome, and returns a
small result dict the worker persists via `QueueBackend.complete()`.

Two entry points:

    async execute(payload) -> dict
        Run one job synchronously in the calling task. This is what
        a `QueueWorker` handler calls.

    (see src/api/handlers.py for the HTTP side)

Design notes:

    - The executor is deliberately thin. It does NOT own the queue,
      the registry, or the runner — all are injected. That keeps it
      testable without a browser, an LLM, or a database.
    - On any exception, the executor raises. The worker's `fail()`
      path then applies the retry policy. Executors should never
      swallow errors — the queue's retry logic depends on seeing
      them.
    - Successful runs return a JSON-serializable dict. That dict
      becomes the job's payload under `_result`, available to any
      caller that later reads `GET /v1/jobs/{id}`.
    - Multi-tenant: task lookup is always scoped by `client_id`.
    - Batch jobs may pin a TaskSpec version and provide an exact
      chunk URL list. The stored TaskSpec is never mutated.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from src.api.registry import ServiceRegistry
from src.core.task_spec import TaskSpec
from src.pipeline_runner import PipelineRunner


logger = logging.getLogger("jobs.executor")


# ---------------------------------------------------------------------------
# Sync/async store shim
# ---------------------------------------------------------------------------
# Real BatchStore instances (src.storage.batch_store) are SYNCHRONOUS.
# Test fakes and future async implementations are coroutines. The queue
# handler must work with either, so every store call is routed through
# the shared helper from src.util.async_helpers. The local alias keeps
# every existing `await _maybe_await(...)` call site unchanged.

from src.util.async_helpers import maybe_await as _maybe_await

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """
    What a successful execution returns. Kept minimal — callers that
    need the full PipelineResult can request it via the executor's
    `run_and_return_full` helper.
    """

    task_id: str
    client_id: str
    records_count: int
    quality_passed: bool
    quality_score: float
    confidence_mean: float
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "client_id": self.client_id,
            "records_count": self.records_count,
            "quality_passed": self.quality_passed,
            "quality_score": self.quality_score,
            "confidence_mean": self.confidence_mean,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class JobExecutor:
    """
    Runs one task through the pipeline.

    Constructor:
        registry  — ServiceRegistry used to fetch the TaskSpec
        scraper   — anything the pipeline's network layer needs
        extractor — same
        runner_kwargs — extra kwargs forwarded to PipelineRunner
            (e.g. alert_engine, incident_tracker, connector_registry)

    The scraper/extractor are passed to PipelineRunner at call time,
    not construction time, so tests can inject fakes per-run.
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        scraper,
        extractor,
        *,
        runner_kwargs: Optional[dict] = None,
    ):
        self.registry = registry
        self.scraper = scraper
        self.extractor = extractor
        self.runner_kwargs = dict(runner_kwargs or {})

    def _resolve_task(self, payload: dict) -> TaskSpec:
        """
        Resolve the requested TaskSpec version and optionally bind the
        exact URLs assigned to a batch chunk.

        For batch jobs:
            - `task_version` selects the frozen TaskSpec version.
            - `urls` replaces the copied spec's start_urls.
            - The persisted TaskSpec itself is never modified.
        """
        task_id = str(payload.get("task_id") or "").strip()
        client_id = str(payload.get("client_id") or "default").strip()

        if not task_id:
            raise ValueError("payload missing 'task_id'")

        raw_version = payload.get("task_version")

        if raw_version in (None, ""):
            spec = self.registry.get_task(
                client_id,
                task_id,
            )
        else:
            try:
                task_version = int(raw_version)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "payload field 'task_version' must be an integer"
                ) from e

            if task_version < 1:
                raise ValueError(
                    "payload field 'task_version' must be >= 1"
                )

            spec = self.registry.get_task(
                client_id,
                task_id,
                version=task_version,
            )

        if spec is None:
            raise LookupError(
                f"task {task_id!r} not found for client {client_id!r}"
            )

        # Batch workers must execute exactly the URLs assigned to the chunk.
        # Copy the TaskSpec so the persisted frozen version is never mutated.
        if "urls" in payload:
            urls = payload["urls"]

            if not isinstance(urls, list):
                raise ValueError(
                    "payload field 'urls' must be a list"
                )

            if any(
                not isinstance(url, str) or not url.strip()
                for url in urls
            ):
                raise ValueError(
                    "payload field 'urls' must contain non-empty strings"
                )

            spec = deepcopy(spec)
            spec.target.start_urls = list(urls)

        return spec

    # ------------------------------------------------------------------
    # Sync execution (called by QueueWorker)
    # ------------------------------------------------------------------

    async def execute(self, payload: dict) -> dict:
        """
        Run the job described by `payload`.

        Required keys:

            task_id       — looked up in the registry
            client_id     — tenant scope
            task_version  — optional pinned TaskSpec version
            urls          — optional exact batch-chunk URLs
        """
        task_id = str(payload.get("task_id") or "").strip()
        client_id = str(
            payload.get("client_id") or "default"
        ).strip()

        # `job_id` is set by the enqueuing caller (the HTTP /run handler
        # or the scheduler). It travels with the payload through the queue
        # so the pipeline can attribute usage events, alerts, and
        # incidents to the correct job record.
        job_id = str(payload.get("job_id") or "").strip()

        spec = self._resolve_task(payload)

        # Build the runner fresh per call so the job is isolated.
        runner = PipelineRunner(
            self.scraper,
            self.extractor,
            client_id=client_id,
            job_id=job_id,
            **self.runner_kwargs,
        )

        # Forward optional knobs if the caller provided them.
        run_kwargs = {}

        for key in (
            "output_path",
            "default_currency",
            "default_country",
            "failed_page_count",
        ):
            if key in payload:
                run_kwargs[key] = payload[key]

        result = await runner.run(
            spec,
            **run_kwargs,
        )

        return ExecutionResult(
            task_id=task_id,
            client_id=client_id,
            records_count=len(result.records),
            quality_passed=result.quality_passed,
            quality_score=result.quality_score,
            confidence_mean=result.confidence_mean,
            warnings=list(result.warnings),
        ).to_dict()

    # ------------------------------------------------------------------
    # Full result (for callers that need the whole PipelineResult)
    # ------------------------------------------------------------------

    async def run_and_return_full(self, payload: dict):
        """
        Same as execute() but returns the full PipelineResult. Used by
        synchronous callers (e.g. the UI runner) that want the rich
        object rather than the serialized dict.
        """
        client_id = str(
            payload.get("client_id") or "default"
        ).strip()
        job_id = str(payload.get("job_id") or "").strip()

        spec = self._resolve_task(payload)

        runner = PipelineRunner(
            self.scraper,
            self.extractor,
            client_id=client_id,
            job_id=job_id,
            **self.runner_kwargs,
        )

        run_kwargs = {}

        for key in (
            "output_path",
            "default_currency",
            "default_country",
            "failed_page_count",
        ):
            if key in payload:
                run_kwargs[key] = payload[key]

        return await runner.run(
            spec,
            **run_kwargs,
        )


# ---------------------------------------------------------------------------
# Queue handler factory
# ---------------------------------------------------------------------------

def make_queue_handler(
    executor: JobExecutor,
    registry: ServiceRegistry,
    batch_store=None,
):
    """
    Build a queue handler that runs the executor and updates the job
    record's state as it progresses.

    When a batch_store is supplied, batch/chunk state is also persisted:

        QUEUED -> RUNNING -> SUCCEEDED

    and failures use Batch.fail_chunk(), which preserves the batch
    retry semantics.

    Existing non-batch jobs continue to work exactly as before.
    """

    async def handler(payload: dict) -> dict:
        task_id = str(payload.get("task_id") or "")
        client_id = str(
            payload.get("client_id") or "default"
        )
        job_id = str(payload.get("job_id") or "")

        batch_id = str(payload.get("batch_id") or "")
        chunk_id = str(payload.get("chunk_id") or "")

        # --------------------------------------------------------------
        # Batch/chunk pre-flight
        # --------------------------------------------------------------
        batch = None
        chunk = None

        if batch_store is not None and batch_id and chunk_id:
            batch = await _maybe_await(
                batch_store.get(client_id, batch_id)
            )

            if batch is None:
                raise LookupError(
                    f"batch {batch_id!r} not found "
                    f"for client {client_id!r}"
                )

            chunk = next(
                (
                    c
                    for c in batch.chunks
                    if c.chunk_id == chunk_id
                ),
                None,
            )

            if chunk is None:
                raise LookupError(
                    f"chunk {chunk_id!r} not found "
                    f"in batch {batch_id!r}"
                )

            # A completed/cancelled/paused chunk must not execute again.
            if chunk.state.value in (
                "succeeded",
                "cancelled",
                "paused",
            ):
                return {
                    "batch_id": batch_id,
                    "chunk_id": chunk_id,
                    "state": chunk.state.value,
                    "skipped": True,
                }

            # A normal first attempt or queue retry starts the chunk.
            # If recovery finds the chunk already RUNNING, keep it RUNNING
            # and allow the queue retry to continue safely.
            if chunk.state.value in (
                "queued",
                "failed",
            ):
                chunk.mark_running()
                batch._refresh_counts()
                await _maybe_await(batch_store.save(batch))

        # --------------------------------------------------------------
        # Normal queue job state
        # --------------------------------------------------------------
        if job_id:
            job = registry.get_job(
                client_id,
                job_id,
            )
            if job is not None:
                job["state"] = "running"

        # Batch chunks need the actual records persisted into the chunk
        # so the merged dataset can be reconstructed later. Single jobs
        # keep the lightweight `execute()` path (no records in response).
        is_batch_chunk = (
            batch is not None and chunk is not None
        )

        try:
            if is_batch_chunk:
                full = await executor.run_and_return_full(payload)
                result = {
                    "task_id": task_id,
                    "client_id": client_id,
                    "records_count": len(full.records),
                    "records": list(full.records),
                    "quality_passed": full.quality_passed,
                    "quality_score": full.quality_score,
                    "confidence_mean": full.confidence_mean,
                    "warnings": list(full.warnings),
                }
            else:
                result = await executor.execute(payload)

        except Exception as e:
            # Persist batch failure/retry state before re-raising so the
            # queue can apply its own retry policy.
            if batch is not None and chunk is not None:
                error_message = (
                    f"{type(e).__name__}: {e}"
                )
                batch.fail_chunk(
                    chunk_id,
                    error=error_message,
                )
                await _maybe_await(batch_store.save(batch))

            if job_id:
                job = registry.get_job(
                    client_id,
                    job_id,
                )
                if job is not None:
                    job["state"] = "failed"
                    job["error"] = (
                        f"{type(e).__name__}: {e}"
                    )

            raise

        # --------------------------------------------------------------
        # Successful completion
        # --------------------------------------------------------------
        if batch is not None and chunk is not None:
            if chunk.state.value == "running":
                records_count = int(
                    result.get("records_count", 0)
                )
                # Prefer the explicit `records` list on the result;
                # fall back to an empty list so old executors that only
                # return a count still work.
                records = result.get("records")
                if not isinstance(records, list):
                    records = []
                chunk.mark_succeeded(
                    records_count=records_count,
                    records=records,
                )
                batch._refresh_counts()
                await _maybe_await(batch_store.save(batch))

        if job_id:
            job = registry.get_job(
                client_id,
                job_id,
            )
            if job is not None:
                job["state"] = "completed"
                job["result"] = result

        return result

    return handler


# ---------------------------------------------------------------------------
# Smoke test — in-memory fakes, no network, no LLM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    from src.api.registry import ServiceRegistry
    from src.core.task_spec import (
        FieldSpec,
        Target,
        TaskSpec,
    )

    class FakeScraper:
        def __init__(self, html="<html>SENTINEL</html>"):
            self.html = html

        async def fetch_html(self, url, timeout=None):
            return self.html

        async def fetch_screenshot(self, url, timeout=None):
            return b"png"

    class FakeExtractor:
        def __init__(self, items=None):
            self.items = items if items is not None else [
                {"title": "A", "price": "$1"},
                {"title": "B", "price": "$2"},
            ]

        def extract_list(self, html, instruction):
            return list(self.items)

        def extract_from_image(self, img, instr):
            return []

    async def run():
        registry = ServiceRegistry()

        # Create a task under 'acme'
        spec = TaskSpec(
            natural_language_prompt="find laptop prices",
            target=Target(
                start_urls=["https://93.184.216.34/a"]
            ),
            fields=[
                FieldSpec(name="title"),
                FieldSpec(name="price"),
            ],
        )
        spec.client_id = "acme"
        registry.save_task("acme", spec)

        # ---- 1. Happy path ----
        ex = JobExecutor(
            registry,
            FakeScraper(),
            FakeExtractor(),
        )

        result = await ex.execute({
            "task_id": spec.task_id,
            "client_id": "acme",
        })

        assert result["task_id"] == spec.task_id
        assert result["client_id"] == "acme"
        assert result["records_count"] == 2
        assert isinstance(result["quality_passed"], bool)
        assert isinstance(result["warnings"], list)

        # ---- 2. Missing task_id → ValueError ----
        try:
            await ex.execute({
                "client_id": "acme",
            })
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

        # ---- 3. Unknown task → LookupError ----
        try:
            await ex.execute({
                "task_id": "does-not-exist",
                "client_id": "acme",
            })
            raise AssertionError("expected LookupError")
        except LookupError:
            pass

        # ---- 4. Cross-tenant cannot fetch the task ----
        try:
            await ex.execute({
                "task_id": spec.task_id,
                "client_id": "other",
            })
            raise AssertionError(
                "expected LookupError for wrong tenant"
            )
        except LookupError:
            pass

        # ---- 5. run_and_return_full returns PipelineResult ----
        full = await ex.run_and_return_full({
            "task_id": spec.task_id,
            "client_id": "acme",
        })

        assert hasattr(full, "records")
        assert len(full.records) == 2

        # ---- 6. Empty extractor → records_count == 0 ----
        ex_empty = JobExecutor(
            registry,
            FakeScraper(),
            FakeExtractor(items=[]),
        )

        result = await ex_empty.execute({
            "task_id": spec.task_id,
            "client_id": "acme",
        })

        assert result["records_count"] == 0

        print("JobExecutor OK.")

    asyncio.run(run())