"""
HTTP server — spec §45.

Binds the framework-agnostic internal router to FastAPI so the API is
reachable over real HTTP. Everything underneath (handlers, middleware,
auth, error shapes, rate limiting, body size limits) is reused as-is —
this module is a thin translation layer.

Design:

    The internal router has a single entry point:
        internal_router.dispatch(internal_Request) -> internal_Response

    FastAPI hands us an HTTP request. We:

        1. Wrap it into an internal Request (method, path, headers, body)
        2. Call dispatch() — auth, rate-limit, and body-size middleware
           all run inside that call, unchanged
        3. Translate the internal Response back to HTTP

    That means the API behaves *identically* whether it is called from
    Python or over the wire. There is exactly one place where auth is
    enforced. There is exactly one place where 404s are produced.

Why a catch-all route and not one FastAPI route per internal route:

    The internal router already has an efficient path-matching regex.
    Re-registering each route on FastAPI would either duplicate that
    logic (bug risk) or require dynamic signature generation (fragile).
    A catch-all keeps the two systems decoupled: FastAPI owns transport,
    the internal router owns semantics.

Trade-off of that choice:

    FastAPI cannot auto-generate per-endpoint OpenAPI schemas because
    it only sees one route. The `/docs` page still works; it just shows
    a generic "call any ACES endpoint" interface. Full OpenAPI schemas
    are a follow-up (see the roadmap). Do not remove the /docs endpoint
    though — it remains useful for interactive exploration.

Testing:

    `build_app()` returns a plain ASGI app, so tests can drive it with
    httpx.ASGITransport — no network, no subprocess. The `__main__`
    smoke test does exactly that.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse, Response as FastAPIResponse

from src.api.handlers import build_router as build_internal_router
from src.api.keys import ApiKeyStore
from src.api.registry import ServiceRegistry
from src.api.router import (
    Router as InternalRouter,
    auth_middleware,
    body_size_limit_middleware,
    rate_limit_middleware,
)
from src.api.schemas import Request as InternalRequest
from src.queue.types import QueuedJob
from src.util.async_helpers import maybe_await


logger = logging.getLogger("api.server")

# ---------------------------------------------------------------------------
# Async fast-path patterns
# ---------------------------------------------------------------------------
# Endpoints that must run inside the FastAPI event loop (they `await`
# the JobExecutor / BatchCoordinator) are matched here, bypassing the
# sync internal router.
_RUN_RE = re.compile(r"^/v1/tasks/(?P<id>[^/]+)/run$")
_BATCH_CREATE_RE = re.compile(r"^/v1/batches$")
_BATCH_GET_RE = re.compile(r"^/v1/batches/(?P<id>[^/]+)$")
_BATCH_CONTROL_RE = re.compile(
r"^/v1/batches/(?P<id>[^/]+)/(?P<action>pause|resume|cancel)$")
_BATCH_RESULT_RE = re.compile(
    r"^/v1/batches/(?P<id>[^/]+)/result$"
)
# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app(
    registry: ServiceRegistry,
    key_store: ApiKeyStore,
    *,
    internal_router: Optional[InternalRouter] = None,
    executor=None,
    queue_backend=None,
    batch_store=None,
    batch_coordinator=None,
    worker_pool=None,
    worker_count: int = 2,
    scheduler_loop=None,
    diagnostics_fn=None,
    rate_limit_per_minute: int = 120,
    max_body_bytes: int = 1_000_000,
) -> FastAPI:
    """
    Build a fully-wired ASGI app.

    Run mode is picked by what is supplied:

        worker_pool supplied      → POST /run enqueues; a background
                                    worker pool consumes and runs.

        queue_backend supplied    → POST /run enqueues; a WorkerPool is
                                    built automatically.

        executor only             → POST /run runs synchronously.

        neither                   → POST /run returns 503.

    The pool, when built, is started on app lifespan startup and
    stopped on shutdown. ASGI test clients (httpx.ASGITransport) do
    NOT run lifespans by default, so tests that want to see queued
    jobs complete must either start the pool manually via
    `app.state.worker_pool.start()` or supply an already-started pool.

    `scheduler_loop` is optional. When supplied, it is started and
    stopped alongside the pool. It is NOT auto-built from other args —
    scheduling is a deliberate opt-in so that single-shot callers
    don't get a background firing loop by accident.
    """

    from src.jobs.executor import make_queue_handler
    from src.queue.lifecycle import WorkerPool

    # ------------------------------------------------------------------
    # Internal router, with middleware already attached
    # ------------------------------------------------------------------
    if internal_router is None:
        internal_router = build_internal_router(registry)

    internal_router.add_middleware(auth_middleware(key_store))
    internal_router.add_middleware(
        rate_limit_middleware(per_minute=rate_limit_per_minute),
    )
    internal_router.add_middleware(
        body_size_limit_middleware(max_bytes=max_body_bytes),
    )

    # ------------------------------------------------------------------
    # Worker pool (if a queue is in play)
    # ------------------------------------------------------------------
    pool = worker_pool

    if pool is None and queue_backend is not None and executor is not None:
        handler = make_queue_handler(
            executor,
            registry,
            batch_store=batch_store,
        )

        pool = WorkerPool(
            queue_backend,
            handler,
            worker_count=worker_count,
        )

    # ------------------------------------------------------------------
    # Lifespan: start/stop pool + scheduler
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        if pool is not None:
            await pool.start()

        if scheduler_loop is not None:
            await scheduler_loop.start()

        try:
            yield
        finally:
            if scheduler_loop is not None:
                await scheduler_loop.stop()

            if pool is not None:
                await pool.stop()

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------
    app = FastAPI(
        title="ACES",
        description="Autonomous Cognitive Extraction System — HTTP API",
        version="1.0.0",
        lifespan=_lifespan,
    )

    # ------------------------------------------------------------------
    # App state
    # ------------------------------------------------------------------
    app.state.internal_router = internal_router
    app.state.registry = registry
    app.state.key_store = key_store
    app.state.executor = executor
    app.state.queue_backend = queue_backend
    app.state.batch_store = batch_store
    app.state.batch_coordinator = batch_coordinator
    app.state.worker_pool = pool
    app.state.scheduler_loop = scheduler_loop

    # ------------------------------------------------------------------
    # Root handler FIRST so it wins over the catch-all
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def _root():
        return {
            "name": "ACES API",
            "version": "1.0.0",
            "docs": "/docs",
            "hint": "All API routes live under /v1/*",
        }

    # ------------------------------------------------------------------
    # Diagnostics endpoint
    # ------------------------------------------------------------------
    if diagnostics_fn is not None:

        @app.get("/_info", include_in_schema=False)
        async def _info():
            return diagnostics_fn()

    # ------------------------------------------------------------------
    # Catch-all route: every method, every path
    # ------------------------------------------------------------------
    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    async def _catch_all(
        request: FastAPIRequest,
        full_path: str,
    ):
        return await _dispatch(request)

    return app


# ---------------------------------------------------------------------------
# Dispatch bridge
# ---------------------------------------------------------------------------

async def _dispatch(
    request: FastAPIRequest,
) -> FastAPIResponse:
    """
    Translate a FastAPI request → internal Request → internal Response,
    then back to HTTP.

    Async endpoints are handled directly here because they need to
    await injected async services. Everything else flows through the
    sync internal router with its middleware chain.
    """

    internal_router: InternalRouter = request.app.state.internal_router
    # Normalize path — FastAPI strips the leading slash in path_params
    full_path = request.path_params.get("full_path") or ""
    path = "/" + full_path.lstrip("/")
    m = _RUN_RE.match(path)
    ...
    if _BATCH_CREATE_RE.match(path) and request.method == "POST":
        return await _handle_batch_create(request)
    ...
    m = _BATCH_GET_RE.match(path)
    if m and request.method == "GET":
        return await _handle_batch_get(request, batch_id=m.group("id"))

    # ------------------------------------------------------------------
    # Async fast-path: GET /v1/batches/{id}/result
    # ------------------------------------------------------------------
    m = _BATCH_RESULT_RE.match(path)

    if m and request.method == "GET":
        return await _handle_batch_result(
            request,
            batch_id=m.group("id"),
        )
    # ------------------------------------------------------------------
    # Async fast-path: POST /v1/tasks/{id}/run
    # ------------------------------------------------------------------
    m = _RUN_RE.match(path)

    if m and request.method == "POST":
        return await _handle_run(
            request,
            task_id=m.group("id"),
            key_store=request.app.state.key_store,
            registry=request.app.state.registry,
            executor=request.app.state.executor,
        )

    # ------------------------------------------------------------------
    # Async fast-path: POST /v1/batches
    # ------------------------------------------------------------------
    if _BATCH_CREATE_RE.match(path) and request.method == "POST":
        return await _handle_batch_create(request)

    # ------------------------------------------------------------------
    # Async fast-path: GET /v1/batches/{id}
    # ------------------------------------------------------------------
    m = _BATCH_GET_RE.match(path)

    if m and request.method == "GET":
        return await _handle_batch_get(
            request,
            batch_id=m.group("id"),
        )
    # ------------------------------------------------------------------
    # Async fast-path: POST /v1/batches/{id}/(pause|resume|cancel)
    # ------------------------------------------------------------------
    m = _BATCH_CONTROL_RE.match(path)

    if m and request.method == "POST":
        return await _handle_batch_control(
            request,
            batch_id=m.group("id"),
            action=m.group("action"),
        )
    # ------------------------------------------------------------------
    # Body — only read when there is one
    # ------------------------------------------------------------------
    body = None

    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body = await request.json()
        except Exception:
            body = None

    # ------------------------------------------------------------------
    # Convert FastAPI request → internal request
    # ------------------------------------------------------------------
    internal_request = InternalRequest(
        method=request.method,
        path=path,
        headers=dict(request.headers),
        body=body,
    )

    # ------------------------------------------------------------------
    # Internal router owns normal API semantics
    # ------------------------------------------------------------------
    internal_response = internal_router.dispatch(internal_request)

    # ------------------------------------------------------------------
    # Convert internal response → FastAPI response
    # ------------------------------------------------------------------
    headers = dict(internal_response.headers or {})

    if internal_response.body is None:
        return FastAPIResponse(
            status_code=internal_response.status,
            headers=headers,
        )

    return JSONResponse(
        status_code=internal_response.status,
        content=internal_response.body,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Convenience: uvicorn runner
# ---------------------------------------------------------------------------

def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    registry: Optional[ServiceRegistry] = None,
    key_store: Optional[ApiKeyStore] = None,
    **kwargs,
) -> None:
    """
    Run the app under uvicorn.

    Intended for local development / manual smoke tests.
    """

    import uvicorn

    registry = registry or ServiceRegistry()
    key_store = key_store or ApiKeyStore()

    app = build_app(
        registry,
        key_store,
        **kwargs,
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
    )


# ---------------------------------------------------------------------------
# Async route handler: run a task
# ---------------------------------------------------------------------------

async def _handle_run(
    request: FastAPIRequest,
    *,
    task_id: str,
    key_store: ApiKeyStore,
    registry: ServiceRegistry,
    executor,
) -> FastAPIResponse:
    """
    POST /v1/tasks/{id}/run

    Two modes:

        Async (queue_backend configured):
            Creates a `queued` job record, enqueues a payload, returns
            immediately with 202. The worker pool consumes it and flips
            the record's state as it progresses.

        Sync (executor only):
            Runs the pipeline inline and returns 200 with a
            `completed` record.
    """

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    raw = request.headers.get("authorization", "")

    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()

    if not raw:
        raw = request.headers.get("x-api-key", "")

    record = key_store.verify(raw) if raw else None

    if record is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "unauthorized",
                    "message": "missing or invalid API key",
                },
            },
        )

    client_id = record.client_id

    # ------------------------------------------------------------------
    # Executor must be configured
    # ------------------------------------------------------------------
    if executor is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "executor_unavailable",
                    "message": "no job executor is configured",
                },
            },
        )

    # ------------------------------------------------------------------
    # Task must exist and belong to this client
    # ------------------------------------------------------------------
    spec = registry.get_task(
        client_id,
        task_id,
    )

    if spec is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "not_found",
                    "message": f"task {task_id} not found",
                },
            },
        )

    # ------------------------------------------------------------------
    # Optional body
    # Only forward explicitly supported run-time options.
    # ------------------------------------------------------------------
    body = {}

    try:
        parsed = await request.json()

        if isinstance(parsed, dict):
            body = parsed

    except Exception:
        pass

    allowed_keys = (
        "output_path",
        "default_currency",
        "default_country",
        "failed_page_count",
    )

    forwarded = {
        key: value
        for key, value in body.items()
        if key in allowed_keys
    }

    # ------------------------------------------------------------------
    # Async path: enqueue and return immediately
    # ------------------------------------------------------------------
    queue_backend = request.app.state.queue_backend

    if queue_backend is not None:
        job_id = str(uuid.uuid4())

        registry.save_job(
            client_id,
            job_id,
            {
                "job_id": job_id,
                "task_id": task_id,
                "client_id": client_id,
                "state": "queued",
            },
        )

        payload = {
            "task_id": task_id,
            "client_id": client_id,
            "job_id": job_id,
            **forwarded,
        }

        try:
            await queue_backend.enqueue(
                QueuedJob(
                    job_id=job_id,
                    payload=payload,
                    client_id=client_id,
                    label=f"task:{task_id}",
                )
            )

        except Exception as e:
            registry.save_job(
                client_id,
                job_id,
                {
                    "job_id": job_id,
                    "task_id": task_id,
                    "client_id": client_id,
                    "state": "failed",
                    "error": (
                        f"enqueue failed: "
                        f"{type(e).__name__}: {e}"
                    ),
                },
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "enqueue_failed",
                        "message": (
                            f"{type(e).__name__}: {e}"
                        ),
                    },
                },
            )

        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "task_id": task_id,
                "client_id": client_id,
                "state": "queued",
            },
        )

    # ------------------------------------------------------------------
    # Sync path: run inline
    # ------------------------------------------------------------------
    payload = {
        "task_id": task_id,
        "client_id": client_id,
        **forwarded,
    }

    try:
        result = await executor.execute(payload)

    except LookupError as e:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "not_found",
                    "message": str(e),
                },
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "run_failed",
                    "message": (
                        f"{type(e).__name__}: {e}"
                    ),
                },
            },
        )

    # ------------------------------------------------------------------
    # Persist completed job
    # ------------------------------------------------------------------
    job_id = str(uuid.uuid4())

    registry.save_job(
        client_id,
        job_id,
        {
            "job_id": job_id,
            "task_id": task_id,
            "client_id": client_id,
            "state": "completed",
            "result": result,
        },
    )

    response_body = {
        **result,
        "job_id": job_id,
        "task_id": task_id,
        "client_id": client_id,
        "state": "completed",
    }

    return JSONResponse(
        status_code=200,
        content=response_body,
    )


# ---------------------------------------------------------------------------
# Batch API authentication helper
# ---------------------------------------------------------------------------

def _auth_record(request: FastAPIRequest):
    """
    Verify the API key using the same rules as the run endpoint.
    """

    raw = request.headers.get("authorization", "")

    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()

    if not raw:
        raw = request.headers.get("x-api-key", "")

    key_store = request.app.state.key_store

    return key_store.verify(raw) if raw else None


# ---------------------------------------------------------------------------
# Async batch API: create
# ---------------------------------------------------------------------------

async def _handle_batch_create(
    request: FastAPIRequest,
) -> FastAPIResponse:
    """
    POST /v1/batches

    Body:

        {
            "task_id": "...",
            "task_version": 1,
            "urls": [
                "https://...",
                "https://..."
            ],
            "chunk_size": 100,
            "max_attempts": 3,
            "input_manifest": [...]
        }
    """

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    record = _auth_record(request)

    if record is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "unauthorized",
                    "message": "missing or invalid API key",
                },
            },
        )

    # ------------------------------------------------------------------
    # Coordinator must be configured
    # ------------------------------------------------------------------
    coordinator = request.app.state.batch_coordinator

    if coordinator is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "batch_unavailable",
                    "message": "batch processing is not configured",
                },
            },
        )

    # ------------------------------------------------------------------
    # Parse JSON
    # ------------------------------------------------------------------
    try:
        body = await request.json()

    except Exception:
        body = None

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "request body must be a JSON object",
                },
            },
        )

    client_id = record.client_id

    task_id = body.get("task_id")
    task_version = body.get("task_version")
    urls = body.get("urls")

    # ------------------------------------------------------------------
    # Validate task_id
    # ------------------------------------------------------------------
    if not isinstance(task_id, str) or not task_id.strip():
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "task_id is required",
                },
            },
        )

    # ------------------------------------------------------------------
    # Validate task_version
    # ------------------------------------------------------------------
    if (
        not isinstance(task_version, int)
        or isinstance(task_version, bool)
        or task_version < 1
    ):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "task_version must be an integer >= 1",
                },
            },
        )

    # ------------------------------------------------------------------
    # Validate URLs
    # ------------------------------------------------------------------
    if not isinstance(urls, list) or not urls:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "urls must be a non-empty list",
                },
            },
        )

    if any(
        not isinstance(url, str) or not url.strip()
        for url in urls
    ):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": (
                        "urls must contain only non-empty strings"
                    ),
                },
            },
        )

    # ------------------------------------------------------------------
    # Validate chunk_size
    # ------------------------------------------------------------------
    chunk_size = body.get(
        "chunk_size",
        100,
    )

    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": (
                        "chunk_size must be an integer >= 1"
                    ),
                },
            },
        )

    # ------------------------------------------------------------------
    # Validate max_attempts
    # ------------------------------------------------------------------
    max_attempts = body.get(
        "max_attempts",
        3,
    )

    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts < 1
    ):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": (
                        "max_attempts must be an integer >= 1"
                    ),
                },
            },
        )

    # ------------------------------------------------------------------
    # Requested frozen task version must exist for this tenant
    # ------------------------------------------------------------------
    registry = request.app.state.registry

    spec = registry.get_task(
        client_id,
        task_id.strip(),
        version=task_version,
    )

    if spec is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "not_found",
                    "message": (
                        f"task {task_id!r} version "
                        f"{task_version} not found"
                    ),
                },
            },
        )

    # ------------------------------------------------------------------
    # Validate input_manifest
    # ------------------------------------------------------------------
    input_manifest = body.get(
        "input_manifest",
        [],
    )

    if input_manifest is None:
        input_manifest = []

    if (
        not isinstance(input_manifest, list)
        or any(
            not isinstance(item, str)
            for item in input_manifest
        )
    ):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": (
                        "input_manifest must be a list of strings"
                    ),
                },
            },
        )

    # ------------------------------------------------------------------
    # Submit batch
    # ------------------------------------------------------------------
    try:
        batch = await coordinator.submit(
            client_id=client_id,
            task_id=task_id.strip(),
            task_version=task_version,
            urls=urls,
            chunk_size=chunk_size,
            max_attempts=max_attempts,
            input_manifest=input_manifest,
        )

    except Exception as e:
        logger.exception(
            "batch submission failed",
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "batch_submission_failed",
                    "message": (
                        f"{type(e).__name__}: {e}"
                    ),
                },
            },
        )

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------
    return JSONResponse(
        status_code=201,
        content={
            "batch_id": batch.batch_id,
            "batch": batch.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Async batch API: get
# ---------------------------------------------------------------------------

async def _handle_batch_get(
    request: FastAPIRequest,
    *,
    batch_id: str,
) -> FastAPIResponse:
    """
    GET /v1/batches/{id}

    Returns the full serialized batch (including every chunk's state,
    attempts, error, and — thanks to the chunk-records persistence
    change — its records). Scoped to the authenticated client.

    The real BatchStore is synchronous; test doubles and future async
    implementations may be coroutines. We tolerate both.
    """
    

    # --- auth --------------------------------------------------------
    record = _auth_record(request)
    if record is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "unauthorized",
                    "message": "missing or invalid API key",
                },
            },
        )

    # --- store must be configured -----------------------------------
    batch_store = request.app.state.batch_store
    if batch_store is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "batch_unavailable",
                    "message": "batch storage is not configured",
                },
            },
        )

    # --- fetch, scoped by authenticated client ----------------------
    try:
        batch = await maybe_await(
            batch_store.get(record.client_id, batch_id)
        )
    except Exception as e:
        logger.exception("batch read failed for %r", batch_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "batch_read_failed",
                    "message": f"{type(e).__name__}: {e}",
                },
            },
        )

    if batch is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "not_found",
                    "message": f"batch {batch_id!r} not found",
                },
            },
        )

    # --- success -----------------------------------------------------
    return JSONResponse(
        status_code=200,
        content={
            "batch_id": batch.batch_id,
            "batch": batch.to_dict(),
        },
    )

async def _handle_batch_result(
    request: FastAPIRequest,
    *,
    batch_id: str,
) -> FastAPIResponse:
    """
    GET /v1/batches/{id}/result

    Merges the records persisted in each succeeded chunk into one
    dataset. Chunks that are still queued/running/paused or that have
    failed are excluded from `records` but still listed in
    `chunk_manifest`, so the caller can see exactly what went in.

    Behaviour:
        - returns partial results by default
        - `?require_complete=1` returns 409 when any chunk is missing
          or failed (this is the "give me the finished dataset or
          nothing" mode)

    Response shape:
        {
          batch_id, state, task_id, task_version, client_id,
          total_chunks, completed_chunks, failed_chunks, pending_chunks,
          records_count, records,
          chunk_manifest: [{chunk_id, index, state, records_count,
                            attempts, error}, ...],
          complete: bool
        }
    """
   

    # --- auth --------------------------------------------------------
    record = _auth_record(request)
    if record is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "unauthorized",
                    "message": "missing or invalid API key",
                },
            },
        )

    # --- store must be configured -----------------------------------
    batch_store = request.app.state.batch_store
    if batch_store is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "batch_unavailable",
                    "message": "batch storage is not configured",
                },
            },
        )

    # --- fetch, scoped by authenticated client ----------------------
    try:
        batch = await maybe_await(
            batch_store.get(record.client_id, batch_id)
        )
    except Exception as e:
        logger.exception("batch read failed for %r", batch_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "batch_read_failed",
                    "message": f"{type(e).__name__}: {e}",
                },
            },
        )

    if batch is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "not_found",
                    "message": f"batch {batch_id!r} not found",
                },
            },
        )

    # --- merge ------------------------------------------------------
    merged: list[dict] = []
    manifest: list[dict] = []
    completed = failed = pending = 0

    for chunk in batch.chunks:
        state = chunk.state.value
        if state == "succeeded":
            completed += 1
            merged.extend(chunk.records)
        elif state == "failed":
            failed += 1
        else:
            pending += 1

        manifest.append({
            "chunk_id": chunk.chunk_id,
            "index": chunk.index,
            "state": state,
            "records_count": chunk.records_count,
            "attempts": chunk.attempts,
            "error": chunk.error,
        })

    complete = (failed == 0 and pending == 0)

    # --- optional strictness ----------------------------------------
    require_complete = request.query_params.get(
        "require_complete", ""
    ).lower() in ("1", "true", "yes")

    if require_complete and not complete:
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "batch_not_complete",
                    "message": (
                        f"batch {batch_id!r} is not complete: "
                        f"{completed} succeeded, "
                        f"{failed} failed, "
                        f"{pending} pending"
                    ),
                },
            },
        )

    # --- success ----------------------------------------------------
    return JSONResponse(
        status_code=200,
        content={
            "batch_id": batch.batch_id,
            "state": batch.state.value,
            "task_id": batch.task_id,
            "task_version": batch.task_version,
            "client_id": batch.client_id,
            "total_chunks": batch.total_chunks,
            "completed_chunks": completed,
            "failed_chunks": failed,
            "pending_chunks": pending,
            "records_count": len(merged),
            "records": merged,
            "chunk_manifest": manifest,
            "complete": complete,
        },
    )
# ---------------------------------------------------------------------------
# Async batch API: pause / resume / cancel
# ---------------------------------------------------------------------------

async def _handle_batch_control(
    request: FastAPIRequest,
    *,
    batch_id: str,
    action: str,
) -> FastAPIResponse:
    """
    POST /v1/batches/{id}/pause
    POST /v1/batches/{id}/resume
    POST /v1/batches/{id}/cancel

    Shared handler for the three batch lifecycle controls. Delegates state
    transitions to the `Batch` model's own `pause()` / `resume()` /
    `cancel()` methods so the transition rules live in exactly one place.

    Response shape on success:
        {"batch_id": "...", "batch": {...}, "changed": true|false}
    `changed` is False when the call was a no-op (idempotent pause or
    cancel), and True when the batch state actually moved.

    Failure codes:
        401 unauthorized          — missing or invalid API key
        503 batch_unavailable     — batch_store not configured
        404 not_found             — batch does not exist for this tenant
        409 invalid_state         — action not allowed from current state
        500 batch_read_failed     — the batch store threw on read
        500 batch_save_failed     — the batch store threw on write
    """

    # --- auth --------------------------------------------------------
    record = _auth_record(request)
    if record is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "unauthorized",
                    "message": "missing or invalid API key",
                },
            },
        )

    client_id = record.client_id

    # --- store must be configured -----------------------------------
    batch_store = request.app.state.batch_store
    if batch_store is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "batch_unavailable",
                    "message": "batch storage is not configured",
                },
            },
        )

    # --- fetch (tolerate sync OR async store implementations) -------
    try:
        batch = await maybe_await(
            batch_store.get(client_id, batch_id)
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "batch_read_failed",
                    "message": f"{type(e).__name__}: {e}",
                },
            },
        )

    if batch is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "not_found",
                    "message": f"batch {batch_id!r} not found",
                },
            },
        )

    # --- state validation + transition ------------------------------
    from src.jobs.batch import BatchState

    terminal = (BatchState.COMPLETED, BatchState.FAILED, BatchState.CANCELLED)

    if action == "pause":
        if batch.state in terminal:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "invalid_state",
                        "message": (
                            f"cannot pause a batch in state "
                            f"{batch.state.value!r}"
                        ),
                    },
                },
            )
        if batch.state == BatchState.PAUSED:
            # Idempotent no-op — do not re-save.
            return JSONResponse(
                status_code=200,
                content={
                    "batch_id": batch_id,
                    "batch": batch.to_dict(),
                    "changed": False,
                },
            )
        batch.pause()

    elif action == "resume":
        if batch.state != BatchState.PAUSED:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "invalid_state",
                        "message": (
                            f"cannot resume a batch in state "
                            f"{batch.state.value!r}; only paused "
                            f"batches can be resumed"
                        ),
                    },
                },
            )
        batch.resume()

    elif action == "cancel":
        if batch.state in (BatchState.COMPLETED, BatchState.FAILED):
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "invalid_state",
                        "message": (
                            f"cannot cancel a batch in state "
                            f"{batch.state.value!r}"
                        ),
                    },
                },
            )
        if batch.state == BatchState.CANCELLED:
            return JSONResponse(
                status_code=200,
                content={
                    "batch_id": batch_id,
                    "batch": batch.to_dict(),
                    "changed": False,
                },
            )
        batch.cancel()

    else:
        # Unreachable — the regex only matches pause|resume|cancel. Kept
        # as a defensive guard in case the regex is ever loosened.
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_action",
                    "message": f"unknown action {action!r}",
                },
            },
        )

    # --- persist -----------------------------------------------------
    try:
        await maybe_await(batch_store.save(batch))
    except Exception as e:
        logger.exception("batch save failed for %r", batch_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "batch_save_failed",
                    "message": f"{type(e).__name__}: {e}",
                },
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "batch_id": batch_id,
            "batch": batch.to_dict(),
            "changed": True,
        },
    )


# ---------------------------------------------------------------------------
# Smoke test — drives the app in-process via httpx ASGI transport.
# No uvicorn, no network, no subprocess.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from httpx import ASGITransport, AsyncClient

    async def run():
        key_store = ApiKeyStore()
        _, raw_key = key_store.create("acme")

        registry = ServiceRegistry()

        app = build_app(
            registry,
            key_store,
        )

        transport = ASGITransport(
            app=app,
        )

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:

            # ----------------------------------------------------------
            # 1. Root endpoint works without auth
            # ----------------------------------------------------------
            r = await client.get("/")

            assert r.status_code == 200, r.text
            assert r.json()["name"] == "ACES API"

            # ----------------------------------------------------------
            # 2. No auth → 401
            # ----------------------------------------------------------
            r = await client.get(
                "/v1/tasks",
            )

            assert r.status_code == 401, r.text
            assert r.json()["error"]["code"] == "unauthorized"

            # ----------------------------------------------------------
            # 3. Auth header → 200, empty list
            # ----------------------------------------------------------
            auth = {
                "Authorization": f"Bearer {raw_key}",
            }

            r = await client.get(
                "/v1/tasks",
                headers=auth,
            )

            assert r.status_code == 200, r.text
            assert r.json()["tasks"] == []

            # ----------------------------------------------------------
            # 4. Create a task
            # ----------------------------------------------------------
            r = await client.post(
                "/v1/tasks",
                headers=auth,
                json={
                    "natural_language_prompt": (
                        "find laptop prices"
                    ),
                    "target": {
                        "start_urls": [
                            "https://example.com/a",
                        ],
                    },
                    "fields": [
                        {
                            "name": "title",
                        },
                        {
                            "name": "price",
                            "type": "currency",
                        },
                    ],
                },
            )

            assert r.status_code == 201, r.text

            task_id = r.json()["task_id"]

            # ----------------------------------------------------------
            # 5. Fetch it back
            # ----------------------------------------------------------
            r = await client.get(
                f"/v1/tasks/{task_id}",
                headers=auth,
            )

            assert r.status_code == 200, r.text
            assert r.json()["task"]["task_id"] == task_id

            # ----------------------------------------------------------
            # 6. Cross-tenant isolation
            # ----------------------------------------------------------
            _, other_key = key_store.create("other")

            r = await client.get(
                f"/v1/tasks/{task_id}",
                headers={
                    "Authorization": f"Bearer {other_key}",
                },
            )

            assert r.status_code == 404, r.text

            # ----------------------------------------------------------
            # 7. Unknown path → 404
            # ----------------------------------------------------------
            r = await client.get(
                "/v1/totally-not-a-route",
                headers=auth,
            )

            assert r.status_code == 404, r.text

            # ----------------------------------------------------------
            # 8. Wrong method → 404
            # ----------------------------------------------------------
            r = await client.post(
                "/v1/does-not-exist",
                headers=auth,
                json={},
            )

            assert r.status_code == 404, r.text

            # ----------------------------------------------------------
            # 9. Validation endpoint
            # ----------------------------------------------------------
            r = await client.post(
                f"/v1/tasks/{task_id}/validate",
                headers=auth,
            )

            assert r.status_code == 200, r.text
            assert r.json()["valid"] is True

            # ----------------------------------------------------------
            # 10. Delete
            # ----------------------------------------------------------
            r = await client.delete(
                f"/v1/tasks/{task_id}",
                headers=auth,
            )

            assert r.status_code == 204, r.text

        print("API server OK.")

    asyncio.run(run())