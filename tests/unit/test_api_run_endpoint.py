"""
Unit tests for POST /v1/tasks/{id}/run and the job views
(GET /v1/jobs/{id}, GET /v1/jobs/{id}/result).

Covers:
    - Run endpoint: auth required, task must exist, tenant-scoped
    - Run returns a persisted job record with the executor's payload
    - Job status view: hides the (potentially large) result payload
    - Job result view: 400 when the job isn't completed
    - Cross-tenant isolation on both job routes
    - Executor errors surface as 500 with a `run_failed` code
    - Missing executor → 503
    - Body pass-through: only whitelisted keys reach the executor
"""
import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.keys import ApiKeyStore
from src.api.registry import ServiceRegistry
from src.api.server import build_app
from src.core.task_spec import FieldSpec, Target, TaskSpec


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

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


class RecordingExecutor:
    """
    Minimal executor stand-in that lets tests observe the payload it
    received. Matches JobExecutor's public surface (async execute()).
    """
    def __init__(self, result=None, raises=None):
        self.result = result or {
            "task_id": "x",
            "client_id": "acme",
            "records_count": 3,
            "quality_passed": True,
            "quality_score": 1.0,
            "confidence_mean": 0.9,
            "warnings": [],
        }
        self.raises = raises
        self.received: list[dict] = []

    async def execute(self, payload):
        self.received.append(dict(payload))
        if self.raises:
            raise self.raises
        return dict(self.result)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """
    Build a fully-wired app with a fake executor and a task under 'acme'.
    Returns (client, acme_key, other_key, registry, executor, task_id).
    """
    key_store = ApiKeyStore()
    _, acme_key = key_store.create("acme")
    _, other_key = key_store.create("other")

    registry = ServiceRegistry()
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.client_id = "acme"
    registry.save_task("acme", spec)

    executor = RecordingExecutor()
    app = build_app(registry, key_store, executor=executor)
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")

    return client, acme_key, other_key, registry, executor, spec.task_id


def _run(coro):
    return asyncio.run(coro)


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


# ===========================================================================
# Auth
# ===========================================================================

def test_run_requires_auth(env):
    client, _, _, _, _, task_id = env
    r = _run(client.post(f"/v1/tasks/{task_id}/run"))
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"


def test_run_rejects_bad_key(env):
    client, _, _, _, _, task_id = env
    r = _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth("aces_wrong"),
    ))
    assert r.status_code == 401


def test_run_accepts_x_api_key_header(env):
    client, acme_key, _, _, _, task_id = env
    r = _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers={"X-API-Key": acme_key},
    ))
    assert r.status_code == 200


# ===========================================================================
# Task lookup
# ===========================================================================

def test_run_unknown_task_returns_404(env):
    client, acme_key, *_ = env
    r = _run(client.post(
        "/v1/tasks/does-not-exist/run",
        headers=_auth(acme_key),
    ))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_run_other_tenants_task_returns_404(env):
    client, _, other_key, _, _, task_id = env
    r = _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(other_key),
    ))
    assert r.status_code == 404


# ===========================================================================
# Successful run
# ===========================================================================

def test_run_returns_completed_job(env):
    client, acme_key, _, _, executor, task_id = env
    r = _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(acme_key),
    ))
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "completed"
    assert body["task_id"] == task_id
    assert body["client_id"] == "acme"
    assert "job_id" in body
    # Executor's result is merged into the response
    assert body["records_count"] == 3
    assert body["quality_passed"] is True


def test_run_persists_job_record(env):
    client, acme_key, _, registry, _, task_id = env
    r = _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(acme_key),
    ))
    job_id = r.json()["job_id"]
    job = registry.get_job("acme", job_id)
    assert job is not None
    assert job["state"] == "completed"
    assert job["task_id"] == task_id
    assert job["result"]["records_count"] == 3


def test_run_payload_contains_task_and_client(env):
    client, acme_key, _, _, executor, task_id = env
    _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(acme_key),
    ))
    assert len(executor.received) == 1
    payload = executor.received[0]
    assert payload["task_id"] == task_id
    assert payload["client_id"] == "acme"

def test_executor_cannot_override_authoritative_response_fields(env):
    """
    Regression guard: the run handler must not let an executor's result
    override server-owned identity fields (job_id, task_id, client_id,
    state). A misbehaving or buggy executor that returns those keys
    with wrong values must not corrupt the response.
    """
    client, acme_key, _, _, executor, task_id = env

    # Executor returns hostile values for every authoritative field
    executor.result = {
        "task_id": "HIJACKED",
        "client_id": "HIJACKED",
        "job_id": "HIJACKED",
        "state": "HIJACKED",
        "records_count": 3,     # a legit result field — must survive
        "quality_passed": True,
        "quality_score": 1.0,
        "confidence_mean": 0.9,
        "warnings": [],
    }

    r = _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(acme_key),
    ))
    assert r.status_code == 200
    body = r.json()

    # The server-owned fields are correct, not hijacked
    assert body["task_id"] == task_id
    assert body["client_id"] == "acme"
    assert body["state"] == "completed"
    assert body["job_id"] != "HIJACKED"

    # The executor's legitimate result fields still come through
    assert body["records_count"] == 3
# ===========================================================================
# Body pass-through whitelist
# ===========================================================================

def test_run_passes_whitelisted_body_keys(env):
    client, acme_key, _, _, executor, task_id = env
    _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(acme_key),
        json={
            "default_currency": "USD",
            "default_country": "US",
            "failed_page_count": 2,
        },
    ))
    payload = executor.received[0]
    assert payload["default_currency"] == "USD"
    assert payload["default_country"] == "US"
    assert payload["failed_page_count"] == 2


def test_run_ignores_non_whitelisted_body_keys(env):
    client, acme_key, _, _, executor, task_id = env
    _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(acme_key),
        json={
            "evil_key": "should not reach executor",
            "task_id": "overridden",  # must not override the path param
            "client_id": "other",     # must not override auth
        },
    ))
    payload = executor.received[0]
    assert "evil_key" not in payload
    assert payload["task_id"] == task_id
    assert payload["client_id"] == "acme"


def test_run_with_no_body_is_fine(env):
    client, acme_key, _, _, executor, task_id = env
    r = _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(acme_key),
    ))
    assert r.status_code == 200
    assert len(executor.received) == 1


# ===========================================================================
# Job views
# ===========================================================================

def test_get_job_hides_result_payload(env):
    client, acme_key, _, _, _, task_id = env
    run = _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(acme_key),
    ))
    job_id = run.json()["job_id"]

    r = _run(client.get(f"/v1/jobs/{job_id}", headers=_auth(acme_key)))
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == job_id
    assert body["state"] == "completed"
    # The status view deliberately strips the (large) result
    assert "result" not in body


def test_get_job_result_returns_full_payload(env):
    client, acme_key, _, _, _, task_id = env
    run = _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(acme_key),
    ))
    job_id = run.json()["job_id"]

    r = _run(client.get(f"/v1/jobs/{job_id}/result", headers=_auth(acme_key)))
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == job_id
    assert body["result"]["records_count"] == 3


def test_get_unknown_job_returns_404(env):
    client, acme_key, *_ = env
    r = _run(client.get("/v1/jobs/does-not-exist", headers=_auth(acme_key)))
    assert r.status_code == 404


def test_get_other_tenants_job_returns_404(env):
    client, acme_key, other_key, _, _, task_id = env
    run = _run(client.post(
        f"/v1/tasks/{task_id}/run",
        headers=_auth(acme_key),
    ))
    job_id = run.json()["job_id"]

    r = _run(client.get(f"/v1/jobs/{job_id}", headers=_auth(other_key)))
    assert r.status_code == 404
    r = _run(client.get(
        f"/v1/jobs/{job_id}/result", headers=_auth(other_key),
    ))
    assert r.status_code == 404


def test_get_job_result_400_when_not_completed(env):
    """A job in a non-terminal state cannot have its result fetched."""
    client, acme_key, _, registry, _, _ = env
    # Inject a job directly in a non-terminal state
    registry.save_job("acme", "pending-job", {
        "job_id": "pending-job",
        "task_id": "t",
        "client_id": "acme",
        "state": "running",
        "result": None,
    })
    r = _run(client.get(
        "/v1/jobs/pending-job/result", headers=_auth(acme_key),
    ))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"


# ===========================================================================
# Failure paths
# ===========================================================================

def test_run_executor_failure_returns_500(env):
    """An executor that raises produces a 500 with a `run_failed` code."""
    key_store = ApiKeyStore()
    _, acme_key = key_store.create("acme")
    registry = ServiceRegistry()
    spec = TaskSpec(
        natural_language_prompt="x",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.client_id = "acme"
    registry.save_task("acme", spec)

    executor = RecordingExecutor(raises=RuntimeError("boom"))
    app = build_app(registry, key_store, executor=executor)

    async def scenario():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as c:
            return await c.post(
                f"/v1/tasks/{spec.task_id}/run",
                headers=_auth(acme_key),
            )

    r = asyncio.run(scenario())
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "run_failed"
    assert "boom" in r.json()["error"]["message"]


def test_run_missing_executor_returns_503(env):
    """A server built without an executor cannot run tasks."""
    key_store = ApiKeyStore()
    _, acme_key = key_store.create("acme")
    registry = ServiceRegistry()
    spec = TaskSpec(
        natural_language_prompt="x",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.client_id = "acme"
    registry.save_task("acme", spec)

    app = build_app(registry, key_store, executor=None)

    async def scenario():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as c:
            return await c.post(
                f"/v1/tasks/{spec.task_id}/run",
                headers=_auth(acme_key),
            )

    r = asyncio.run(scenario())
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "executor_unavailable"


# ===========================================================================
# Integration with a real JobExecutor
# ===========================================================================

def test_run_with_real_job_executor():
    """
    The fast-path must work with the real JobExecutor too — this is the
    integration guard that ties the two together.
    """
    from src.jobs.executor import JobExecutor

    key_store = ApiKeyStore()
    _, acme_key = key_store.create("acme")
    registry = ServiceRegistry()
    spec = TaskSpec(
        natural_language_prompt="x",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.client_id = "acme"
    registry.save_task("acme", spec)

    executor = JobExecutor(registry, FakeScraper(), FakeExtractor())
    app = build_app(registry, key_store, executor=executor)

    async def scenario():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as c:
            return await c.post(
                f"/v1/tasks/{spec.task_id}/run",
                headers=_auth(acme_key),
            )

    r = asyncio.run(scenario())
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "completed"
    assert body["records_count"] == 2
    # Job was persisted by the run handler
    assert registry.get_job("acme", body["job_id"]) is not None