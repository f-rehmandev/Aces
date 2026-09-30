"""
Unit tests for the FastAPI HTTP server (spec §45).

Drives the ASGI app in-process via httpx.ASGITransport — no uvicorn, no
sockets, no subprocess. Every test exercises the full HTTP path:
    FastAPI → _dispatch → internal Router → auth/rate-limit/body-size
    → handler → internal Response → FastAPI Response

Covers:
    - Route ordering (regression guard for the Y.9.1-fix)
    - Root endpoint reachable without auth
    - Auth middleware runs on every /v1/* route
    - Full CRUD lifecycle over HTTP
    - Cross-tenant isolation
    - Validation endpoint
    - History / changes / quality read-only views
    - Body-size and rate-limit middleware (inherited from internal router)
    - Error responses carry the correct shape
"""
import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.keys import ApiKeyStore
from src.api.registry import ServiceRegistry
from src.api.server import build_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Returns (client, acme_key, other_key, registry, key_store)."""
    key_store = ApiKeyStore()
    _, acme_key = key_store.create("acme")
    _, other_key = key_store.create("other")

    registry = ServiceRegistry()
    app = build_app(registry, key_store)

    transport = ASGITransport(app=app)
    # Small rate limit so we can exercise that middleware too, but high
    # enough that normal tests never trip it.
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, acme_key, other_key, registry, key_store


def _run(coro):
    return asyncio.run(coro)


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _task_body():
    return {
        "natural_language_prompt": "find laptop prices",
        "target": {"start_urls": ["https://example.com/a"]},
        "fields": [
            {"name": "title"},
            {"name": "price", "type": "currency"},
        ],
    }


# ===========================================================================
# Route ordering (regression guard)
# ===========================================================================

def test_root_route_precedes_catch_all():
    """
    Regression guard for Y.9.1-fix.

    FastAPI evaluates routes in registration order. If the catch-all
    `/{full_path:path}` is registered before `/`, the root request
    falls into the internal router and 404s. This must not happen.
    """
    app = build_app(ServiceRegistry(), ApiKeyStore())
    paths = [r.path for r in app.routes]
    assert "/" in paths
    assert "/{full_path:path}" in paths
    assert paths.index("/") < paths.index("/{full_path:path}")


# ===========================================================================
# Root endpoint
# ===========================================================================

def test_root_reachable_without_auth(env):
    client, *_ = env
    r = _run(client.get("/"))
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "ACES API"
    assert "/v1/*" in body["hint"]


# ===========================================================================
# Auth
# ===========================================================================

def test_unauthenticated_request_returns_401(env):
    client, *_ = env
    r = _run(client.get("/v1/tasks"))
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"


def test_invalid_key_returns_401(env):
    client, *_ = env
    r = _run(client.get(
        "/v1/tasks",
        headers=_auth("aces_definitely-not-a-real-key"),
    ))
    assert r.status_code == 401


def test_api_key_header_also_works(env):
    client, acme_key, *_ = env
    r = _run(client.get("/v1/tasks", headers={"X-API-Key": acme_key}))
    assert r.status_code == 200


def test_revoked_key_rejected(env):
    client, acme_key, _, _, key_store = env
    # Revoke via the store (find record by prefix)
    for rec in key_store.list_for_client("acme"):
        key_store.revoke(rec.key_id)
    r = _run(client.get("/v1/tasks", headers=_auth(acme_key)))
    assert r.status_code == 401


# ===========================================================================
# Task CRUD over HTTP
# ===========================================================================

def test_create_task_returns_201(env):
    client, acme_key, *_ = env
    r = _run(client.post("/v1/tasks", headers=_auth(acme_key), json=_task_body()))
    assert r.status_code == 201
    body = r.json()
    assert "task_id" in body
    assert body["task"]["natural_language_prompt"] == "find laptop prices"


def test_list_tasks_after_create(env):
    client, acme_key, *_ = env
    _run(client.post("/v1/tasks", headers=_auth(acme_key), json=_task_body()))
    _run(client.post("/v1/tasks", headers=_auth(acme_key), json=_task_body()))
    r = _run(client.get("/v1/tasks", headers=_auth(acme_key)))
    assert r.status_code == 200
    assert len(r.json()["tasks"]) == 2


def test_get_task_by_id(env):
    client, acme_key, *_ = env
    create = _run(client.post(
        "/v1/tasks", headers=_auth(acme_key), json=_task_body(),
    ))
    task_id = create.json()["task_id"]
    r = _run(client.get(f"/v1/tasks/{task_id}", headers=_auth(acme_key)))
    assert r.status_code == 200
    assert r.json()["task"]["task_id"] == task_id


def test_get_unknown_task_returns_404(env):
    client, acme_key, *_ = env
    r = _run(client.get("/v1/tasks/does-not-exist", headers=_auth(acme_key)))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_delete_task_returns_204(env):
    client, acme_key, *_ = env
    create = _run(client.post(
        "/v1/tasks", headers=_auth(acme_key), json=_task_body(),
    ))
    task_id = create.json()["task_id"]
    r = _run(client.delete(f"/v1/tasks/{task_id}", headers=_auth(acme_key)))
    assert r.status_code == 204
    # Second delete → 404
    r = _run(client.delete(f"/v1/tasks/{task_id}", headers=_auth(acme_key)))
    assert r.status_code == 404


# ===========================================================================
# Cross-tenant isolation
# ===========================================================================

def test_other_client_cannot_see_acme_task(env):
    client, acme_key, other_key, *_ = env
    create = _run(client.post(
        "/v1/tasks", headers=_auth(acme_key), json=_task_body(),
    ))
    task_id = create.json()["task_id"]
    r = _run(client.get(f"/v1/tasks/{task_id}", headers=_auth(other_key)))
    assert r.status_code == 404


def test_listing_scoped_to_client(env):
    client, acme_key, other_key, *_ = env
    _run(client.post("/v1/tasks", headers=_auth(acme_key), json=_task_body()))
    _run(client.post("/v1/tasks", headers=_auth(other_key), json=_task_body()))
    r = _run(client.get("/v1/tasks", headers=_auth(acme_key)))
    assert len(r.json()["tasks"]) == 1
    r = _run(client.get("/v1/tasks", headers=_auth(other_key)))
    assert len(r.json()["tasks"]) == 1


# ===========================================================================
# Validate endpoint
# ===========================================================================

def test_validate_returns_true_for_valid_task(env):
    client, acme_key, *_ = env
    create = _run(client.post(
        "/v1/tasks", headers=_auth(acme_key), json=_task_body(),
    ))
    task_id = create.json()["task_id"]
    r = _run(client.post(f"/v1/tasks/{task_id}/validate", headers=_auth(acme_key)))
    assert r.status_code == 200
    assert r.json()["valid"] is True
    assert r.json()["problems"] == []


def test_validate_flags_missing_fields(env):
    client, acme_key, *_ = env
    create = _run(client.post(
        "/v1/tasks",
        headers=_auth(acme_key),
        json={"natural_language_prompt": "x"},  # no fields, no URLs
    ))
    task_id = create.json()["task_id"]
    r = _run(client.post(f"/v1/tasks/{task_id}/validate", headers=_auth(acme_key)))
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert any("start URLs" in p for p in body["problems"])
    assert any("no fields" in p for p in body["problems"])


# ===========================================================================
# Read-only views
# ===========================================================================

def test_history_empty_after_create(env):
    client, acme_key, *_ = env
    create = _run(client.post(
        "/v1/tasks", headers=_auth(acme_key), json=_task_body(),
    ))
    task_id = create.json()["task_id"]
    r = _run(client.get(f"/v1/tasks/{task_id}/history", headers=_auth(acme_key)))
    assert r.status_code == 200
    assert r.json()["versions"] == []


def test_history_lists_dataset_versions(env):
    client, acme_key, _, registry, _ = env
    create = _run(client.post(
        "/v1/tasks", headers=_auth(acme_key), json=_task_body(),
    ))
    task_id = create.json()["task_id"]

    registry.append_dataset_version("acme", task_id, records=[{"title": "A"}])
    registry.append_dataset_version("acme", task_id, records=[{"title": "B"}])

    r = _run(client.get(f"/v1/tasks/{task_id}/history", headers=_auth(acme_key)))
    assert r.status_code == 200
    assert len(r.json()["versions"]) == 2


def test_changes_endpoint_detects_diff(env):
    client, acme_key, _, registry, _ = env
    create = _run(client.post(
        "/v1/tasks", headers=_auth(acme_key), json=_task_body(),
    ))
    task_id = create.json()["task_id"]

    registry.append_dataset_version(
        "acme", task_id,
        records=[{"title": "A", "price": "$10"}],
    )
    registry.append_dataset_version(
        "acme", task_id,
        records=[
            {"title": "A", "price": "$12"},
            {"title": "B", "price": "$20"},
        ],
    )

    r = _run(client.get(f"/v1/tasks/{task_id}/changes", headers=_auth(acme_key)))
    assert r.status_code == 200
    summary = r.json()["summary"]
    assert summary["modified"] == 1
    assert summary["new"] == 1


def test_quality_404_when_no_dataset(env):
    client, acme_key, *_ = env
    create = _run(client.post(
        "/v1/tasks", headers=_auth(acme_key), json=_task_body(),
    ))
    task_id = create.json()["task_id"]
    r = _run(client.get(f"/v1/tasks/{task_id}/quality", headers=_auth(acme_key)))
    assert r.status_code == 404


def test_quality_returns_latest(env):
    client, acme_key, _, registry, _ = env
    create = _run(client.post(
        "/v1/tasks", headers=_auth(acme_key), json=_task_body(),
    ))
    task_id = create.json()["task_id"]

    registry.append_dataset_version(
        "acme", task_id,
        records=[{"title": "A"}],
        quality_score=0.95, confidence_mean=0.88,
    )
    r = _run(client.get(f"/v1/tasks/{task_id}/quality", headers=_auth(acme_key)))
    assert r.status_code == 200
    body = r.json()
    assert body["quality_score"] == 0.95
    assert body["confidence_mean"] == 0.88


# ===========================================================================
# Error shapes
# ===========================================================================

def test_unknown_route_returns_structured_404(env):
    client, acme_key, *_ = env
    r = _run(client.get("/v1/nonexistent", headers=_auth(acme_key)))
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "not_found"
    assert "no route" in body["error"]["message"].lower()


def test_wrong_method_returns_404(env):
    client, acme_key, *_ = env
    # /v1/tasks only accepts GET and POST
    r = _run(client.patch("/v1/tasks", headers=_auth(acme_key), json={}))
    assert r.status_code == 404


def test_create_requires_body(env):
    client, acme_key, *_ = env
    r = _run(client.post("/v1/tasks", headers=_auth(acme_key)))
    # FastAPI treats empty body as no JSON; internal dispatcher sends
    # `body=None`, handler rejects it.
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"


def test_malformed_json_is_handled_gracefully(env):
    client, acme_key, *_ = env
    r = _run(client.post(
        "/v1/tasks",
        headers=_auth(acme_key),
        content=b"this is not json",
    ))
    # _dispatch treats unparseable body as None; handler returns 400.
    assert r.status_code == 400


# ===========================================================================
# Body size limit middleware
# ===========================================================================

def test_body_size_limit_enforced():
    """A small limit forces the middleware to reject large payloads."""
    key_store = ApiKeyStore()
    _, acme_key = key_store.create("acme")
    registry = ServiceRegistry()
    app = build_app(registry, key_store, max_body_bytes=200)

    transport = ASGITransport(app=app)

    async def scenario():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/v1/tasks",
                headers=_auth(acme_key),
                json={
                    "natural_language_prompt": "x" * 500,
                    "target": {"start_urls": ["https://example.com/a"]},
                    "fields": [{"name": "title"}],
                },
            )

    r = asyncio.run(scenario())
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


# ===========================================================================
# Rate limit middleware
# ===========================================================================

def test_rate_limit_enforced():
    """A tiny limit forces the middleware to reject the (N+1)th request."""
    key_store = ApiKeyStore()
    _, acme_key = key_store.create("acme")
    registry = ServiceRegistry()
    app = build_app(
        registry, key_store,
        rate_limit_per_minute=3,
    )

    transport = ASGITransport(app=app)

    async def scenario():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            responses = []
            for _ in range(5):
                responses.append(
                    await client.get("/v1/tasks", headers=_auth(acme_key))
                )
            return responses

    responses = asyncio.run(scenario())
    statuses = [r.status_code for r in responses]
    assert statuses.count(200) == 3
    assert statuses.count(429) == 2
    # 429 responses carry Retry-After
    assert "retry-after" in responses[3].headers