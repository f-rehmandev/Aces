"""Unit tests for the API router + middleware (spec §45.3, §45.4)."""
import pytest

from src.api.keys import ApiKeyStore
from src.api.schemas import Request, Response, error_response
from src.api.router import (
    Router, Route, auth_middleware, rate_limit_middleware,
    body_size_limit_middleware,
)


def _store_with_key(client_id="acme"):
    store = ApiKeyStore()
    _, raw = store.create(client_id)
    return store, raw


# --- route matching --------------------------------------------------

def test_route_matches_static_path():
    r = Route("GET", "/v1/ping", lambda req: Response())
    assert r.match("GET", "/v1/ping") == {}
    assert r.match("POST", "/v1/ping") is None
    assert r.match("GET", "/v1/other") is None


def test_route_extracts_params():
    r = Route("GET", "/v1/jobs/{id}", lambda req: Response())
    params = r.match("GET", "/v1/jobs/abc-123")
    assert params == {"id": "abc-123"}


def test_route_multiple_params():
    r = Route("GET", "/v1/t/{tid}/j/{jid}", lambda req: Response())
    params = r.match("GET", "/v1/t/t1/j/j1")
    assert params == {"tid": "t1", "jid": "j1"}


def test_route_param_does_not_match_slash():
    r = Route("GET", "/v1/jobs/{id}", lambda req: Response())
    assert r.match("GET", "/v1/jobs/a/b") is None


# --- dispatch --------------------------------------------------------

def test_dispatch_ok():
    r = Router()
    r.add("GET", "/v1/ping", lambda req: Response(200, {"pong": True}))
    resp = r.dispatch(Request(method="GET", path="/v1/ping"))
    assert resp.status == 200
    assert resp.body["pong"] is True


def test_dispatch_404_for_unknown_path():
    r = Router()
    resp = r.dispatch(Request(method="GET", path="/nope"))
    assert resp.status == 404


def test_dispatch_404_for_wrong_method():
    r = Router()
    r.add("GET", "/v1/x", lambda req: Response(200))
    resp = r.dispatch(Request(method="POST", path="/v1/x"))
    assert resp.status == 404


def test_dispatch_runs_middleware_before_handler():
    order = []
    r = Router()
    def mw(req):
        order.append("mw")
        return None
    def handler(req):
        order.append("handler")
        return Response(200)
    r.add("GET", "/v1/x", handler)
    r.add_middleware(mw)
    r.dispatch(Request(method="GET", path="/v1/x"))
    assert order == ["mw", "handler"]


def test_dispatch_short_circuits_on_middleware_response():
    called = {"handler": False}
    def mw(req):
        return Response(403, {"denied": True})
    def handler(req):
        called["handler"] = True
        return Response(200)
    r = Router()
    r.add("GET", "/v1/x", handler)
    r.add_middleware(mw)
    resp = r.dispatch(Request(method="GET", path="/v1/x"))
    assert resp.status == 403
    assert called["handler"] is False


def test_handler_exception_becomes_500():
    def boom(req): raise RuntimeError("nope")
    r = Router()
    r.add("GET", "/v1/x", boom)
    resp = r.dispatch(Request(method="GET", path="/v1/x"))
    assert resp.status == 500
    assert "nope" in resp.body["error"]["message"]


# --- auth middleware -------------------------------------------------

def test_auth_middleware_missing_header():
    store, _ = _store_with_key()
    r = Router()
    r.add("GET", "/v1/x", lambda req: Response(200))
    r.add_middleware(auth_middleware(store))
    resp = r.dispatch(Request(method="GET", path="/v1/x"))
    assert resp.status == 401


def test_auth_middleware_invalid_key():
    store, _ = _store_with_key()
    r = Router()
    r.add("GET", "/v1/x", lambda req: Response(200))
    r.add_middleware(auth_middleware(store))
    resp = r.dispatch(Request(
        method="GET", path="/v1/x",
        headers={"Authorization": "Bearer wrong"},
    ))
    assert resp.status == 401


def test_auth_middleware_bearer_header():
    store, raw = _store_with_key("acme")
    captured = {}
    def handler(req):
        captured["client_id"] = req.client_id
        return Response(200)
    r = Router()
    r.add("GET", "/v1/x", handler)
    r.add_middleware(auth_middleware(store))
    resp = r.dispatch(Request(
        method="GET", path="/v1/x",
        headers={"Authorization": f"Bearer {raw}"},
    ))
    assert resp.status == 200
    assert captured["client_id"] == "acme"


def test_auth_middleware_api_key_header():
    store, raw = _store_with_key("acme")
    r = Router()
    r.add("GET", "/v1/x", lambda req: Response(200, {"c": req.client_id}))
    r.add_middleware(auth_middleware(store))
    resp = r.dispatch(Request(
        method="GET", path="/v1/x",
        headers={"X-API-Key": raw},
    ))
    assert resp.status == 200
    assert resp.body["c"] == "acme"


def test_auth_middleware_revoked_key_rejected():
    store = ApiKeyStore()
    record, raw = store.create("acme")
    store.revoke(record.key_id)
    r = Router()
    r.add("GET", "/v1/x", lambda req: Response(200))
    r.add_middleware(auth_middleware(store))
    resp = r.dispatch(Request(
        method="GET", path="/v1/x",
        headers={"Authorization": f"Bearer {raw}"},
    ))
    assert resp.status == 401


# --- rate limit middleware ------------------------------------------

def test_rate_limit_allows_under_limit():
    store, raw = _store_with_key()
    r = Router()
    r.add("GET", "/v1/x", lambda req: Response(200))
    r.add_middleware(auth_middleware(store))
    r.add_middleware(rate_limit_middleware(per_minute=5))
    for _ in range(5):
        resp = r.dispatch(Request(
            method="GET", path="/v1/x",
            headers={"X-API-Key": raw},
        ))
        assert resp.status == 200


def test_rate_limit_blocks_over_limit():
    store, raw = _store_with_key()
    clock_val = [0.0]
    r = Router()
    r.add("GET", "/v1/x", lambda req: Response(200))
    r.add_middleware(auth_middleware(store))
    r.add_middleware(rate_limit_middleware(per_minute=2, clock=lambda: clock_val[0]))
    for _ in range(2):
        r.dispatch(Request(method="GET", path="/v1/x",
                           headers={"X-API-Key": raw}))
    resp = r.dispatch(Request(method="GET", path="/v1/x",
                              headers={"X-API-Key": raw}))
    assert resp.status == 429
    assert "Retry-After" in resp.headers


def test_rate_limit_window_resets():
    store, raw = _store_with_key()
    clock_val = [0.0]
    r = Router()
    r.add("GET", "/v1/x", lambda req: Response(200))
    r.add_middleware(auth_middleware(store))
    r.add_middleware(rate_limit_middleware(per_minute=2, clock=lambda: clock_val[0]))
    for _ in range(2):
        r.dispatch(Request(method="GET", path="/v1/x",
                           headers={"X-API-Key": raw}))
    # 61s later -> window resets
    clock_val[0] = 61.0
    resp = r.dispatch(Request(method="GET", path="/v1/x",
                              headers={"X-API-Key": raw}))
    assert resp.status == 200


# --- body size middleware ------------------------------------------

def test_body_size_limit_allows_small_body():
    r = Router()
    r.add("POST", "/v1/x", lambda req: Response(200))
    r.add_middleware(body_size_limit_middleware(max_bytes=1000))
    resp = r.dispatch(Request(
        method="POST", path="/v1/x", body={"small": "yes"},
    ))
    assert resp.status == 200


def test_body_size_limit_rejects_large_body():
    r = Router()
    r.add("POST", "/v1/x", lambda req: Response(200))
    r.add_middleware(body_size_limit_middleware(max_bytes=50))
    resp = r.dispatch(Request(
        method="POST", path="/v1/x", body={"data": "x" * 200},
    ))
    assert resp.status == 413
    assert resp.body["error"]["code"] == "payload_too_large"


def test_body_size_limit_allows_missing_body():
    r = Router()
    r.add("POST", "/v1/x", lambda req: Response(200))
    r.add_middleware(body_size_limit_middleware(max_bytes=10))
    resp = r.dispatch(Request(method="POST", path="/v1/x"))
    assert resp.status == 200