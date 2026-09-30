"""
API router + middleware — spec §45.3, §45.4.

Framework-agnostic. `Router` maps (METHOD, path_template) → handler and
runs middleware in order:

    auth  →  rate-limit  →  handler

Path templates use `{name}` for capture, e.g. `/v1/jobs/{id}`.

Middleware and handler signatures:
    middleware(request) -> Optional[Response]        # None = pass through
    handler(request)    -> Response
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.api.keys import ApiKeyStore
from src.api.schemas import Request, Response, unauthorized, rate_limited, not_found


Handler = Callable[[Request], Response]
Middleware = Callable[[Request], Optional[Response]]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@dataclass
class Route:
    method: str
    template: str
    handler: Handler
    _regex: re.Pattern = field(init=False)
    _param_names: list[str] = field(init=False)

    def __post_init__(self):
        # Turn "/v1/jobs/{id}" into regex + param names
        parts = re.split(r"(\{[^}]+\})", self.template)
        regex_parts = []
        names: list[str] = []
        for part in parts:
            if part.startswith("{") and part.endswith("}"):
                names.append(part[1:-1])
                regex_parts.append(r"([^/]+)")
            else:
                regex_parts.append(re.escape(part))
        self._regex = re.compile("^" + "".join(regex_parts) + "$")
        self._param_names = names

    def match(self, method: str, path: str) -> Optional[dict[str, str]]:
        if method.upper() != self.method.upper():
            return None
        m = self._regex.match(path)
        if not m:
            return None
        return dict(zip(self._param_names, m.groups()))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    def __init__(self):
        self._routes: list[Route] = []
        self._middleware: list[Middleware] = []

    # --- registration ---
    def add(self, method: str, path: str, handler: Handler) -> None:
        self._routes.append(Route(method=method, template=path, handler=handler))

    def add_middleware(self, mw: Middleware) -> None:
        self._middleware.append(mw)

    # --- dispatch ---
    def dispatch(self, request: Request) -> Response:
        # Match route first so 404 beats 401 for unknown paths.
        for route in self._routes:
            params = route.match(request.method, request.path)
            if params is None:
                continue
            request.path_params = params
            # Middleware in order
            for mw in self._middleware:
                outcome = mw(request)
                if outcome is not None:
                    return outcome
            try:
                return route.handler(request)
            except Exception as e:
                from src.api.schemas import server_error
                return server_error(f"{type(e).__name__}: {e}")
        return not_found(f"no route for {request.method} {request.path}")


# ---------------------------------------------------------------------------
# Middleware factories
# ---------------------------------------------------------------------------

def auth_middleware(store: ApiKeyStore) -> Middleware:
    """
    §45.1: API keys scoped per client; keys hashed at rest.
    Expects `Authorization: Bearer <key>` or `X-API-Key: <key>`.
    """
    def mw(request: Request) -> Optional[Response]:
        raw = request.header("Authorization")
        if raw.lower().startswith("bearer "):
            raw = raw[7:].strip()
        if not raw:
            raw = request.header("X-API-Key")
        if not raw:
            return unauthorized()
        record = store.verify(raw)
        if record is None:
            return unauthorized()
        request.client_id = record.client_id
        return None
    return mw


def rate_limit_middleware(
    per_minute: int,
    clock: Callable[[], float] = time.monotonic,
) -> Middleware:
    """
    §45.4: token-bucket style limit per client. Simple fixed window for now.
    """
    window: dict[str, list[float]] = {}

    def mw(request: Request) -> Optional[Response]:
        client = request.client_id or "anonymous"
        now = clock()
        cutoff = now - 60.0
        hits = [t for t in window.get(client, []) if t >= cutoff]
        if len(hits) >= per_minute:
            retry_after = int(60 - (now - hits[0])) + 1
            return rate_limited(retry_after_seconds=max(1, retry_after))
        hits.append(now)
        window[client] = hits
        return None
    return mw


def body_size_limit_middleware(max_bytes: int) -> Middleware:
    """
    §45.3: reject oversized request bodies before touching the handler.
    """
    import json
    def mw(request: Request) -> Optional[Response]:
        if request.body is None:
            return None
        try:
            size = len(json.dumps(request.body))
        except (TypeError, ValueError):
            size = 0
        if size > max_bytes:
            from src.api.schemas import error_response
            return error_response(
                413, "payload_too_large",
                f"body exceeds {max_bytes} bytes ({size} given)",
            )
        return None
    return mw


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.api.keys import ApiKeyStore

    def ping(req: Request) -> Response:
        return Response(status=200, body={"pong": True, "client": req.client_id})

    def echo(req: Request) -> Response:
        return Response(status=200, body={"id": req.path_params["id"]})

    store = ApiKeyStore()
    _, raw = store.create("acme")

    r = Router()
    r.add("GET", "/v1/ping", ping)
    r.add("GET", "/v1/items/{id}", echo)
    r.add_middleware(auth_middleware(store))

    # No auth → 401
    resp = r.dispatch(Request(method="GET", path="/v1/ping"))
    assert resp.status == 401

    # Valid auth → 200
    resp = r.dispatch(Request(
        method="GET", path="/v1/ping",
        headers={"Authorization": f"Bearer {raw}"},
    ))
    assert resp.status == 200
    assert resp.body["client"] == "acme"

    # Path param extraction
    resp = r.dispatch(Request(
        method="GET", path="/v1/items/abc",
        headers={"X-API-Key": raw},
    ))
    assert resp.status == 200
    assert resp.body["id"] == "abc"

    # Unknown route → 404
    resp = r.dispatch(Request(
        method="GET", path="/v1/nope",
        headers={"X-API-Key": raw},
    ))
    assert resp.status == 404

    # Wrong method → 404 (route match fails on method)
    resp = r.dispatch(Request(
        method="POST", path="/v1/ping",
        headers={"X-API-Key": raw},
    ))
    assert resp.status == 404

    # Rate limit
    clock_value = [100.0]
    def clock(): return clock_value[0]
    r2 = Router()
    r2.add("GET", "/v1/ping", ping)
    r2.add_middleware(auth_middleware(store))
    r2.add_middleware(rate_limit_middleware(per_minute=3, clock=clock))
    for _ in range(3):
        resp = r2.dispatch(Request(
            method="GET", path="/v1/ping",
            headers={"X-API-Key": raw},
        ))
        assert resp.status == 200, resp.body
    resp = r2.dispatch(Request(
        method="GET", path="/v1/ping",
        headers={"X-API-Key": raw},
    ))
    assert resp.status == 429
    assert "Retry-After" in resp.headers

    # Body size limit
    r3 = Router()
    r3.add("POST", "/v1/x", lambda req: Response(200, {"ok": True}))
    r3.add_middleware(body_size_limit_middleware(max_bytes=50))
    resp = r3.dispatch(Request(
        method="POST", path="/v1/x",
        body={"data": "x" * 100},
    ))
    assert resp.status == 413

    print("API router OK.")