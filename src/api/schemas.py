"""
API request/response schemas — spec §45, §45.3.

Framework-agnostic dataclasses. The FastAPI adapter (added later) will
serialize/deserialize these; the core router deals only in these objects.

Request shape:
    Request(method, path, path_params, query, headers, body, client_id)

Response shape:
    Response(status, body, headers)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

@dataclass
class Request:
    method: str
    path: str
    path_params: dict[str, str] = field(default_factory=dict)
    query: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    body: Optional[dict] = None
    client_id: str = ""          # populated by auth middleware

    def header(self, name: str, default: str = "") -> str:
        # Case-insensitive lookup
        for k, v in self.headers.items():
            if k.lower() == name.lower():
                return v
        return default


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

@dataclass
class Response:
    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def to_dict(self) -> dict:
        return {"status": self.status, "body": self.body, "headers": dict(self.headers)}


# ---------------------------------------------------------------------------
# Errors (§45.3 — user-visible error contracts)
# ---------------------------------------------------------------------------

@dataclass
class ApiError:
    code: str
    message: str
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {"error": {"code": self.code, "message": self.message}}
        if self.details:
            d["error"]["details"] = self.details
        return d


def error_response(
    status: int, code: str, message: str, details: Optional[dict] = None,
) -> Response:
    return Response(status=status, body=ApiError(code, message, details).to_dict())


# ---------------------------------------------------------------------------
# Common errors as helpers
# ---------------------------------------------------------------------------

def unauthorized(message: str = "missing or invalid API key") -> Response:
    return error_response(401, "unauthorized", message)


def forbidden(message: str = "not permitted") -> Response:
    return error_response(403, "forbidden", message)


def not_found(message: str = "not found") -> Response:
    return error_response(404, "not_found", message)


def bad_request(message: str, details: Optional[dict] = None) -> Response:
    return error_response(400, "bad_request", message, details)


def rate_limited(retry_after_seconds: int) -> Response:
    return Response(
        status=429,
        body=ApiError("rate_limited", "too many requests").to_dict(),
        headers={"Retry-After": str(retry_after_seconds)},
    )


def server_error(message: str = "internal error") -> Response:
    return error_response(500, "internal_error", message)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    r = Request(method="GET", path="/v1/jobs/j-1", path_params={"id": "j-1"})
    assert r.header("X-Nonexistent") == ""
    r.headers["Authorization"] = "Bearer abc"
    assert r.header("authorization") == "Bearer abc"  # case-insensitive

    resp = Response(status=200, body={"ok": True})
    assert resp.ok
    assert not Response(status=400).ok

    err = error_response(404, "not_found", "job j-1 not found")
    assert err.status == 404
    assert err.body["error"]["code"] == "not_found"

    rl = rate_limited(retry_after_seconds=30)
    assert rl.status == 429
    assert rl.headers["Retry-After"] == "30"

    print("API schemas OK.")