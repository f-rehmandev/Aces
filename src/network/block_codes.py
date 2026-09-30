"""
Block code classification — spec §16.1.

Central place to decide what an HTTP status code (or a provider's error
shape) means for the tier fallback logic. Keeping this in one module
means ScraperAPI's 402, WebScrapingAPI's 429, and Byparr's "unsolvable
challenge" all route through the same decision function.

Public API:

    classify_http_status(status_code) -> BlockKind
    classify_provider_error(error) -> BlockKind
    should_fall_through(kind) -> bool

BlockKinds that trigger a fallback:  RATE_LIMIT, PAYMENT_REQUIRED,
ACCESS_BLOCK, SERVICE_UNAVAILABLE, PROVIDER_FAILURE, UNKNOWN.

BlockKinds that should NOT trigger fallback (the request succeeded, or
retrying elsewhere would waste time):  OK, NOT_FOUND, REDIRECT, BAD_REQUEST.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional


class BlockKind(str, Enum):
    OK = "ok"                                # 2xx
    REDIRECT = "redirect"                    # 3xx (handled by the caller)
    BAD_REQUEST = "bad_request"              # 400 — our fault, don't retry
    ACCESS_BLOCK = "access_block"            # 403 — bot wall
    NOT_FOUND = "not_found"                  # 404 — not a block, page is gone
    RATE_LIMIT = "rate_limit"                # 429 — back off
    PAYMENT_REQUIRED = "payment_required"    # 402 — out of credits
    SERVER_ERROR = "server_error"            # 500
    SERVICE_UNAVAILABLE = "service_unavailable"  # 503 — often a Cloudflare wall
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    PROVIDER_FAILURE = "provider_failure"    # provider-specific error class
    UNKNOWN = "unknown"


# HTTP status → kind
_STATUS_MAP = {
    200: BlockKind.OK, 201: BlockKind.OK, 202: BlockKind.OK,
    204: BlockKind.OK, 206: BlockKind.OK,
    301: BlockKind.REDIRECT, 302: BlockKind.REDIRECT,
    303: BlockKind.REDIRECT, 307: BlockKind.REDIRECT, 308: BlockKind.REDIRECT,
    400: BlockKind.BAD_REQUEST,
    401: BlockKind.ACCESS_BLOCK,       # login required — treat as wall
    402: BlockKind.PAYMENT_REQUIRED,
    403: BlockKind.ACCESS_BLOCK,
    404: BlockKind.NOT_FOUND,
    408: BlockKind.TIMEOUT,
    429: BlockKind.RATE_LIMIT,
    500: BlockKind.SERVER_ERROR,
    502: BlockKind.SERVER_ERROR,
    503: BlockKind.SERVICE_UNAVAILABLE,
    504: BlockKind.SERVICE_UNAVAILABLE,
}


# Which kinds should trigger a fall-through to the next tier
_FALL_THROUGH = {
    BlockKind.ACCESS_BLOCK,
    BlockKind.RATE_LIMIT,
    BlockKind.PAYMENT_REQUIRED,
    BlockKind.SERVER_ERROR,
    BlockKind.SERVICE_UNAVAILABLE,
    BlockKind.TIMEOUT,
    BlockKind.NETWORK_ERROR,
    BlockKind.PROVIDER_FAILURE,
    BlockKind.UNKNOWN,
}


def classify_http_status(status_code: Optional[int]) -> BlockKind:
    """Map an HTTP status code (or None for network-level failure) to a BlockKind."""
    if status_code is None:
        return BlockKind.NETWORK_ERROR
    if status_code in _STATUS_MAP:
        return _STATUS_MAP[status_code]
    if 200 <= status_code < 300:
        return BlockKind.OK
    if 300 <= status_code < 400:
        return BlockKind.REDIRECT
    if 400 <= status_code < 500:
        return BlockKind.BAD_REQUEST
    if 500 <= status_code < 600:
        return BlockKind.SERVER_ERROR
    return BlockKind.UNKNOWN


def classify_provider_error(error: Exception) -> BlockKind:
    """
    Best-effort mapping from a provider's exception to a BlockKind.

    We look at the exception class name and the message, because each
    provider raises its own SDK error class with a numeric code buried
    somewhere. Rather than importing every SDK, we string-match on the
    common patterns.
    """
    if error is None:
        return BlockKind.OK
    name = type(error).__name__.lower()
    msg = str(error).lower()

    # Direct type matches (works when providers use httpx/requests/our types)
    if "timeout" in name or "timeout" in msg or "timed out" in msg:
        return BlockKind.TIMEOUT
    if "ratelimit" in name or "rate limit" in msg or "too many requests" in msg:
        return BlockKind.RATE_LIMIT
    if "payment" in name or "402" in msg or "insufficient" in msg:
        return BlockKind.PAYMENT_REQUIRED
    if "forbidden" in name or "access" in name or "403" in msg:
        return BlockKind.ACCESS_BLOCK
    if "notfound" in name or "404" in msg:
        return BlockKind.NOT_FOUND
    if "unavailable" in name or "503" in msg:
        return BlockKind.SERVICE_UNAVAILABLE

    # Connection-level failures
    if any(s in msg for s in (
        "connection", "connect", "network", "dns", "resolve",
        "unreachable", "reset by peer",
    )):
        return BlockKind.NETWORK_ERROR

    return BlockKind.PROVIDER_FAILURE


def should_fall_through(kind: BlockKind) -> bool:
    """True if this outcome should trigger a fall-through to the next tier."""
    return kind in _FALL_THROUGH


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # HTTP codes
    assert classify_http_status(200) == BlockKind.OK
    assert classify_http_status(204) == BlockKind.OK
    assert classify_http_status(301) == BlockKind.REDIRECT
    assert classify_http_status(400) == BlockKind.BAD_REQUEST
    assert classify_http_status(401) == BlockKind.ACCESS_BLOCK
    assert classify_http_status(402) == BlockKind.PAYMENT_REQUIRED
    assert classify_http_status(403) == BlockKind.ACCESS_BLOCK
    assert classify_http_status(404) == BlockKind.NOT_FOUND
    assert classify_http_status(408) == BlockKind.TIMEOUT
    assert classify_http_status(429) == BlockKind.RATE_LIMIT
    assert classify_http_status(500) == BlockKind.SERVER_ERROR
    assert classify_http_status(503) == BlockKind.SERVICE_UNAVAILABLE
    assert classify_http_status(None) == BlockKind.NETWORK_ERROR

    # Fall-through decisions
    assert not should_fall_through(BlockKind.OK)
    assert not should_fall_through(BlockKind.REDIRECT)
    assert not should_fall_through(BlockKind.BAD_REQUEST)
    assert not should_fall_through(BlockKind.NOT_FOUND)
    assert should_fall_through(BlockKind.ACCESS_BLOCK)
    assert should_fall_through(BlockKind.RATE_LIMIT)
    assert should_fall_through(BlockKind.PAYMENT_REQUIRED)
    assert should_fall_through(BlockKind.SERVICE_UNAVAILABLE)
    assert should_fall_through(BlockKind.TIMEOUT)
    assert should_fall_through(BlockKind.NETWORK_ERROR)
    assert should_fall_through(BlockKind.PROVIDER_FAILURE)

    # Provider error classification
    class FakeTimeout(Exception): pass
    class FakeRateLimit(Exception): pass
    class FakePayment(Exception): pass

    assert classify_provider_error(FakeTimeout("boom")) == BlockKind.TIMEOUT
    assert classify_provider_error(FakeRateLimit("429 too many requests")) == BlockKind.RATE_LIMIT
    assert classify_provider_error(FakePayment("insufficient credits")) == BlockKind.PAYMENT_REQUIRED
    assert classify_provider_error(RuntimeError("connection reset by peer")) == BlockKind.NETWORK_ERROR
    assert classify_provider_error(RuntimeError("dns resolution failed")) == BlockKind.NETWORK_ERROR
    assert classify_provider_error(RuntimeError("totally weird")) == BlockKind.PROVIDER_FAILURE
    assert classify_provider_error(None) == BlockKind.OK

    print("Block codes OK.")