"""
Byparr provider — spec §15.x (Tier 1c).

Byparr is a local Cloudflare-Turnstile solver that speaks the same
JSON API as the (now archived) FlareSolverr. It runs on the user's
machine or a sidecar Docker container, listens on http://localhost:8191
by default, and returns solved HTML plus cookies.

Wire format (both Byparr and FlareSolverr use this):

    POST http://localhost:8191/v1
    {
        "cmd":        "request.get",
        "url":        "https://target.example/",
        "maxTimeout": 60000,
        "cookies":    [{"name": "...", "value": "..."}, ...]   # optional
    }

Response on success:

    {
        "status":   "ok",
        "message":  "Challenge solved!",
        "solution": {
            "url":     "https://target.example/",
            "status":  200,
            "response": "<html>...</html>",
            "cookies": [{"name": "...", "value": "..."}, ...],
            "userAgent": "Mozilla/5.0 ..."
        }
    }

Response on failure:

    {
        "status":  "error",
        "message": "Challenge not solved"
    }

Note: this API returns HTTP 200 for both success and error. The
`status` field inside the JSON is what distinguishes them.

Design:
    - `transport` is injected so tests never hit the network.
    - `is_configured()` returns True as long as the base URL is set —
      Byparr needs no API key. It runs on localhost.
    - Cookie injection is supported via `route.session_cookies`.
    - We detect *unsolved* challenges (Byparr reporting "ok" but the
      HTML still containing Turnstile markers) and treat them as
      blocked, so the manager can fall through to the next tier.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Protocol

from src.network.block_codes import (
    BlockKind, classify_http_status, classify_provider_error,
)
from src.network.types import (
    FetchResult, NetworkRoute, ProviderError, ProviderNotConfigured,
    ProviderName, Transport,
)


logger = logging.getLogger("network.byparr")


DEFAULT_BASE_URL = "http://localhost:8191"
DEFAULT_TIMEOUT_MS = 60_000

# Phrases that mean "we got a page back but it's still the challenge"
_UNSOLVED_MARKERS = (
    "just a moment",
    "checking your browser",
    "cf-chl-",
    "cf_chl_",
    "challenge-platform",
    "turnstile",
    "please verify you are human",
)


# ---------------------------------------------------------------------------
# Transport protocol (fake-able in tests)
# ---------------------------------------------------------------------------

class HttpTransport(Protocol):
    """Minimal HTTP surface the provider needs. Fake in tests."""
    async def post_json(
        self, url: str, body: dict, timeout: float,
    ) -> tuple[int, dict]: ...


class HttpxTransport:
    """Default transport. Lazy-imports httpx so the module loads without it."""

    async def post_json(
        self, url: str, body: dict, timeout: float,
    ) -> tuple[int, dict]:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=body)
            try:
                data = r.json() or {}
            except Exception:
                data = {}
            return r.status_code, data


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ByparrProvider:
    """
    Tier 1c — local Cloudflare solver.

    Because Byparr runs on localhost with no authentication, the only
    "configuration" is the base URL. If the process isn't running,
    requests fail with a connection error and the manager falls through
    to the next tier automatically.
    """

    name = ProviderName.BYPARR.value

    def __init__(
        self,
        base_url: Optional[str] = None,
        transport: Optional[HttpTransport] = None,
        default_timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ):
        # Distinguish "not passed" (None → check env → fall back to
        # default) from "explicitly empty" ("" → disable the provider).
        # Python's `or` treats "" as falsy, which would otherwise
        # silently override a caller's attempt to opt out.
        if base_url is None:
            env_url = os.getenv("BYPARR_URL")
            base_url = DEFAULT_BASE_URL if env_url is None else env_url
        self.base_url = base_url.rstrip("/")
        self._transport = transport or HttpxTransport()
        self.default_timeout_ms = int(default_timeout_ms)

    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        # No key needed; a URL is enough.
        return bool(self.base_url)

    # ------------------------------------------------------------------
    async def fetch(
        self,
        url: str,
        route: Optional[NetworkRoute] = None,
    ) -> FetchResult:
        if not self.is_configured():
            raise ProviderNotConfigured(self.name, "BYPARR_URL")

        route = route or NetworkRoute()
        timeout_ms = route.timeout_seconds * 1000 or self.default_timeout_ms
        # Byparr's maxTimeout is in milliseconds; give it at least 5 s
        # of headroom over the caller's own timeout.
        byparr_timeout_ms = max(5_000, timeout_ms)

        body: dict = {
            "cmd": "request.get",
            "url": url,
            "maxTimeout": byparr_timeout_ms,
        }

        # Cookie injection: Byparr expects a list of {name, value}
        cookies = route.session_cookies or {}
        if cookies:
            body["cookies"] = [
                {"name": str(k), "value": str(v)} for k, v in cookies.items()
            ]

        started = time.monotonic()
        try:
            status, payload = await self._transport.post_json(
                f"{self.base_url}/v1", body, timeout=byparr_timeout_ms / 1000,
            )
        except Exception as e:
            kind = classify_provider_error(e)
            raise ProviderError(
                self.name,
                f"transport error: {type(e).__name__}: {e}",
                retryable=kind in (
                    BlockKind.TIMEOUT, BlockKind.NETWORK_ERROR,
                ),
            ) from e

        latency_ms = int((time.monotonic() - started) * 1000)

        # Byparr returns 200 even for errors; the `status` field is what
        # matters.
        inner_status = str(payload.get("status") or "").lower()
        message = str(payload.get("message") or "")

        if status >= 400 or inner_status == "error":
            kind = classify_http_status(status) if status >= 400 else BlockKind.PROVIDER_FAILURE
            raise ProviderError(
                self.name,
                f"byparr error: {message or f'HTTP {status}'}",
                retryable=kind in (
                    BlockKind.TIMEOUT,
                    BlockKind.SERVICE_UNAVAILABLE,
                    BlockKind.PROVIDER_FAILURE,
                ),
            )

        solution = payload.get("solution") or {}
        html = str(solution.get("response") or "")
        upstream_status = solution.get("status") or 200
        try:
            upstream_status = int(upstream_status)
        except (TypeError, ValueError):
            upstream_status = 200

        # Detect "solution" that is still a challenge page
        blocked = False
        block_reason = ""
        if not html:
            blocked = True
            block_reason = "byparr returned empty html"
        else:
            lower = html[:4000].lower()
            for marker in _UNSOLVED_MARKERS:
                if marker in lower and "challenge-platform" in lower or \
                   marker == "just a moment" and marker in lower:
                    blocked = True
                    block_reason = f"byparr unsolved challenge: {marker!r}"
                    break
            # Simpler catch-all: any Cloudflare challenge page under 30 kB
            if not blocked and len(html) < 30_000:
                for marker in ("cf-chl-", "challenge-platform"):
                    if marker in html.lower():
                        blocked = True
                        block_reason = "byparr page still contains cf-chl markers"
                        break

        # Extract cookies Byparr returned (may be useful downstream)
        byparr_cookies: dict = {}
        for c in solution.get("cookies") or []:
            try:
                byparr_cookies[str(c.get("name"))] = str(c.get("value"))
            except (TypeError, AttributeError):
                continue

        metadata: dict = {}
        if message:
            metadata["byparr_message"] = message
        if byparr_cookies:
            metadata["byparr_cookies"] = byparr_cookies
        if solution.get("userAgent"):
            metadata["user_agent"] = solution["userAgent"]

        return FetchResult(
            url=url,
            html=html,
            status_code=upstream_status,
            provider=self.name,
            transport=Transport.LOCAL_SOLVER.value,
            bytes_received=len(html.encode("utf-8", errors="ignore")),
            latency_ms=latency_ms,
            blocked=blocked,
            block_reason=block_reason,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Smoke test — injectable fake transport, no network
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    class FakeTransport:
        def __init__(self, response):
            self.response = response
            self.calls: list[tuple[str, dict, float]] = []
        async def post_json(self, url, body, timeout):
            self.calls.append((url, body, timeout))
            return self.response

    def _ok_response(html="<html>solved content</html>",
                     cookies=None, status=200):
        return (200, {
            "status": "ok",
            "message": "Challenge solved!",
            "solution": {
                "url": "https://target.example/",
                "status": status,
                "response": html,
                "cookies": cookies or [
                    {"name": "cf_clearance", "value": "abc123"},
                ],
                "userAgent": "Mozilla/5.0 (Byparr)",
            },
        })

    async def run():
        # 1. Happy path
        fake = FakeTransport(_ok_response())
        p = ByparrProvider(transport=fake)
        assert p.is_configured()
        r = await p.fetch("https://target.example/", NetworkRoute())
        assert r.provider == "byparr"
        assert r.transport == "local_solver"
        assert r.status_code == 200
        assert not r.blocked
        assert "solved content" in r.html
        assert r.metadata["byparr_cookies"]["cf_clearance"] == "abc123"
        # URL is POSTed to /v1 with cmd=request.get
        url_called, body, _ = fake.calls[0]
        assert url_called.endswith("/v1")
        assert body["cmd"] == "request.get"
        assert body["url"] == "https://target.example/"
        assert "maxTimeout" in body

        # 2. Cookie injection is passed through
        fake = FakeTransport(_ok_response())
        p = ByparrProvider(transport=fake)
        route = NetworkRoute(session_cookies={"li_at": "xyz", "JSESSIONID": "abc"})
        await p.fetch("https://linkedin.example/", route)
        _, body, _ = fake.calls[0]
        assert "cookies" in body
        names = {c["name"] for c in body["cookies"]}
        assert names == {"li_at", "JSESSIONID"}

        # 3. Unsolved challenge → blocked=True (not an exception)
        unsolved_html = '<html><head><title>Just a moment...</title></head><body>cf-chl-xyz</body></html>'
        fake = FakeTransport((200, {
            "status": "ok",
            "message": "Challenge solved!",
            "solution": {"url": "x", "status": 200, "response": unsolved_html},
        }))
        p = ByparrProvider(transport=fake)
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.blocked is True
        assert "unsolved" in r.block_reason or "cf-chl" in r.block_reason

        # 4. Empty html → blocked
        fake = FakeTransport((200, {
            "status": "ok",
            "solution": {"url": "x", "status": 200, "response": ""},
        }))
        p = ByparrProvider(transport=fake)
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.blocked is True

        # 5. Provider-level error status → ProviderError
        fake = FakeTransport((200, {
            "status": "error",
            "message": "Challenge not solved",
        }))
        p = ByparrProvider(transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert "not solved" in str(e)

        # 6. Transport exception → ProviderError
        class Boom:
            async def post_json(self, *a, **k):
                raise RuntimeError("connection refused")
        p = ByparrProvider(transport=Boom())
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 7. Not configured when URL empty
        p = ByparrProvider(base_url="")
        assert not p.is_configured()
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderNotConfigured")
        except ProviderNotConfigured:
            pass

        print("Byparr provider OK.")

    asyncio.run(run())