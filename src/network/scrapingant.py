"""
ScrapingAnt provider — Tier 3a.

Free tier: 10,000 API credits/month — the most generous of the Tier 3
providers we integrate, which is why it sits first in the chain.

API docs: https://docs.scrapingant.com/
Endpoint: https://api.scrapingant.com/v2/general
Auth:     `x-api-key` header (or `?x-api-key=` query param)

Key request params:
    url          — target URL (required)
    browser      — "true" enables headless Chrome rendering
    proxy_type   — "datacenter" (default) | "residential"
    proxy_country — ISO-2 country code for the egress IP
    return_page_source — "true" returns HTML instead of default JSON
                         wrapper

Response on success: raw HTML (when return_page_source=true) or a JSON
envelope with `content`. We always request HTML directly.

Response on error: JSON with `code` and `message`. Common codes:
    401 / 403  → authentication problems (bad or missing key)
    422        → bad request (invalid URL or param)
    429        → rate limit / quota exceeded
    500 / 503  → upstream error

Design:
    - Transport is injectable (matches byparr.py, scraperapi.py).
    - `is_configured()` false when SCRAPINGANT_API_KEY is missing.
    - Block detection delegates to `src.network.types.looks_blocked`.
    - Errors are classified through `block_codes` so the manager can
      decide whether to fall through.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Protocol

from src.network.block_codes import (
    BlockKind, classify_http_status, classify_provider_error,
    should_fall_through,
)
from src.network.types import (
    FetchResult, NetworkRoute, ProviderError, ProviderNotConfigured,
    ProviderName, Transport, looks_blocked,
)


logger = logging.getLogger("network.scrapingant")


DEFAULT_ENDPOINT = "https://api.scrapingant.com/v2/general"


# ---------------------------------------------------------------------------
# Transport (fake-able)
# ---------------------------------------------------------------------------

class HttpTransport(Protocol):
    async def get_html(
        self, url: str, params: dict, headers: dict, timeout: float,
    ) -> tuple[int, str, dict]: ...


class HttpxTransport:
    """Default transport. Lazy-imports httpx."""

    async def get_html(
        self, url: str, params: dict, headers: dict, timeout: float,
    ) -> tuple[int, str, dict]:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, params=params, headers=headers)
            return r.status_code, r.text or "", dict(r.headers)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ScrapingAntProvider:
    """Tier 3a — ScrapingAnt."""

    name = ProviderName.SCRAPINGANT.value

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        transport: Optional[HttpTransport] = None,
    ):
        if api_key is None:
            api_key = os.getenv("SCRAPINGANT_API_KEY", "")
        self.api_key = api_key
        self.endpoint = endpoint
        self._transport = transport or HttpxTransport()

    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    async def fetch(
        self,
        url: str,
        route: Optional[NetworkRoute] = None,
    ) -> FetchResult:
        if not self.is_configured():
            raise ProviderNotConfigured(self.name, "SCRAPINGANT_API_KEY")

        route = route or NetworkRoute()
        options = route.options or {}

        # ScrapingAnt returns raw HTML when return_page_source=true,
        # otherwise a JSON envelope. We always want HTML.
        params: dict = {
            "url": url,
            "return_page_source": "true",
        }

        # Browser rendering: default ON unless caller says otherwise.
        # ScrapingAnt's "browser" flag renders with headless Chrome.
        if options.get("render", True):
            params["browser"] = "true"

        # Proxy selection
        if options.get("premium"):
            params["proxy_type"] = "residential"
        if options.get("country"):
            params["proxy_country"] = str(options["country"]).lower()

        headers = {"x-api-key": self.api_key, "Accept": "text/html"}

        started = time.monotonic()
        try:
            status, body, resp_headers = await self._transport.get_html(
                self.endpoint, params, headers, route.timeout_seconds,
            )
        except Exception as e:
            kind = classify_provider_error(e)
            raise ProviderError(
                self.name,
                f"transport error: {type(e).__name__}: {e}",
                retryable=should_fall_through(kind),
            ) from e

        latency_ms = int((time.monotonic() - started) * 1000)

        if status >= 400:
            kind = classify_http_status(status)
            # Some 4xx here mean an application-level error — treat all
            # as fall-through candidates since another provider may
            # succeed where this one failed.
            raise ProviderError(
                self.name,
                f"HTTP {status} from ScrapingAnt: {body[:200]}",
                retryable=should_fall_through(kind),
            )

        blocked, reason = looks_blocked(body)

        return FetchResult(
            url=url,
            html=body,
            status_code=status,
            provider=self.name,
            transport=Transport.PROVIDER_API.value,
            bytes_received=len(body.encode("utf-8", errors="ignore")),
            credits_used=1,   # ScrapingAnt charges 1 credit per standard call
            latency_ms=latency_ms,
            blocked=blocked,
            block_reason=reason,
        )


# ---------------------------------------------------------------------------
# Smoke test — injectable fake transport, no network
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    class FakeTransport:
        def __init__(self, response):
            self.response = response
            self.calls: list[tuple[str, dict, dict, float]] = []
        async def get_html(self, url, params, headers, timeout):
            self.calls.append((url, params, headers, timeout))
            return self.response

    def _ok(body="<html><body>" + "x" * 1000 + "</body></html>"):
        return (200, body, {})

    async def run():
        # 1. Not configured when key absent
        p = ScrapingAntProvider(api_key="")
        assert not p.is_configured()
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderNotConfigured")
        except ProviderNotConfigured:
            pass

        # 2. Happy path
        fake = FakeTransport(_ok())
        p = ScrapingAntProvider(api_key="test-key", transport=fake)
        assert p.is_configured()
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.provider == "scrapingant"
        assert r.transport == "provider_api"
        assert r.status_code == 200
        assert not r.blocked
        assert r.credits_used == 1
        # URL, params, headers were sent
        url_called, params, headers, _ = fake.calls[0]
        assert url_called == DEFAULT_ENDPOINT
        assert params["url"] == "https://x.example/"
        assert params["return_page_source"] == "true"
        assert params["browser"] == "true"
        assert headers["x-api-key"] == "test-key"

        # 3. Render option off
        fake = FakeTransport(_ok())
        p = ScrapingAntProvider(api_key="k", transport=fake)
        await p.fetch("https://x.example/", NetworkRoute(options={"render": False}))
        _, params, _, _ = fake.calls[0]
        assert "browser" not in params

        # 4. Premium option switches to residential proxy
        fake = FakeTransport(_ok())
        p = ScrapingAntProvider(api_key="k", transport=fake)
        await p.fetch("https://x.example/", NetworkRoute(options={"premium": True}))
        _, params, _, _ = fake.calls[0]
        assert params["proxy_type"] == "residential"

        # 5. Country passthrough
        fake = FakeTransport(_ok())
        p = ScrapingAntProvider(api_key="k", transport=fake)
        await p.fetch("https://x.example/",
                     NetworkRoute(options={"country": "PK"}))
        _, params, _, _ = fake.calls[0]
        assert params["proxy_country"] == "pk"

        # 6. HTTP error becomes ProviderError
        fake = FakeTransport((429, "rate limited", {}))
        p = ScrapingAntProvider(api_key="k", transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True   # 429 is a fall-through candidate

        # 7. 401 → retryable (another provider may have a valid key)
        fake = FakeTransport((401, "unauthorized", {}))
        p = ScrapingAntProvider(api_key="k", transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 8. Transport exception becomes ProviderError
        class Boom:
            async def get_html(self, *a, **k):
                raise RuntimeError("connection refused")
        p = ScrapingAntProvider(api_key="k", transport=Boom())
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 9. Block marker detected
        fake = FakeTransport((200, "Please enable JavaScript to continue", {}))
        p = ScrapingAntProvider(api_key="k", transport=fake)
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.blocked is True

        print("ScrapingAnt provider OK.")

    asyncio.run(run())