"""
WebScrapingAPI provider — Tier 3b.

Free tier: 1,000 credits/month (after a 7-day trial with full access).

API docs: https://docs.webscrapingapi.com/
Endpoint: https://api.webscrapingapi.com/v1
Auth:     `api_key` query parameter

Key request params:
    url            — target URL (required)
    api_key        — auth
    render_js      — "1" enables headless Chrome rendering
    proxy_type     — "datacenter" (default) | "residential"
    proxy_country  — ISO-2 country code
    timeout        — max ms before giving up

Response on success: raw HTML.
Response on error:   JSON `{"error": {"code": ..., "type": ..., "message": ...}}`.

Design:
    - Same shape as scrapingant.py: injectable transport, block
      detection via `looks_blocked`, errors classified through
      `block_codes`.
    - 401/403 responses mean a bad key; classify as
      PROVIDER_FAILURE-adjacent so the manager falls through.
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


logger = logging.getLogger("network.webscrapingapi")


DEFAULT_ENDPOINT = "https://api.webscrapingapi.com/v2"


class HttpTransport(Protocol):
    async def get_html(
        self, url: str, params: dict, timeout: float,
    ) -> tuple[int, str]: ...


class HttpxTransport:
    """Default transport. Lazy-imports httpx."""

    async def get_html(
        self, url: str, params: dict, timeout: float,
    ) -> tuple[int, str]:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, params=params)
            return r.status_code, r.text or ""


class WebScrapingAPIProvider:
    """Tier 3b — WebScrapingAPI."""

    name = ProviderName.WEBSCRAPINGAPI.value

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        transport: Optional[HttpTransport] = None,
    ):
        if api_key is None:
            api_key = os.getenv("WEBSCRAPING_API_KEY", "")
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
            raise ProviderNotConfigured(self.name, "WEBSCRAPING_API_KEY")

        route = route or NetworkRoute()
        options = route.options or {}

        params: dict = {
            "url": url,
            "api_key": self.api_key,
            "timeout": int(route.timeout_seconds * 1000),
        }

        # The Free plan ("Web Scraping API") does not include JS
        # rendering — that's the separate "Browser API" product. Sending
        # render_js=1 on a Free plan returns a 403 "not allowed to use
        # this service". Default render to False; a caller with the
        # Browser API (or a paid plan) can opt in via
        # options={"render": True}. For JS-heavy pages, the tier chain
        # will naturally fall through to ZenRows / Zenscrape / Apify.
        if options.get("render", False):
            params["render_js"] = "1"

        if options.get("premium"):
            params["proxy_type"] = "residential"
        if options.get("country"):
            params["proxy_country"] = str(options["country"]).lower()

        started = time.monotonic()
        try:
            status, body = await self._transport.get_html(
                self.endpoint, params, route.timeout_seconds,
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
            raise ProviderError(
                self.name,
                f"HTTP {status} from WebScrapingAPI: {body[:200]}",
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
            credits_used=1,
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
            self.calls: list[tuple[str, dict, float]] = []
        async def get_html(self, url, params, timeout):
            self.calls.append((url, params, timeout))
            return self.response

    async def run():
        # 1. Not configured
        p = WebScrapingAPIProvider(api_key="")
        assert not p.is_configured()
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderNotConfigured")
        except ProviderNotConfigured:
            pass

        # 2. Happy path
        fake = FakeTransport((200, "<html><body>" + "y" * 1500 + "</body></html>"))
        p = WebScrapingAPIProvider(api_key="k", transport=fake)
        assert p.is_configured()
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.provider == "webscrapingapi"
        assert r.status_code == 200
        assert not r.blocked
        url_called, params, _ = fake.calls[0]
        assert url_called == DEFAULT_ENDPOINT
        assert params["url"] == "https://x.example/"
        assert params["api_key"] == "k"
        assert params["render_js"] == "1"

        # 3. Render off
        fake = FakeTransport((200, "ok" * 1000))
        p = WebScrapingAPIProvider(api_key="k", transport=fake)
        await p.fetch("https://x.example/",
                     NetworkRoute(options={"render": False}))
        _, params, _ = fake.calls[0]
        assert "render_js" not in params

        # 4. Premium
        fake = FakeTransport((200, "ok" * 1000))
        p = WebScrapingAPIProvider(api_key="k", transport=fake)
        await p.fetch("https://x.example/",
                     NetworkRoute(options={"premium": True}))
        _, params, _ = fake.calls[0]
        assert params["proxy_type"] == "residential"

        # 5. Country
        fake = FakeTransport((200, "ok" * 1000))
        p = WebScrapingAPIProvider(api_key="k", transport=fake)
        await p.fetch("https://x.example/",
                     NetworkRoute(options={"country": "PK"}))
        _, params, _ = fake.calls[0]
        assert params["proxy_country"] == "pk"

        # 6. 429 → ProviderError, retryable
        fake = FakeTransport((429, "quota"))
        p = WebScrapingAPIProvider(api_key="k", transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 7. 401 → ProviderError, retryable
        fake = FakeTransport((401, "bad key"))
        p = WebScrapingAPIProvider(api_key="k", transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 8. Transport exception
        class Boom:
            async def get_html(self, *a, **k):
                raise RuntimeError("dns failure")
        p = WebScrapingAPIProvider(api_key="k", transport=Boom())
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 9. Block marker
        fake = FakeTransport((200, "Please enable JavaScript to continue"))
        p = WebScrapingAPIProvider(api_key="k", transport=fake)
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.blocked is True

        print("WebScrapingAPI provider OK.")

    asyncio.run(run())