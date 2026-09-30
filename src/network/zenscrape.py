"""
Zenscrape provider — Tier 3d.

Free tier: 1,000 API credits/month (permanent free tier, not just a
trial). Smallest of the Tier 3 providers, which is why it sits after
ZenRows.

API docs: https://zenscrape.com/
Endpoint: https://api.zenscrape.com/v2/
Auth:     `apikey` header (some accounts use query param — we use header)

Key request params:
    url            — target URL (required)
    render         — "true" enables JS rendering
    premium        — "true" routes through residential proxies
    country        — ISO-2 country code
    location       — alias for country in some docs
    timeout        — seconds

Response on success: raw HTML.
Response on error: JSON `{"status": "error", "message": "..."}` with a
4xx/5xx status.
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


logger = logging.getLogger("network.zenscrape")


DEFAULT_ENDPOINT = "https://app.zenscrape.com/api/v1/get"


class HttpTransport(Protocol):
    async def get_html(
        self, url: str, params: dict, headers: dict, timeout: float,
    ) -> tuple[int, str]: ...


class HttpxTransport:
    async def get_html(
        self, url: str, params: dict, headers: dict, timeout: float,
    ) -> tuple[int, str]:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, params=params, headers=headers)
            return r.status_code, r.text or ""


class ZenscrapeProvider:
    """Tier 3d — Zenscrape."""

    name = ProviderName.ZENSCRAPE.value

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        transport: Optional[HttpTransport] = None,
    ):
        if api_key is None:
            api_key = os.getenv("ZENSCRAPE_API_KEY", "")
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
            raise ProviderNotConfigured(self.name, "ZENSCRAPE_API_KEY")

        route = route or NetworkRoute()
        options = route.options or {}

        params: dict = {
            "url": url,
            "timeout": int(route.timeout_seconds),
        }

        # Zenscrape rejects render without premium:
        #   "The render_js option must be used together with premium_proxy."
        # So if the caller asks for JS rendering, we must also route
        # through premium residential proxies. Cost: 25 credits/request.
        want_render = bool(options.get("render", False))
        want_premium = bool(options.get("premium", False)) or want_render

        if want_render:
            params["render"] = "true"
        if want_premium:
            params["premium"] = "true"
        if options.get("country"):
            params["country"] = str(options["country"]).lower()

        headers = {"apikey": self.api_key, "Accept": "text/html"}

        started = time.monotonic()
        try:
            status, body = await self._transport.get_html(
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
            raise ProviderError(
                self.name,
                f"HTTP {status} from Zenscrape: {body[:200]}",
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
# Smoke test
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

    async def run():
        # 1. Not configured
        p = ZenscrapeProvider(api_key="")
        assert not p.is_configured()
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderNotConfigured")
        except ProviderNotConfigured:
            pass

        # 2. Happy path
        fake = FakeTransport((200, "<html><body>" + "s" * 1500 + "</body></html>"))
        p = ZenscrapeProvider(api_key="k", transport=fake)
        assert p.is_configured()
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.provider == "zenscrape"
        assert r.status_code == 200
        assert not r.blocked
        url_called, params, headers, _ = fake.calls[0]
        assert url_called == DEFAULT_ENDPOINT
        assert params["url"] == "https://x.example/"
        assert params["render"] == "true"
        assert headers["apikey"] == "k"

        # 3. Render off
        fake = FakeTransport((200, "ok" * 1000))
        p = ZenscrapeProvider(api_key="k", transport=fake)
        await p.fetch("https://x.example/",
                      NetworkRoute(options={"render": False}))
        _, params, _, _ = fake.calls[0]
        assert "render" not in params

        # 4. Premium
        fake = FakeTransport((200, "ok" * 1000))
        p = ZenscrapeProvider(api_key="k", transport=fake)
        await p.fetch("https://x.example/",
                      NetworkRoute(options={"premium": True}))
        _, params, _, _ = fake.calls[0]
        assert params["premium"] == "true"

        # 5. Country
        fake = FakeTransport((200, "ok" * 1000))
        p = ZenscrapeProvider(api_key="k", transport=fake)
        await p.fetch("https://x.example/",
                      NetworkRoute(options={"country": "PK"}))
        _, params, _, _ = fake.calls[0]
        assert params["country"] == "pk"

        # 6. 429 → ProviderError, retryable
        fake = FakeTransport((429, "quota"))
        p = ZenscrapeProvider(api_key="k", transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 7. Transport exception
        class Boom:
            async def get_html(self, *a, **k):
                raise RuntimeError("tls handshake failed")
        p = ZenscrapeProvider(api_key="k", transport=Boom())
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 8. Block marker
        fake = FakeTransport((200, "Please enable JavaScript to continue"))
        p = ZenscrapeProvider(api_key="k", transport=fake)
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.blocked is True

        print("Zenscrape provider OK.")

    asyncio.run(run())