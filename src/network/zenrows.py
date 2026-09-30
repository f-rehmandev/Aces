"""
ZenRows provider — Tier 3c.

Free tier: 5,000 credits/month. Note the multiplier system: standard
requests cost 1 credit, JS rendering costs 5, premium proxies cost 10,
and both together cost 25. So real protected requests are ~200/month.

Paid-per-success: ZenRows does NOT charge for failed or retried
requests. That's a real advantage over the other providers.

API docs: https://docs.zenrows.com/
Endpoint: https://api.zenrows.com/v1/
Auth:     `apikey` query parameter

Key request params:
    url            — target URL (required)
    apikey         — auth
    js_render      — "true" enables headless Chrome
    premium_proxy  — "true" routes through residential proxies
    proxy_country  — ISO-2 country code
    wait           — milliseconds to wait before returning HTML

Response on success: raw HTML.
Response on error: JSON error envelope. 4xx for auth/bad request;
429 for rate limit; 5xx for upstream.
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


logger = logging.getLogger("network.zenrows")


DEFAULT_ENDPOINT = "https://api.zenrows.com/v1/"


class HttpTransport(Protocol):
    async def get_html(
        self, url: str, params: dict, timeout: float,
    ) -> tuple[int, str]: ...


class HttpxTransport:
    async def get_html(
        self, url: str, params: dict, timeout: float,
    ) -> tuple[int, str]:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, params=params)
            return r.status_code, r.text or ""


def estimate_credits(route: Optional[NetworkRoute]) -> int:
    """
    ZenRows's multiplier model: 1 standard, 5 with JS, 10 with premium
    proxy, 25 with both. Used for budget pre-flight checks.
    """
    if route is None:
        return 1
    opts = route.options or {}
    credits = 1
    if opts.get("render", True):
        credits = max(credits, 5)
    if opts.get("premium"):
        credits = max(credits, 10)
    if opts.get("render", True) and opts.get("premium"):
        credits = 25
    return credits


class ZenRowsProvider:
    """Tier 3c — ZenRows."""

    name = ProviderName.ZENROWS.value

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        transport: Optional[HttpTransport] = None,
    ):
        if api_key is None:
            api_key = os.getenv("ZENROWS_API_KEY", "")
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
            raise ProviderNotConfigured(self.name, "ZENROWS_API_KEY")

        route = route or NetworkRoute()
        options = route.options or {}
        credits = estimate_credits(route)

        params: dict = {
            "url": url,
            "apikey": self.api_key,
        }

        if options.get("render", True):
            params["js_render"] = "true"
        if options.get("premium"):
            params["premium_proxy"] = "true"
        if options.get("country"):
            params["proxy_country"] = str(options["country"]).lower()
        if options.get("wait_ms"):
            params["wait"] = int(options["wait_ms"])

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
                f"HTTP {status} from ZenRows: {body[:200]}",
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
            credits_used=credits,
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
            self.calls: list[tuple[str, dict, float]] = []
        async def get_html(self, url, params, timeout):
            self.calls.append((url, params, timeout))
            return self.response

    # Credit model
    assert estimate_credits(None) == 1
    assert estimate_credits(NetworkRoute(options={"render": False})) == 1
    assert estimate_credits(NetworkRoute(options={"render": True})) == 5
    assert estimate_credits(NetworkRoute(options={"premium": True,
                                                    "render": False})) == 10
    assert estimate_credits(NetworkRoute(options={"premium": True,
                                                    "render": True})) == 25

    async def run():
        # 1. Not configured
        p = ZenRowsProvider(api_key="")
        assert not p.is_configured()
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderNotConfigured")
        except ProviderNotConfigured:
            pass

        # 2. Happy path
        fake = FakeTransport((200, "<html><body>" + "z" * 2000 + "</body></html>"))
        p = ZenRowsProvider(api_key="k", transport=fake)
        assert p.is_configured()
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.provider == "zenrows"
        assert r.status_code == 200
        assert not r.blocked
        assert r.credits_used == 5      # default render=True
        url_called, params, _ = fake.calls[0]
        assert url_called == DEFAULT_ENDPOINT
        assert params["url"] == "https://x.example/"
        assert params["apikey"] == "k"
        assert params["js_render"] == "true"

        # 3. Render off → 1 credit
        fake = FakeTransport((200, "ok" * 1000))
        p = ZenRowsProvider(api_key="k", transport=fake)
        r = await p.fetch("https://x.example/",
                          NetworkRoute(options={"render": False}))
        assert r.credits_used == 1
        _, params, _ = fake.calls[0]
        assert "js_render" not in params

        # 4. Premium
        fake = FakeTransport((200, "ok" * 1000))
        p = ZenRowsProvider(api_key="k", transport=fake)
        r = await p.fetch("https://x.example/",
                          NetworkRoute(options={"premium": True,
                                                 "render": False}))
        assert r.credits_used == 10
        _, params, _ = fake.calls[0]
        assert params["premium_proxy"] == "true"

        # 5. Premium + render = 25
        fake = FakeTransport((200, "ok" * 1000))
        p = ZenRowsProvider(api_key="k", transport=fake)
        r = await p.fetch("https://x.example/",
                          NetworkRoute(options={"premium": True}))
        assert r.credits_used == 25

        # 6. Country + wait
        fake = FakeTransport((200, "ok" * 1000))
        p = ZenRowsProvider(api_key="k", transport=fake)
        await p.fetch("https://x.example/",
                      NetworkRoute(options={"country": "PK",
                                             "wait_ms": 3000}))
        _, params, _ = fake.calls[0]
        assert params["proxy_country"] == "pk"
        assert params["wait"] == 3000

        # 7. 402 → retryable (payment required — quota exhausted)
        fake = FakeTransport((402, "out of credits"))
        p = ZenRowsProvider(api_key="k", transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 8. 429 → retryable
        fake = FakeTransport((429, "rate limit"))
        p = ZenRowsProvider(api_key="k", transport=fake)
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 9. Transport exception
        class Boom:
            async def get_html(self, *a, **k):
                raise RuntimeError("connection timeout")
        p = ZenRowsProvider(api_key="k", transport=Boom())
        try:
            await p.fetch("https://x.example/", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        # 10. Block marker
        fake = FakeTransport((200, "Checking your browser before accessing"))
        p = ZenRowsProvider(api_key="k", transport=fake)
        r = await p.fetch("https://x.example/", NetworkRoute())
        assert r.blocked is True

        print("ZenRows provider OK.")

    asyncio.run(run())