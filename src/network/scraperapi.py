"""
ScraperAPI provider — spec §15.5B.

ScraperAPI is a paid residential-proxy + rendering service. ACES uses it
as a fallback transport for sites where Playwright with stealth is not
enough (Walmart, CVS, Amazon, etc.).

Design notes:
    - We fetch the HTML, then hand it to our own extraction pipeline.
      ScraperAPI never sees or produces our data model.
    - The provider is *strictly optional*. If SCRAPERAPI_KEY is unset,
      `is_configured()` returns False and the manager skips it.
    - We track credits used per call so the budget gate can enforce limits.
    - Uses httpx (already a transitive dependency via Supabase).
"""

from __future__ import annotations
import logging
import os
import time
from typing import Optional

from src.network.block_codes import (
    BlockKind, classify_http_status, classify_provider_error,
    should_fall_through,
)
from src.network.types import (
    BudgetExceeded, FetchResult, NetworkRoute, ProviderError,
    ProviderName, Transport, looks_blocked,
)

logger = logging.getLogger("scraperapi")


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
# ScraperAPI charges credits per request. The exact multiplier depends on
# the site and rendering options; this is a conservative default the budget
# gate uses for pre-flight checks. Actual cost is read back from the
# response when the provider reports it.

_BASE_CREDITS = 1
_RENDER_CREDITS = 5         # if we ask for JS rendering
_PREMIUM_CREDITS = 25       # if we ask for a premium (residential) proxy


def estimate_credits(route: NetworkRoute) -> int:
    credits = _BASE_CREDITS
    if route.options.get("render"):
        credits = max(credits, _RENDER_CREDITS)
    if route.options.get("premium"):
        credits = max(credits, _PREMIUM_CREDITS)
    return credits


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ScraperAPIProvider:
    """
    Concrete provider. One instance per process; safe to reuse.

    `budget_tracker` is optional. If provided, it must expose
    `can_use(cost) -> bool` and `consume(actual)` (see src/jobs/budget.py
    ScraperAPIBudgetTracker for the expected interface).
    """

    name = ProviderName.SCRAPERAPI.value

    def __init__(
        self,
        api_key: Optional[str] = None,
        http_endpoint: str = "https://api.scraperapi.com",
        budget_tracker=None,
        client_factory=None,
    ):
        # Prefer the new env var name (SCRAPER_API_KEY); fall back to
        # the old one (SCRAPERAPI_KEY) so existing .env files keep
        # working. Passing `api_key=""` explicitly still disables the
        # provider — that's the "not configured" test path.
        if api_key is None:
            api_key = (
                os.getenv("SCRAPER_API_KEY")
                or os.getenv("SCRAPERAPI_KEY")
                or ""
            )
        self.api_key = api_key
        self.http_endpoint = http_endpoint.rstrip("/")
        self.budget_tracker = budget_tracker
        # inject for tests: a callable returning an httpx.AsyncClient
        self._client_factory = client_factory
    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    async def fetch(self, url: str, route: NetworkRoute) -> FetchResult:
        if not self.is_configured():
            from src.network.types import ProviderNotConfigured
            raise ProviderNotConfigured(self.name, "SCRAPERAPI_KEY")

        cost = estimate_credits(route)
        if self.budget_tracker is not None:
            if not self.budget_tracker.can_use(cost):
                raise BudgetExceeded(self.name, needed=cost, available=0)

        # NOTE: ScraperAPI's public docs use `premium=true` for the
        # residential/paid tier. When that tier is not available on the
        # account, the request comes back as HTTP 500 (not 401/403), so
        # we detect that and fall back to a plain render request.
        params = {"api_key": self.api_key, "url": url}
        if route.options.get("render"):
            params["render"] = "true"
        if route.options.get("premium"):
            params["premium"] = "true"
        if route.options.get("country"):
            params["country_code"] = route.options["country"]

        # Remember whether premium was requested so we can retry without it
        premium_requested = bool(route.options.get("premium"))

        started = time.monotonic()
        client = await self._make_client()
        try:
            response = await client.get(
                self.http_endpoint,
                params=params,
                timeout=route.timeout_seconds,
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            status = response.status_code
            html = response.text or ""

            # ScraperAPI returns 200 with a body; non-2xx indicates an
            # upstream problem we surface as a provider error.
            if status >= 400:
                # A 500 when we asked for premium usually means the account
                # tier does not allow it. Retry once with plain render.
                if status == 500 and premium_requested and "premium" in params:
                    logger.info(
                        "ScraperAPI premium request failed (500); "
                        "retrying once without premium"
                    )
                    retry_params = {k: v for k, v in params.items() if k != "premium"}
                    response = await client.get(
                        self.http_endpoint,
                        params=retry_params,
                        timeout=route.timeout_seconds,
                    )
                    status = response.status_code
                    html = response.text or ""
                    if status >= 400:
                        kind = classify_http_status(status)
                        raise ProviderError(
                            self.name,
                            f"HTTP {status} from ScraperAPI (retry without premium also failed)",
                            retryable=should_fall_through(kind),
                        )
                else:
                    kind = classify_http_status(status)
                    raise ProviderError(
                        self.name,
                        f"HTTP {status} from ScraperAPI",
                        retryable=should_fall_through(kind),
                    )

            blocked, reason = looks_blocked(html)
            result = FetchResult(
                url=url,
                html=html,
                status_code=status,
                provider=self.name,
                transport=Transport.PROVIDER_API.value,
                bytes_received=len(html.encode("utf-8", errors="ignore")),
                credits_used=cost,
                latency_ms=latency_ms,
                blocked=blocked,
                block_reason=reason,
            )

            if self.budget_tracker is not None:
                self.budget_tracker.consume(cost)

            return result

        except ProviderError:
            raise
        except Exception as e:
            kind = classify_provider_error(e)
            raise ProviderError(
                self.name,
                f"transport error: {type(e).__name__}: {e}",
                retryable=should_fall_through(kind),
            ) from e
        finally:
            await client.aclose()

    # ------------------------------------------------------------------
    async def _make_client(self):
        if self._client_factory is not None:
            return await self._client_factory()
        import httpx
        return httpx.AsyncClient(follow_redirects=True)


# ---------------------------------------------------------------------------
# Smoke test — uses a fake client; no network
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    class FakeResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

    class FakeClient:
        def __init__(self, response):
            self.response = response
            self.last_params = None
        async def get(self, url, params=None, timeout=None):
            self.last_params = params
            return self.response
        async def aclose(self):
            pass

    class FakeBudget:
        def __init__(self, allow=True):
            self.allow = allow
            self.consumed = 0
        def can_use(self, cost): return self.allow
        def consume(self, actual): self.consumed += actual

    async def main():
        # Not configured
        p = ScraperAPIProvider(api_key="")
        assert not p.is_configured()
        try:
            await p.fetch("https://x", NetworkRoute())
            raise AssertionError("expected ProviderNotConfigured")
        except Exception as e:
            assert "not configured" in str(e).lower()

        # Success path with a real-looking HTML body
        big_html = "<html><body>" + "x" * 5000 + "</body></html>"
        fake_client = FakeClient(FakeResponse(200, big_html))
        async def client_factory():
            return fake_client
        budget = FakeBudget(allow=True)
        p = ScraperAPIProvider(
            api_key="test-key",
            client_factory=client_factory,
            budget_tracker=budget,
        )
        assert p.is_configured()

        result = await p.fetch("https://shop.example/p/1", NetworkRoute())
        assert result.status_code == 200
        assert result.bytes_received > 5000
        assert result.credits_used == 1
        assert budget.consumed == 1
        # URL and key were passed to the client
        assert fake_client.last_params["url"] == "https://shop.example/p/1"
        assert fake_client.last_params["api_key"] == "test-key"

        # render option bumps cost
        route = NetworkRoute(options={"render": True})
        assert estimate_credits(route) == 5

        # premium option bumps cost
        route = NetworkRoute(options={"premium": True})
        assert estimate_credits(route) == 25

        # budget denies
        budget2 = FakeBudget(allow=False)
        p2 = ScraperAPIProvider(
            api_key="k", client_factory=client_factory, budget_tracker=budget2,
        )
        try:
            await p2.fetch("https://x", NetworkRoute())
            raise AssertionError("expected BudgetExceeded")
        except BudgetExceeded:
            pass

        # Block marker detected
        fake_client2 = FakeClient(FakeResponse(
            200, "<html><body>Please enable JavaScript to continue</body></html>"
        ))
        async def cf2(): return fake_client2
        p3 = ScraperAPIProvider(api_key="k", client_factory=cf2)
        result = await p3.fetch("https://x", NetworkRoute())
        assert result.blocked
        assert "javascript" in result.block_reason.lower()

        # HTTP error surfaces as ProviderError
        fake_client3 = FakeClient(FakeResponse(500, "boom"))
        async def cf3(): return fake_client3
        p4 = ScraperAPIProvider(api_key="k", client_factory=cf3)
        try:
            await p4.fetch("https://x", NetworkRoute())
            raise AssertionError("expected ProviderError")
        except ProviderError as e:
            assert e.retryable is True

        print("ScraperAPI provider OK.")

    asyncio.run(main())