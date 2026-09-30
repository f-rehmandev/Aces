"""Unit tests for ScraperAPI provider (spec §15.5B)."""
import asyncio

import pytest

from src.network.scraperapi import ScraperAPIProvider, estimate_credits
from src.network.types import (
    BudgetExceeded, NetworkRoute, ProviderError, ProviderNotConfigured,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.last_url = None
        self.last_params = None
        self.closed = False
    async def get(self, url, params=None, timeout=None):
        self.last_url = url
        self.last_params = params
        return self.response
    async def aclose(self):
        self.closed = True


class FakeBudget:
    def __init__(self, allow=True):
        self.allow = allow
        self.consumed = 0
    def can_use(self, cost): return self.allow
    def consume(self, actual): self.consumed += actual


def _provider_with_client(response, **kwargs):
    fake_client = FakeClient(response)
    async def factory():
        return fake_client
    provider = ScraperAPIProvider(
        api_key="test-key", client_factory=factory, **kwargs,
    )
    return provider, fake_client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_not_configured_when_no_key():
    p = ScraperAPIProvider(api_key="")
    assert not p.is_configured()


def test_configured_with_key():
    p = ScraperAPIProvider(api_key="abc")
    assert p.is_configured()


def test_fetch_raises_when_not_configured():
    p = ScraperAPIProvider(api_key="")
    with pytest.raises(ProviderNotConfigured):
        asyncio.run(p.fetch("https://x", NetworkRoute()))


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def test_estimate_basic():
    assert estimate_credits(NetworkRoute()) == 1


def test_estimate_render():
    assert estimate_credits(NetworkRoute(options={"render": True})) == 5


def test_estimate_premium():
    assert estimate_credits(NetworkRoute(options={"premium": True})) == 25


def test_estimate_render_and_premium_takes_max():
    assert estimate_credits(NetworkRoute(options={"render": True, "premium": True})) == 25


# ---------------------------------------------------------------------------
# Successful fetch
# ---------------------------------------------------------------------------

def test_fetch_success_returns_fetch_result():
    html = "<html><body>" + "x" * 5000 + "</body></html>"
    p, client = _provider_with_client(FakeResponse(200, html))
    result = asyncio.run(p.fetch("https://shop.example/p/1", NetworkRoute()))
    assert result.status_code == 200
    assert result.bytes_received > 5000
    assert result.provider == "scraperapi"
    assert result.transport == "provider_api"
    assert result.credits_used == 1


def test_fetch_passes_url_and_key():
    p, client = _provider_with_client(FakeResponse(200, "x" * 5000))
    asyncio.run(p.fetch("https://shop.example/p/1", NetworkRoute()))
    assert client.last_params["url"] == "https://shop.example/p/1"
    assert client.last_params["api_key"] == "test-key"


def test_fetch_render_option_sent():
    p, client = _provider_with_client(FakeResponse(200, "x" * 5000))
    asyncio.run(p.fetch("https://x", NetworkRoute(options={"render": True})))
    assert client.last_params.get("render") == "true"


def test_fetch_premium_option_sent():
    p, client = _provider_with_client(FakeResponse(200, "x" * 5000))
    asyncio.run(p.fetch("https://x", NetworkRoute(options={"premium": True})))
    assert client.last_params.get("premium") == "true"


def test_fetch_country_option_sent():
    p, client = _provider_with_client(FakeResponse(200, "x" * 5000))
    asyncio.run(p.fetch("https://x", NetworkRoute(options={"country": "pk"})))
    assert client.last_params.get("country_code") == "pk"


def test_fetch_consumes_budget():
    budget = FakeBudget(allow=True)
    p, _ = _provider_with_client(FakeResponse(200, "x" * 5000),
                                  budget_tracker=budget)
    asyncio.run(p.fetch("https://x", NetworkRoute()))
    assert budget.consumed == 1


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

def test_budget_denies_fetch():
    budget = FakeBudget(allow=False)
    p, _ = _provider_with_client(FakeResponse(200, "x" * 5000),
                                  budget_tracker=budget)
    with pytest.raises(BudgetExceeded):
        asyncio.run(p.fetch("https://x", NetworkRoute()))


def test_http_error_raises_provider_error():
    p, _ = _provider_with_client(FakeResponse(500, "boom"))
    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(p.fetch("https://x", NetworkRoute()))
    assert exc_info.value.retryable is True


def test_http_404_not_retryable():
    p, _ = _provider_with_client(FakeResponse(404, "not found"))
    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(p.fetch("https://x", NetworkRoute()))
    assert exc_info.value.retryable is False


def test_block_marker_detected():
    p, _ = _provider_with_client(FakeResponse(
        200, "<html>Please enable JavaScript to continue</html>"
    ))
    result = asyncio.run(p.fetch("https://x", NetworkRoute()))
    assert result.blocked
    assert "javascript" in result.block_reason.lower()


def test_tiny_body_marked_blocked():
    p, _ = _provider_with_client(FakeResponse(200, "hi"))
    result = asyncio.run(p.fetch("https://x", NetworkRoute()))
    assert result.blocked


def test_exception_in_client_becomes_provider_error():
    class BoomClient:
        async def get(self, *a, **k):
            raise RuntimeError("dns fail")
        async def aclose(self):
            pass

    async def factory(): return BoomClient()
    p = ScraperAPIProvider(api_key="k", client_factory=factory)
    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(p.fetch("https://x", NetworkRoute()))
    assert "dns fail" in str(exc_info.value)
    assert exc_info.value.retryable is True


def test_client_closed_even_on_error():
    class BoomClient:
        def __init__(self): self.closed = False
        async def get(self, *a, **k):
            raise RuntimeError("nope")
        async def aclose(self):
            self.closed = True

    boom = BoomClient()
    async def factory(): return boom
    p = ScraperAPIProvider(api_key="k", client_factory=factory)
    try:
        asyncio.run(p.fetch("https://x", NetworkRoute()))
    except ProviderError:
        pass
    assert boom.closed is True