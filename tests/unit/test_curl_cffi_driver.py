"""
Unit tests for the curl_cffi Tier 0 provider.

Covers:
    - Happy-path fetch returns a valid FetchResult
    - Block detection fires on challenge pages
    - Custom `curl_impersonate` overrides the default
    - Session cookies pass through
    - Transport errors become ProviderError
    - ProviderNotConfigured when curl_cffi is absent and no factory injected
    - Provider registered as Tier 0 in the NetworkManager chain
    - Tier 0 is tried before Tier 1 in a full chain
    - Tier 0 is skipped when the provider is missing
    - local_only / no_local policies treat Tier 0 as local
"""
import asyncio

import pytest

from src.network.curl_cffi_driver import CurlCffiProvider
from src.network.manager import NetworkManager
from src.network.types import (
    FetchResult,
    NetworkRoute,
    ProviderError,
    ProviderNotConfigured,
    Tier,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.last_kwargs = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url, **kwargs):
        self.last_kwargs = kwargs
        return self.response


def _provider_with(response, **kwargs):
    fake = _FakeSession(response)
    provider = CurlCffiProvider(
        session_factory=lambda: fake, **kwargs,
    )
    return provider, fake


def _good_html(n: int = 2000) -> str:
    return (
        "<html><head><title>Test</title></head>"
        "<body><h1>Test</h1>" + ("x" * n) + "</body></html>"
    )


# ---------------------------------------------------------------------------
# Provider-level
# ---------------------------------------------------------------------------

def test_happy_path_fetch():
    provider, fake = _provider_with(_FakeResponse(200, _good_html()))
    result = _run(provider.fetch("https://x.example/", NetworkRoute()))
    assert result.provider == "curl_cffi"
    assert result.status_code == 200
    assert result.blocked is False
    assert result.bytes_received > 1000
    assert result.metadata["impersonate"] == "chrome"


def test_default_impersonate_sent_to_session():
    provider, fake = _provider_with(_FakeResponse(200, _good_html()))
    _run(provider.fetch("https://x.example/", NetworkRoute()))
    assert fake.last_kwargs["impersonate"] == "chrome"


def test_custom_impersonate_via_route():
    provider, fake = _provider_with(_FakeResponse(200, _good_html()))
    _run(provider.fetch(
        "https://x.example/",
        NetworkRoute(options={"curl_impersonate": "firefox133"}),
    ))
    assert fake.last_kwargs["impersonate"] == "firefox133"


def test_constructor_default_impersonate_override():
    provider, fake = _provider_with(
        _FakeResponse(200, _good_html()),
        default_impersonate="safari17_0",
    )
    _run(provider.fetch("https://x.example/", NetworkRoute()))
    assert fake.last_kwargs["impersonate"] == "safari17_0"


def test_session_cookies_passed_through():
    provider, fake = _provider_with(_FakeResponse(200, _good_html()))
    _run(provider.fetch(
        "https://x.example/",
        NetworkRoute(session_cookies={"li_at": "abc", "JSESSIONID": "xyz"}),
    ))
    assert fake.last_kwargs["cookies"] == {"li_at": "abc", "JSESSIONID": "xyz"}


def test_no_cookies_kwarg_when_empty():
    provider, fake = _provider_with(_FakeResponse(200, _good_html()))
    _run(provider.fetch("https://x.example/", NetworkRoute()))
    assert "cookies" not in fake.last_kwargs


def test_block_marker_detected():
    provider, _ = _provider_with(
        _FakeResponse(200, "Please enable JavaScript to continue"),
    )
    result = _run(provider.fetch("https://x.example/", NetworkRoute()))
    assert result.blocked is True
    assert "javascript" in result.block_reason.lower()


def test_transport_error_becomes_provider_error():
    class _BoomSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, *args, **kwargs):
            raise RuntimeError("connection refused")

    p = CurlCffiProvider(session_factory=lambda: _BoomSession())
    with pytest.raises(ProviderError) as exc_info:
        _run(p.fetch("https://x.example/", NetworkRoute()))
    assert "connection refused" in str(exc_info.value)


def test_not_configured_when_no_factory_and_no_module(monkeypatch):
    # Force the import to fail
    import builtins
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "curl_cffi" or name.startswith("curl_cffi."):
            raise ImportError("simulated: curl_cffi not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    p = CurlCffiProvider()
    assert p.is_configured() is False
    with pytest.raises(ProviderNotConfigured):
        _run(p.fetch("https://x.example/", NetworkRoute()))


# ---------------------------------------------------------------------------
# Manager wiring
# ---------------------------------------------------------------------------

class _StubProvider:
    name = "stub"

    def __init__(self, result=None, raises=None):
        self.result = result
        self.raises = raises
        self.calls = 0

    def is_configured(self):
        return True

    async def fetch(self, url, route):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.result


def _good_result():
    return FetchResult(
        url="https://x", html=_good_html(), status_code=200,
        provider="curl_cffi", transport="direct_http",
        bytes_received=2000,
    )


def test_tier_0_is_first_in_chain():
    """Tier 0 must be tried before Tier 1."""
    curl = _StubProvider(result=_good_result())
    playwright = _StubProvider()

    # Minimal scraper wrapper — NetworkManager wraps it in _PlaywrightAdapter
    class _FakeScraper:
        async def fetch_html(self, url, timeout=None):
            return ""

    m = NetworkManager(_FakeScraper(), None, curl_cffi=curl)
    # Patch the playwright adapter to a stub so we don't run real browser
    m._providers["playwright"] = playwright

    result = _run(m.fetch("https://x.example/"))
    assert result.tier_used == Tier.TIER_0.value
    assert curl.calls == 1
    assert playwright.calls == 0


def test_tier_0_falls_through_to_tier_1_when_blocked():
    curl = _StubProvider(result=FetchResult(
        url="https://x", html="Please enable JavaScript", status_code=200,
        provider="curl_cffi", transport="direct_http",
        bytes_received=26, blocked=True, block_reason="block marker",
    ))
    playwright = _StubProvider(result=FetchResult(
        url="https://x", html=_good_html(), status_code=200,
        provider="playwright", transport="browser", bytes_received=2000,
    ))

    class _FakeScraper:
        async def fetch_html(self, url, timeout=None):
            return ""

    m = NetworkManager(_FakeScraper(), None, curl_cffi=curl)
    m._providers["playwright"] = playwright

    result = _run(m.fetch("https://x.example/"))
    assert curl.calls == 1
    assert playwright.calls == 1
    assert result.tier_used == Tier.TIER_1.value


def test_tier_0_skipped_when_not_registered():
    playwright = _StubProvider(result=FetchResult(
        url="https://x", html=_good_html(), status_code=200,
        provider="playwright", transport="browser", bytes_received=2000,
    ))

    class _FakeScraper:
        async def fetch_html(self, url, timeout=None):
            return ""

    m = NetworkManager(_FakeScraper(), None)   # no curl_cffi
    m._providers["playwright"] = playwright

    result = _run(m.fetch("https://x.example/"))
    assert result.tier_used == Tier.TIER_1.value
    assert playwright.calls == 1


def test_local_only_policy_includes_tier_0():
    curl = _StubProvider(result=_good_result())
    provider = _StubProvider()

    class _FakeScraper:
        async def fetch_html(self, url, timeout=None):
            return ""

    m = NetworkManager(_FakeScraper(), provider, curl_cffi=curl)
    _run(m.fetch(
        "https://x.example/",
        route=NetworkRoute(tier_policy="local_only"),
    ))
    # Tier 0 is local, so it must still be tried.
    assert curl.calls == 1
    # Tier 2 provider must be skipped.
    assert provider.calls == 0


def test_no_local_policy_skips_tier_0_and_tier_1():
    curl = _StubProvider(result=_good_result())
    provider = _StubProvider(result=FetchResult(
        url="https://x", html=_good_html(), status_code=200,
        provider="scraperapi", transport="provider_api",
        bytes_received=2000,
    ))

    class _FakeScraper:
        async def fetch_html(self, url, timeout=None):
            return ""

    m = NetworkManager(_FakeScraper(), provider, curl_cffi=curl)
    result = _run(m.fetch(
        "https://x.example/",
        route=NetworkRoute(tier_policy="no_local"),
    ))
    assert curl.calls == 0
    assert provider.calls == 1
    assert result.tier_used == Tier.TIER_2.value


def test_max_tier_tier_0_allows_only_tier_0():
    curl = _StubProvider(result=_good_result())
    provider = _StubProvider()

    class _FakeScraper:
        async def fetch_html(self, url, timeout=None):
            return ""

    m = NetworkManager(_FakeScraper(), provider, curl_cffi=curl)
    _run(m.fetch(
        "https://x.example/",
        route=NetworkRoute(max_tier="tier_0"),
    ))
    assert curl.calls == 1
    assert provider.calls == 0