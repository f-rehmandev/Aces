"""
Unit tests for the Camoufox Tier 1b provider.

Covers:
    - Happy-path fetch returns a valid FetchResult
    - Empty response with error → ProviderError
    - Challenge detection → blocked=True, no exception
    - Cookie + timeout pass-through to the injected fetch_fn
    - ProviderNotConfigured when camoufox isn't installed
    - Provider registered as Tier 1b in the NetworkManager chain
    - Chain order: curl_cffi → seleniumbase → camoufox → playwright
"""
import asyncio

import pytest

from src.network.camoufox import CamoufoxProvider
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
# Provider-level
# ---------------------------------------------------------------------------

def test_happy_path_fetch():
    async def fake(url, cookies, timeout, headless):
        return "<html><body>" + ("x" * 1000) + "</body></html>", 200, ""

    p = CamoufoxProvider(fetch_fn=fake)
    assert p.is_configured()
    r = _run(p.fetch("https://x.example/", NetworkRoute()))
    assert r.provider == "camoufox"
    assert r.transport == "browser"
    assert r.status_code == 200
    assert r.blocked is False
    assert r.bytes_received > 1000


def test_empty_response_with_error_raises_provider_error():
    async def fake(url, cookies, timeout, headless):
        return "", 0, "RuntimeError: chrome crashed"

    p = CamoufoxProvider(fetch_fn=fake)
    with pytest.raises(ProviderError) as exc_info:
        _run(p.fetch("https://x.example/", NetworkRoute()))
    assert "chrome crashed" in str(exc_info.value)


def test_empty_response_without_error_is_blocked():
    async def fake(url, cookies, timeout, headless):
        return "", 200, ""

    p = CamoufoxProvider(fetch_fn=fake)
    r = _run(p.fetch("https://x.example/", NetworkRoute()))
    assert r.blocked is True
    assert "empty html" in r.block_reason


def test_challenge_page_detected_as_blocked():
    async def fake(url, cookies, timeout, headless):
        return (
            '<html><title>Just a moment...</title>'
            '<body>cf-chl-xyz</body></html>',
            200, "",
        )

    p = CamoufoxProvider(fetch_fn=fake)
    r = _run(p.fetch("https://x.example/", NetworkRoute()))
    assert r.blocked is True
    assert "unsolved challenge" in r.block_reason


def test_cookies_and_timeout_pass_through():
    captured = {}

    async def fake(url, cookies, timeout, headless):
        captured.update({
            "url": url, "cookies": dict(cookies),
            "timeout": timeout, "headless": headless,
        })
        return "<html>ok</html>", 200, ""

    p = CamoufoxProvider(headless=True, fetch_fn=fake)
    _run(p.fetch(
        "https://linkedin.example/",
        NetworkRoute(
            session_cookies={"li_at": "abc"},
            timeout_seconds=45,
        ),
    ))
    assert captured["url"] == "https://linkedin.example/"
    assert captured["cookies"] == {"li_at": "abc"}
    assert captured["timeout"] == 45
    assert captured["headless"] is True


def test_headless_false_respected():
    captured = {}

    async def fake(url, cookies, timeout, headless):
        captured["headless"] = headless
        return "<html>ok</html>", 200, ""

    p = CamoufoxProvider(headless=False, fetch_fn=fake)
    _run(p.fetch("https://x.example/", NetworkRoute()))
    assert captured["headless"] is False


def test_transport_exception_becomes_provider_error():
    async def boom(url, cookies, timeout, headless):
        raise RuntimeError("network unreachable")

    p = CamoufoxProvider(fetch_fn=boom)
    with pytest.raises(ProviderError) as exc_info:
        _run(p.fetch("https://x.example/", NetworkRoute()))
    assert "network unreachable" in str(exc_info.value)


def test_not_configured_when_camoufox_not_installed(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "camoufox" or name.startswith("camoufox."):
            raise ImportError("simulated: camoufox not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    p = CamoufoxProvider()
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


def _good_html(n=2000):
    return "<html><body>" + ("x" * n) + "</body></html>"


def test_camoufox_is_tier_1b_in_chain():
    """The chain order is: curl_cffi → seleniumbase → camoufox → playwright."""
    curl = _StubProvider()
    sb = _StubProvider()
    camoufox = _StubProvider()
    playwright = _StubProvider()

    class _FakeScraper:
        async def fetch_html(self, url, timeout=None):
            return ""

    m = NetworkManager(
        _FakeScraper(), None,
        curl_cffi=curl, seleniumbase=sb, camoufox=camoufox,
    )
    m._providers["playwright"] = playwright

    chain, _ = m._build_chain(NetworkRoute())
    provider_names = [name for _, name, _ in chain]
    # All four should be present, in that order
    assert provider_names.index("curl_cffi") < provider_names.index("seleniumbase")
    assert provider_names.index("seleniumbase") < provider_names.index("camoufox")
    assert provider_names.index("camoufox") < provider_names.index("playwright")