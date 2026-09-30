"""Unit tests for the NetworkManager fallback policy (spec §15.5D)."""
import asyncio
import pytest

from src.network.manager import NetworkManager
from src.network.types import FetchResult, ProviderError


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeScraper:
    def __init__(self, html="", raises=None):
        self.html = html
        self.raises = raises
        self.calls = 0
    async def fetch_html(self, url, timeout=None):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.html


class FakeProvider:
    def __init__(self, result=None, raises=None, configured=True):
        self.result = result
        self.raises = raises
        self.configured = configured
        self.calls = 0
    def is_configured(self):
        return self.configured
    async def fetch(self, url, route):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.result


def _good_html(n=2000):
    """Realistic-looking page: title, heading, query-term content."""
    return (
        "<html><head><title>Test Page</title></head>"
        "<body><h1>Test Content</h1><p>This is a real page.</p>"
        + "x" * n
        + "</body></html>"
    )


def _good_provider_result():
    return FetchResult(
        url="https://x", html=_good_html(),
        status_code=200, provider="scraperapi",
        transport="provider_api", bytes_received=2000, credits_used=5,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_healthy_playwright_no_fallback():
    scraper = FakeScraper(_good_html())
    provider = FakeProvider()
    m = NetworkManager(scraper, provider)
    r = asyncio.run(m.fetch("https://x"))
    assert r.provider == "playwright"
    assert r.blocked is False, r.block_reason
    assert provider.calls == 0


def test_blocked_playwright_uses_provider():
    scraper = FakeScraper("Please enable JavaScript to continue")
    provider = FakeProvider(result=_good_provider_result())
    m = NetworkManager(scraper, provider)
    r = asyncio.run(m.fetch("https://x"))
    assert r.provider == "scraperapi"
    assert r.credits_used == 5
    assert r.metadata["fallback_from"] == "playwright"
    assert provider.calls == 1


def test_blocked_no_provider_configured_stays_on_playwright():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(configured=False)
    m = NetworkManager(scraper, provider)
    r = asyncio.run(m.fetch("https://x"))
    assert r.provider == "playwright"
    assert r.blocked is True
    assert "fallback_skipped" in r.metadata
    assert provider.calls == 0


def test_no_provider_object_stays_on_playwright():
    scraper = FakeScraper("Please enable JavaScript")
    m = NetworkManager(scraper, None)
    r = asyncio.run(m.fetch("https://x"))
    assert r.provider == "playwright"
    assert r.blocked is True
    assert "fallback_skipped" in r.metadata


def test_playwright_raises_provider_used():
    scraper = FakeScraper(raises=RuntimeError("dns fail"))
    provider = FakeProvider(result=_good_provider_result())
    m = NetworkManager(scraper, provider)
    r = asyncio.run(m.fetch("https://x"))
    assert r.provider == "scraperapi"
    assert "playwright_error" in r.metadata


def test_provider_raises_keeps_playwright_result():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(raises=ProviderError("scraperapi", "down"))
    m = NetworkManager(scraper, provider)
    r = asyncio.run(m.fetch("https://x"))
    assert r.provider == "playwright"
    assert r.blocked is True
    assert "fallback_error" in r.metadata


def test_provider_also_blocked_returns_provider_result():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(result=FetchResult(
        url="https://x", html="Please enable JavaScript",
        status_code=200, provider="scraperapi",
        transport="provider_api", bytes_received=26,
        credits_used=5, blocked=True, block_reason="block marker",
    ))
    m = NetworkManager(scraper, provider)
    r = asyncio.run(m.fetch("https://x"))
    assert r.provider == "scraperapi"
    assert "fallback_failed" in r.metadata


def test_metadata_records_playwright_block_reason_on_success():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(result=_good_provider_result())
    m = NetworkManager(scraper, provider)
    r = asyncio.run(m.fetch("https://x"))
    assert "playwright_block_reason" in r.metadata


def test_provider_not_called_when_playwright_healthy():
    scraper = FakeScraper(_good_html())
    provider = FakeProvider(result=_good_provider_result())
    m = NetworkManager(scraper, provider)
    asyncio.run(m.fetch("https://x"))
    assert provider.calls == 0