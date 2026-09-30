"""Integration tests: _scrape_source honours NetworkManager provider choice."""
import asyncio
from unittest.mock import MagicMock

from src.assistant import _scrape_source
from src.network.manager import NetworkManager
from src.network.types import FetchResult
from src.security.trace import SecurityTrace


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeScraper:
    def __init__(self, html=""):
        self.html = html
    async def fetch_html(self, url, timeout=None):
        return self.html
    async def fetch_screenshot(self, url, timeout=None):
        return b""


class FakeExtractor:
    def __init__(self, records=None):
        self.records = records if records is not None else []
        self.last_html = None
    def extract_list(self, html, instruction):
        self.last_html = html
        return list(self.records)
    def extract_from_image(self, img, instr):
        return []


class FakeProvider:
    def __init__(self, html=""):
        self.html = html
        self.calls = 0
    def is_configured(self):
        return True
    async def fetch(self, url, route):
        self.calls += 1
        return FetchResult(
            url=url, html=self.html, status_code=200,
            provider="scraperapi", transport="provider_api",
            bytes_received=len(self.html), credits_used=5,
        )


def _good_html(body="<p>content</p>"):
    # Must exceed looks_blocked's 500 char minimum
    return "<html><body>" + body + ("x" * 1000) + "</body></html>"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trace_records_playwright_when_healthy():
    scraper = FakeScraper(_good_html())
    extractor = FakeExtractor([{"title": "A"}])
    trace = SecurityTrace()
    asyncio.run(_scrape_source(
        scraper, extractor,
        "https://93.184.216.34/x",
        "query",
        trace=trace,
        network_manager=NetworkManager(scraper, None),
    ))
    assert trace.provider_used == "playwright"
    assert trace.fallback_used is False


def test_trace_records_scraperapi_when_fallback_used():
    # Playwright returns a block page; provider returns good HTML
    scraper = FakeScraper("Please enable JavaScript to continue")
    provider = FakeProvider(html=_good_html())
    extractor = FakeExtractor([{"title": "A"}])
    trace = SecurityTrace()
    result = asyncio.run(_scrape_source(
        scraper, extractor,
        "https://93.184.216.34/x",
        "query",
        trace=trace,
        network_manager=NetworkManager(scraper, provider),
    ))
    assert trace.provider_used == "scraperapi"
    assert trace.fallback_used is True
    assert provider.calls == 1
    # Extractor saw the provider's HTML, not the blocked one
    assert "Please enable JavaScript" not in (extractor.last_html or "")
    assert result == [{"title": "A", "source_url": "https://93.184.216.34/x"}]


def test_block_reason_recorded_in_trace():
    scraper = FakeScraper("Please enable JavaScript to continue")
    extractor = FakeExtractor([])
    trace = SecurityTrace()
    asyncio.run(_scrape_source(
        scraper, extractor,
        "https://93.184.216.34/x",
        "query",
        trace=trace,
        network_manager=NetworkManager(scraper, None),
    ))
    assert trace.block_reason
    assert "javascript" in trace.block_reason.lower()


def test_no_html_returns_empty():
    scraper = FakeScraper("")
    extractor = FakeExtractor([{"title": "should not matter"}])
    result = asyncio.run(_scrape_source(
        scraper, extractor,
        "https://93.184.216.34/x",
        "query",
        network_manager=NetworkManager(scraper, None),
    ))
    assert result == []