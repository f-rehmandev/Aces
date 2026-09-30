"""Tests that _scrape_source never silently swallows stage failures."""
import asyncio
import logging

from src.assistant import _scrape_source
from src.network.manager import NetworkManager


class FakeScraper:
    def __init__(self, html):
        self.html = html
    async def fetch_html(self, url, timeout=None):
        return self.html
    async def fetch_screenshot(self, url, timeout=None):
        return b""


class FakeExtractor:
    def __init__(self, records=None):
        self.records = records or []
    def extract_list(self, html, instruction):
        return list(self.records)
    def extract_from_image(self, img, instr):
        return []


GOOD_HTML = "<html><body>" + ("x" * 5000) + "</body></html>"


def test_sanitizer_exception_logged_not_swallowed(caplog, monkeypatch):
    """If the sanitizer raises, the caller sees a WARNING with the URL."""
    import src.assistant as assistant_mod

    class BoomSanitizer:
        def sanitize(self, html):
            raise RuntimeError("simulated sanitizer crash")

    monkeypatch.setattr(assistant_mod, "_dom_sanitizer", BoomSanitizer())

    caplog.set_level(logging.WARNING)
    result = asyncio.run(_scrape_source(
        FakeScraper(GOOD_HTML), FakeExtractor([{"title": "nope"}]),
        "https://93.184.216.34/x",
        "query",
        network_manager=NetworkManager(FakeScraper(GOOD_HTML), None),
    ))
    assert result == []
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "DOM sanitizer failed" in text
    assert "simulated sanitizer crash" in text


def test_honeypot_exception_logged_not_swallowed(caplog, monkeypatch):
    import src.assistant as assistant_mod

    class BoomHoneypot:
        def detect_in_html(self, html):
            raise RuntimeError("simulated honeypot crash")

    monkeypatch.setattr(assistant_mod, "_honeypot_detector", BoomHoneypot())

    caplog.set_level(logging.WARNING)
    # Should still reach Rung 2 (LLM) successfully since honeypot failure
    # must not kill the pipeline
    asyncio.run(_scrape_source(
        FakeScraper(GOOD_HTML), FakeExtractor([{"title": "A"}]),
        "https://93.184.216.34/x",
        "query",
        network_manager=NetworkManager(FakeScraper(GOOD_HTML), None),
    ))
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Honeypot detector failed" in text


def test_rung1_exception_logged_not_swallowed(caplog, monkeypatch):
    import src.assistant as assistant_mod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated rung1 crash")

    monkeypatch.setattr(assistant_mod, "try_deterministic_extraction", boom)

    caplog.set_level(logging.WARNING)
    # Should still fall through to Rung 2
    result = asyncio.run(_scrape_source(
        FakeScraper(GOOD_HTML), FakeExtractor([{"title": "B"}]),
        "https://93.184.216.34/x",
        "query",
        network_manager=NetworkManager(FakeScraper(GOOD_HTML), None),
    ))
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Rung 1 failed" in text
    assert result and result[0]["title"] == "B"