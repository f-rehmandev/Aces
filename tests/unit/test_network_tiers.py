"""Unit tests for the tiered NetworkManager (spec §15.5D)."""
import pytest

from src.network.manager import (
    NetworkManager,
    _PlaywrightAdapter,
    _provider_configured,
)
from src.network.types import (
    FetchResult,
    NetworkRoute,
    ProviderError,
    Tier,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _good_html(n: int = 2500) -> str:
    return (
        "<html><head><title>Test</title></head>"
        "<body><h1>Test</h1><p>real page</p>"
        + "x" * n
        + "</body></html>"
    )


def _good_result(url="https://x", provider="x"):
    return FetchResult(
        url=url, html=_good_html(), status_code=200,
        provider=provider, transport="provider_api",
        bytes_received=2500,
    )


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
    name = "fake"
    def __init__(self, result=None, raises=None, configured=True, name="fake"):
        self.result = result
        self.raises = raises
        self.configured = configured
        self.name = name
        self.calls = 0
        self.last_route = None
    def is_configured(self):
        return self.configured
    async def fetch(self, url, route):
        self.calls += 1
        self.last_route = route
        if self.raises:
            raise self.raises
        return self.result


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------

def test_no_providers_at_all_returns_clean_failure():
    m = NetworkManager()
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert r.blocked is True
    assert r.tier_used == "none"
    assert "no providers" in r.block_reason


def test_tier1_only_when_playwright_healthy():
    scraper = FakeScraper(_good_html())
    provider = FakeProvider(result=_good_result())
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert r.tier_used == "tier_1"
    assert provider.calls == 0


def test_tier2_used_when_tier1_blocked():
    scraper = FakeScraper("Please enable JavaScript to continue")
    provider = FakeProvider(result=_good_result(provider="scraperapi"))
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert r.tier_used == "tier_2"
    assert provider.calls == 1


def test_tier3_used_when_tier1_and_2_fail():
    scraper = FakeScraper("Please enable JavaScript to continue")
    provider = FakeProvider(configured=False)   # ScraperAPI not configured
    sa = FakeProvider(result=_good_result(provider="scrapingant"),
                      name="scrapingant")
    m = NetworkManager(scraper, provider, scrapingant=sa)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert r.tier_used == "tier_3"
    assert sa.calls == 1


# ---------------------------------------------------------------------------
# tier_policy
# ---------------------------------------------------------------------------

def test_local_only_policy_skips_paid_tiers():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(result=_good_result())
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch(
        "https://x",
        route=NetworkRoute(tier_policy="local_only"),
    ))
    assert provider.calls == 0
    assert r.blocked is True


def test_no_local_policy_skips_tier1():
    scraper = FakeScraper(_good_html())   # would be fine, but skipped
    provider = FakeProvider(result=_good_result(provider="scraperapi"))
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch(
        "https://x",
        route=NetworkRoute(tier_policy="no_local"),
    ))
    assert scraper.calls == 0
    assert r.provider == "scraperapi"


def test_max_tier_restricts_chain():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(result=_good_result())
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch(
        "https://x",
        route=NetworkRoute(max_tier="tier_1"),
    ))
    assert provider.calls == 0
    assert r.blocked is True


def test_allowed_tiers_overrides_policy():
    scraper = FakeScraper(_good_html())
    provider = FakeProvider(result=_good_result(provider="scraperapi"))
    m = NetworkManager(scraper, provider)
    import asyncio
    # Explicitly allowed only tier 2 — Playwright is skipped
    r = asyncio.run(m.fetch(
        "https://x",
        route=NetworkRoute(allowed_tiers=["tier_2"]),
    ))
    assert scraper.calls == 0
    assert r.provider == "scraperapi"


# ---------------------------------------------------------------------------
# Per-request reset (no persistent circuit-breaker state)
# ---------------------------------------------------------------------------

def test_each_fetch_starts_at_tier1_even_after_prior_failure():
    # First call: Tier 1 fails, Tier 2 succeeds
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(result=_good_result())
    m = NetworkManager(scraper, provider)
    import asyncio
    r1 = asyncio.run(m.fetch("https://x"))
    assert r1.tier_used == "tier_2"

    # Second call: Tier 1 is healthy now — it must be tried again
    scraper.html = _good_html()
    r2 = asyncio.run(m.fetch("https://x"))
    assert r2.tier_used == "tier_1"
    assert r2.provider == "playwright"


# ---------------------------------------------------------------------------
# Attempt trail
# ---------------------------------------------------------------------------

def test_attempt_trail_records_every_provider():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(raises=RuntimeError("down"))
    sa = FakeProvider(result=_good_result(provider="scrapingant"),
                      name="scrapingant")
    m = NetworkManager(scraper, provider, scrapingant=sa)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert len(r.tier_attempts) == 3
    providers = [a["provider"] for a in r.tier_attempts]
    assert providers == ["playwright", "scraperapi", "scrapingant"]
    assert r.tier_attempts[0]["ok"] is False
    assert r.tier_attempts[1]["ok"] is False
    assert r.tier_attempts[2]["ok"] is True


def test_attempt_trail_records_block_reason():
    scraper = FakeScraper("Please enable JavaScript to continue")
    provider = FakeProvider(result=_good_result())
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    pw_attempt = next(a for a in r.tier_attempts if a["provider"] == "playwright")
    assert pw_attempt["ok"] is False
    assert pw_attempt["blocked"] is True
    assert "enable javascript" in pw_attempt["block_reason"].lower()


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

def test_provider_missing_key_is_skipped():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(configured=False)
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert provider.calls == 0
    assert r.blocked is True


def test_provider_not_configured_chain_is_only_tier1():
    scraper = FakeScraper(_good_html())
    provider = FakeProvider(configured=False)
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert r.tier_used == "tier_1"
    assert provider.calls == 0


# ---------------------------------------------------------------------------
# Backward-compat metadata keys
# ---------------------------------------------------------------------------

def test_fallback_from_metadata_present():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(result=_good_result(provider="scraperapi"))
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert r.metadata.get("fallback_from") == "playwright"


def test_fallback_skipped_when_provider_unconfigured():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(configured=False)
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert "fallback_skipped" in r.metadata


def test_fallback_failed_when_provider_also_blocked():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(result=FetchResult(
        url="https://x", html="Please enable JavaScript",
        status_code=200, provider="scraperapi",
        transport="provider_api", blocked=True, block_reason="blocked",
    ))
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert r.metadata.get("fallback_failed") == "provider also blocked"


def test_fallback_error_when_provider_raises():
    scraper = FakeScraper("Please enable JavaScript")
    provider = FakeProvider(raises=RuntimeError("scraperapi down"))
    m = NetworkManager(scraper, provider)
    import asyncio
    r = asyncio.run(m.fetch("https://x"))
    assert "fallback_error" in r.metadata


# ---------------------------------------------------------------------------
# Bot-shell → premium routing
# ---------------------------------------------------------------------------

def test_bot_shell_triggers_premium_on_next_tier():
    shell = (
        "<html><head><title></title></head><body>"
        + "<script>x</script>" * 200
        + "</body></html>"
    )
    scraper = FakeScraper(shell)
    provider = FakeProvider(result=_good_result())
    m = NetworkManager(scraper, provider)
    import asyncio
    asyncio.run(m.fetch("https://x", query_terms=["panadol"]))
    assert provider.last_route.options.get("premium") is True


def test_small_page_does_not_trigger_premium():
    scraper = FakeScraper("Please enable JavaScript to continue")
    provider = FakeProvider(result=_good_result())
    m = NetworkManager(scraper, provider)
    import asyncio
    asyncio.run(m.fetch("https://x"))
    assert provider.last_route.options.get("premium") is False


# ---------------------------------------------------------------------------
# Cookie injection
# ---------------------------------------------------------------------------

def test_session_cookies_passed_to_provider():
    scraper = FakeScraper("Please enable JavaScript")
    captured = {}
    class CaptureProvider:
        name = "capture"
        def is_configured(self): return True
        async def fetch(self, url, route):
            captured["cookies"] = dict(route.session_cookies)
            return _good_result()
    m = NetworkManager(scraper, CaptureProvider())
    import asyncio
    asyncio.run(m.fetch(
        "https://x",
        route=NetworkRoute(session_cookies={"li_at": "xyz"}),
    ))
    assert captured["cookies"] == {"li_at": "xyz"}


# ---------------------------------------------------------------------------
# Jitter
# ---------------------------------------------------------------------------

def test_jitter_called_once_per_fetch():
    calls: list[str] = []
    class FakeJitter:
        async def sleep_for(self, domain):
            calls.append(domain)
            return 0.01
    scraper = FakeScraper(_good_html())
    m = NetworkManager(scraper, jitter=FakeJitter())
    import asyncio
    asyncio.run(m.fetch("https://example.com/p"))
    assert calls == ["example.com"]


def test_jitter_skipped_when_route_disables():
    calls: list[str] = []
    class FakeJitter:
        async def sleep_for(self, domain):
            calls.append(domain)
            return 0.01
    scraper = FakeScraper(_good_html())
    m = NetworkManager(scraper, jitter=FakeJitter())
    import asyncio
    asyncio.run(m.fetch(
        "https://example.com/p",
        route=NetworkRoute(jitter=False),
    ))
    assert calls == []


# ---------------------------------------------------------------------------
# _PlaywrightAdapter
# ---------------------------------------------------------------------------

def test_playwright_adapter_marks_small_page_blocked():
    adapter = _PlaywrightAdapter(FakeScraper("hi"))
    import asyncio
    r = asyncio.run(adapter.fetch("https://x", NetworkRoute()))
    assert r.blocked is True
    assert r.provider == "playwright"


def test_playwright_adapter_passes_good_page():
    adapter = _PlaywrightAdapter(FakeScraper(_good_html()))
    import asyncio
    r = asyncio.run(adapter.fetch("https://x", NetworkRoute()))
    assert r.blocked is False
    assert r.provider == "playwright"


def test_playwright_adapter_handles_raises():
    class Boom:
        async def fetch_html(self, url, timeout=None):
            raise RuntimeError("dns fail")
    adapter = _PlaywrightAdapter(Boom())
    import asyncio
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.fetch("https://x", NetworkRoute()))


# ---------------------------------------------------------------------------
# _provider_configured helper
# ---------------------------------------------------------------------------

def test_provider_configured_true_when_no_method():
    class Bare:
        async def fetch(self, url, route): ...
    assert _provider_configured(Bare()) is True


def test_provider_configured_reads_method():
    class WithFlag:
        def __init__(self, v): self.v = v
        def is_configured(self): return self.v
    assert _provider_configured(WithFlag(True)) is True
    assert _provider_configured(WithFlag(False)) is False


def test_provider_configured_handles_exception():
    class Broken:
        def is_configured(self): raise RuntimeError("boom")
    assert _provider_configured(Broken()) is False


# ---------------------------------------------------------------------------
# Tier enum sanity
# ---------------------------------------------------------------------------

def test_tier_values_present():
    assert Tier.TIER_1.value == "tier_1"
    assert Tier.TIER_2.value == "tier_2"
    assert Tier.TIER_3.value == "tier_3"
    assert Tier.NONE.value == "none"