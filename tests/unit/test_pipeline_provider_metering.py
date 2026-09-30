"""
Unit tests for per-run provider-credit metering (spec §41.4).

Covers:
    - NetworkManager initializes the credit meter at zero
    - reset_usage() zeroes the meter
    - _record_provider_usage accumulates credits by provider
    - Zero-credit results (Playwright) don't add to the total
    - The `_PlaywrightAdapter` reports zero credits
    - PipelineRunner writes a PROVIDER_CREDIT event with the
      per-provider breakdown in metadata
    - No PROVIDER_CREDIT event when no paid provider was used
    - The network manager's meter is reset at the start of every run
    - Managers without the meter attribute don't break the pipeline
"""
import asyncio

import pytest

from src.core.task_spec import FieldSpec, Target, TaskSpec
from src.network.manager import NetworkManager, _PlaywrightAdapter
from src.network.types import FetchResult, NetworkRoute
from src.pipeline_runner import PipelineRunner
from src.usage.store import InMemoryUsageStore
from src.usage.types import ResourceType


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# NetworkManager meter: init / reset / accumulate
# ---------------------------------------------------------------------------

def test_manager_starts_with_zero_meter():
    m = NetworkManager()
    assert m.total_provider_credits == 0
    assert m.provider_credits_by_provider == {}
    assert m.provider_call_count == 0


def test_manager_reset_usage_zeroes_counters():
    m = NetworkManager()
    m.total_provider_credits = 42
    m.provider_credits_by_provider = {"scraperapi": 42}
    m.provider_call_count = 9

    m.reset_usage()
    assert m.total_provider_credits == 0
    assert m.provider_credits_by_provider == {}
    assert m.provider_call_count == 0


def test_record_provider_usage_accumulates_credits():
    m = NetworkManager()

    r1 = FetchResult(
        url="https://x", html="<html>" + "y" * 1000,
        status_code=200, provider="scraperapi", credits_used=5,
    )
    r2 = FetchResult(
        url="https://y", html="<html>" + "y" * 1000,
        status_code=200, provider="scraperapi", credits_used=25,
    )
    m._record_provider_usage(r1)
    m._record_provider_usage(r2)

    assert m.total_provider_credits == 30
    assert m.provider_credits_by_provider == {"scraperapi": 30}
    assert m.provider_call_count == 2


def test_record_provider_usage_per_provider_breakdown():
    m = NetworkManager()

    m._record_provider_usage(FetchResult(
        url="x", html="", status_code=200,
        provider="scraperapi", credits_used=5,
    ))
    m._record_provider_usage(FetchResult(
        url="y", html="", status_code=200,
        provider="scrapingant", credits_used=1,
    ))
    m._record_provider_usage(FetchResult(
        url="z", html="", status_code=200,
        provider="scraperapi", credits_used=25,
    ))

    assert m.total_provider_credits == 31
    assert m.provider_credits_by_provider == {
        "scraperapi": 30,
        "scrapingant": 1,
    }
    assert m.provider_call_count == 3


def test_record_provider_usage_zero_credits_still_bumps_call_count():
    """Playwright/direct fetches report zero credits — count the call, add nothing."""
    m = NetworkManager()
    m._record_provider_usage(FetchResult(
        url="x", html="<html>" + "y" * 1000, status_code=200,
        provider="playwright", credits_used=0,
    ))
    assert m.total_provider_credits == 0
    assert m.provider_credits_by_provider == {}
    assert m.provider_call_count == 1


def test_playwright_adapter_reports_zero_credits():
    """The adapter wraps the local browser and never bills a provider."""
    class _FakeScraper:
        async def fetch_html(self, url, timeout=None):
            return "<html><body>" + ("x" * 1000) + "</body></html>"

    adapter = _PlaywrightAdapter(_FakeScraper())
    result = _run(adapter.fetch("https://x", NetworkRoute()))
    assert result.provider == "playwright"
    assert result.credits_used == 0


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class _FakeScraper:
    async def fetch_html(self, url, timeout=None):
        return "<html><body>" + ("x" * 1000) + "</body></html>"

    async def fetch_screenshot(self, url, timeout=None):
        return b""


class _FakeExtractor:
    def extract_list(self, html, instruction):
        return [{"title": "A"}]

    def extract_from_image(self, img, instr):
        return []


class _FakeNetworkManager:
    """Reports configurable credits per run."""
    def __init__(self, credits: int = 0, breakdown: dict | None = None):
        self.total_provider_credits = credits
        self.provider_credits_by_provider = breakdown or {}
        self.provider_call_count = 1 if credits else 0
        self.reset_calls = 0

    def reset_usage(self):
        self.reset_calls += 1
        self.total_provider_credits = 0
        self.provider_credits_by_provider = {}
        self.provider_call_count = 0

    async def fetch(self, url, **kwargs):
        # The pipeline never calls .fetch() directly in these tests —
        # _scrape_source does, and the fake scraper's fetch_html is what
        # gets used. This method exists only for API compatibility.
        return FetchResult(url=url, html="", status_code=200)


def _task(task_id="t-provider"):
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.task_id = task_id
    return spec


def test_pipeline_writes_provider_credit_event():
    """
    A network manager reporting credits must produce a PROVIDER_CREDIT
    event with the per-provider breakdown in metadata.
    """
    store = InMemoryUsageStore()
    nm = _FakeNetworkManager(
        credits=30, breakdown={"scraperapi": 30},
    )

    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-provider-1",
        usage_store=store,
        network_manager=nm,
    )
    # Manually re-inject the credits because reset_usage() zeroes them.
    # This mimics a real run where the manager accumulates during fetch.
    async def _run_and_inject():
        # Run in the background so the reset happens, then set the meter
        # to what a real fetch would have accumulated before _record_run_usage
        # reads it.
        # Simpler: patch _record_run_usage to read directly.
        pass

    # Instead of the above, just check the accumulator is forwarded by
    # directly setting it after reset but before reading.
    # The pipeline's reset_usage runs first thing in run(); we can't inject
    # mid-run from outside, so we test the wiring with a NetworkManager
    # that ignores reset.
    class _StickyManager(_FakeNetworkManager):
        def reset_usage(self):
            # Do NOT zero the meter — tests that matter are the ones
            # checking the value flows through.
            pass

    sticky = _StickyManager(credits=30, breakdown={"scraperapi": 30})
    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-provider-1b",
        usage_store=store,
        network_manager=sticky,
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-provider-1b"))
    by_type = {e.resource_type: e for e in events}
    assert ResourceType.PROVIDER_CREDIT in by_type

    pc = by_type[ResourceType.PROVIDER_CREDIT]
    assert pc.quantity == 30.0
    assert pc.metadata["breakdown"] == {"scraperapi": 30}
    assert pc.metadata["provider_calls"] == 1
    assert pc.unit == "credit"


def test_pipeline_skips_provider_event_when_zero():
    """A run that used no paid providers writes no PROVIDER_CREDIT event."""
    store = InMemoryUsageStore()
    nm = _FakeNetworkManager(credits=0, breakdown={})

    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-no-provider",
        usage_store=store,
        network_manager=nm,
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-no-provider"))
    types = {e.resource_type for e in events}
    assert ResourceType.PROVIDER_CREDIT not in types
    # PAGE event still written
    assert ResourceType.PAGE in types


def test_pipeline_calls_network_manager_reset_at_run_start():
    """The manager's reset_usage() must run before the first fetch."""
    store = InMemoryUsageStore()
    nm = _FakeNetworkManager(credits=0)

    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-nm-reset",
        usage_store=store,
        network_manager=nm,
    )
    _run(runner.run(_task()))
    assert nm.reset_calls == 1


def test_pipeline_tolerates_network_manager_without_meter():
    """Legacy managers without the credit attributes must still work."""
    class _BareManager:
        async def fetch(self, url, **kwargs):
            # Real HTML so _scrape_source reaches the extractor. An empty
            # body would abort the run before the pipeline metering step.
            return FetchResult(
                url=url,
                html="<html><body>" + ("x" * 1000) + "</body></html>",
                status_code=200,
            )

    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-bare-nm",
        usage_store=store,
        network_manager=_BareManager(),
    )
    result = _run(runner.run(_task()))
    assert len(result.records) == 1
    events = _run(store.events_for_job("job-bare-nm"))
    types = {e.resource_type for e in events}
    assert ResourceType.PROVIDER_CREDIT not in types


def test_pipeline_writes_all_four_resource_types_when_applicable():
    """Integration check: all four meters fire together."""
    class _StickyManager(_FakeNetworkManager):
        def reset_usage(self):
            pass

    class _ExtractorWithTokens(_FakeExtractor):
        def __init__(self):
            self.total_input_tokens = 500
            self.total_output_tokens = 120
            self.call_count = 1
        def reset_usage(self):
            self.total_input_tokens = 500
            self.total_output_tokens = 120
            self.call_count = 1

    class _ScraperWithBrowser(_FakeScraper):
        def __init__(self):
            self.total_browser_seconds = 3.0
            self.browser_call_count = 1
        def reset_usage(self):
            self.total_browser_seconds = 3.0
            self.browser_call_count = 1

    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _ScraperWithBrowser(), _ExtractorWithTokens(),
        client_id="acme", job_id="job-all-four",
        usage_store=store,
        network_manager=_StickyManager(
            credits=5, breakdown={"scraperapi": 5},
        ),
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-all-four"))
    types = {e.resource_type for e in events}
    assert types == {
        ResourceType.PAGE,
        ResourceType.TOKEN,
        ResourceType.BROWSER_SECOND,
        ResourceType.PROVIDER_CREDIT,
    }