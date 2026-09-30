"""
Unit tests for per-run browser-second metering (spec §41.4).

Covers:
    - ScraperEngine initializes the meter at zero
    - reset_usage() zeros the meter
    - _metered decorator accumulates elapsed time on success
    - _metered decorator accumulates time even when the wrapped fn raises
    - PipelineRunner writes a BROWSER_SECOND event alongside PAGE and TOKEN
    - No BROWSER_SECOND event when the scraper reports zero seconds
    - The scraper's meter is reset at the start of each pipeline run
    - Scrapers without the meter attribute don't break the pipeline
"""
import asyncio

import pytest

from src.core.task_spec import FieldSpec, Target, TaskSpec
from src.extractor.schema_extractor import DataExtractor
from src.pipeline_runner import PipelineRunner
from src.scraper.engine import ScraperEngine, _metered
from src.usage.store import InMemoryUsageStore
from src.usage.types import ResourceType


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# ScraperEngine meter: init, reset, decorator
# ---------------------------------------------------------------------------

def test_scraper_engine_starts_with_zero_meter():
    engine = ScraperEngine()
    assert engine.total_browser_seconds == 0.0
    assert engine.browser_call_count == 0


def test_scraper_reset_usage_zeroes_counters():
    engine = ScraperEngine()
    engine.total_browser_seconds = 12.5
    engine.browser_call_count = 7

    engine.reset_usage()
    assert engine.total_browser_seconds == 0.0
    assert engine.browser_call_count == 0


def test_metered_decorator_records_time_on_success():
    """
    Apply the decorator to a lightweight async function and confirm it
    accumulates. Does not launch a real browser.
    """
    class _Dummy:
        total_browser_seconds = 0.0
        browser_call_count = 0

        @_metered
        async def work(self, sleep_for):
            await asyncio.sleep(sleep_for)
            return "done"

    d = _Dummy()
    result = _run(d.work(0.05))

    assert result == "done"
    assert d.browser_call_count == 1
    # Invariant: time was recorded. Exact value depends on scheduler.
    assert d.total_browser_seconds > 0


def test_metered_decorator_records_time_on_exception():
    """Timing is recorded even when the wrapped method raises."""
    class _Dummy:
        total_browser_seconds = 0.0
        browser_call_count = 0

        @_metered
        async def work(self):
            await asyncio.sleep(0.03)
            raise RuntimeError("boom")

    d = _Dummy()
    with pytest.raises(RuntimeError, match="boom"):
        _run(d.work())

    assert d.browser_call_count == 1
    # Time must be recorded even when the wrapped method raises.
    assert d.total_browser_seconds > 0


def test_metered_decorator_accumulates_across_calls():
    class _Dummy:
        total_browser_seconds = 0.0
        browser_call_count = 0

        @_metered
        async def work(self):
            await asyncio.sleep(0.02)

    d = _Dummy()
    for _ in range(3):
        _run(d.work())

    assert d.browser_call_count == 3
    # Time is accumulated across every call; don't assert an exact floor
    # because scheduler jitter can shorten individual sleeps.
    assert d.total_browser_seconds > 0


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class _FakeScraperWithMeter:
    """
    Scraper stand-in that reports a configurable number of browser seconds
    per fetch. Simulates the real engine's meter without launching Chrome.
    """
    def __init__(self, seconds_per_fetch: float = 3.0):
        self.seconds_per_fetch = seconds_per_fetch
        self.total_browser_seconds = 0.0
        self.browser_call_count = 0
        self.reset_calls = 0

    def reset_usage(self):
        self.reset_calls += 1
        self.total_browser_seconds = 0.0
        self.browser_call_count = 0

    async def fetch_html(self, url, timeout=None):
        self.total_browser_seconds += self.seconds_per_fetch
        self.browser_call_count += 1
        return "<html><body>" + ("x" * 1000) + "</body></html>"

    async def fetch_screenshot(self, url, timeout=None):
        return b""


class _FakeExtractorWithTokens:
    """
    Extractor stand-in that reports a configurable number of tokens per
    `extract_list()` call. The real extractor increments its counters
    inside extract_list — this fake does the same, so it works correctly
    with the pipeline's per-run `reset_usage()` call.
    """
    def __init__(self, in_t: int = 100, out_t: int = 30):
        self._per_call_in = in_t
        self._per_call_out = out_t
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.reset_calls = 0

    def reset_usage(self):
        self.reset_calls += 1
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def extract_list(self, html, instruction):
        # Simulate one LLM call: increment the running counters.
        self.total_input_tokens += self._per_call_in
        self.total_output_tokens += self._per_call_out
        self.call_count += 1
        return [{"title": "A"}]

    def extract_from_image(self, img, instr):
        return []


def _task(task_id="t-browser"):
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.task_id = task_id
    return spec


def test_pipeline_writes_browser_second_event():
    store = InMemoryUsageStore()
    scraper = _FakeScraperWithMeter(seconds_per_fetch=4.5)
    extractor = _FakeExtractorWithTokens()

    runner = PipelineRunner(
        scraper, extractor,
        client_id="acme",
        job_id="job-browser-1",
        usage_store=store,
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-browser-1"))
    by_type = {e.resource_type: e for e in events}

    assert ResourceType.BROWSER_SECOND in by_type
    bs = by_type[ResourceType.BROWSER_SECOND]
    assert bs.quantity == 4.5
    assert bs.metadata["browser_calls"] == 1
    assert bs.unit == "second"


def test_pipeline_writes_all_three_resource_types():
    store = InMemoryUsageStore()
    scraper = _FakeScraperWithMeter(seconds_per_fetch=2.0)
    extractor = _FakeExtractorWithTokens(in_t=500, out_t=120)

    runner = PipelineRunner(
        scraper, extractor,
        client_id="acme", job_id="job-all",
        usage_store=store,
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-all"))
    types = {e.resource_type for e in events}
    assert types == {
        ResourceType.PAGE,
        ResourceType.TOKEN,
        ResourceType.BROWSER_SECOND,
    }


def test_pipeline_skips_browser_event_when_zero():
    """A fake scraper reporting zero seconds writes no BROWSER_SECOND event."""
    store = InMemoryUsageStore()
    scraper = _FakeScraperWithMeter(seconds_per_fetch=0.0)
    extractor = _FakeExtractorWithTokens()

    runner = PipelineRunner(
        scraper, extractor,
        client_id="acme", job_id="job-zero-browser",
        usage_store=store,
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-zero-browser"))
    types = {e.resource_type for e in events}
    assert ResourceType.BROWSER_SECOND not in types
    # But PAGE and TOKEN still made it
    assert ResourceType.PAGE in types
    assert ResourceType.TOKEN in types


def test_pipeline_calls_scraper_reset_usage_at_run_start():
    """The scraper's reset_usage() must run before any fetch."""
    store = InMemoryUsageStore()
    scraper = _FakeScraperWithMeter(seconds_per_fetch=1.0)
    extractor = _FakeExtractorWithTokens()

    runner = PipelineRunner(
        scraper, extractor,
        client_id="acme", job_id="job-reset",
        usage_store=store,
    )
    # Pre-seed the meter to prove it gets zeroed before the run.
    scraper.total_browser_seconds = 99.0
    scraper.browser_call_count = 9

    _run(runner.run(_task()))

    # One reset at run start, one fetch inside the run.
    assert scraper.reset_calls == 1
    # The event reflects the run, not the pre-seed value.
    events = _run(store.events_for_job("job-reset"))
    bs = next(
        (e for e in events if e.resource_type == ResourceType.BROWSER_SECOND),
        None,
    )
    assert bs is not None
    assert bs.quantity == 1.0


def test_pipeline_tolerates_scraper_without_meter():
    """A legacy scraper without `total_browser_seconds` must still work."""
    class _BareScraper:
        async def fetch_html(self, url, timeout=None):
            return "<html><body>" + ("x" * 1000) + "</body></html>"

        async def fetch_screenshot(self, url, timeout=None):
            return b""

    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _BareScraper(), _FakeExtractorWithTokens(),
        client_id="acme", job_id="job-bare-scraper",
        usage_store=store,
    )
    result = _run(runner.run(_task()))

    assert len(result.records) == 1
    events = _run(store.events_for_job("job-bare-scraper"))
    types = {e.resource_type for e in events}
    assert ResourceType.BROWSER_SECOND not in types
    assert ResourceType.PAGE in types