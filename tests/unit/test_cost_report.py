"""Unit tests for the cost-report layer (spec §41.4)."""
import asyncio

import pytest

from src.ops.cost_report import CostRates, CostReport, CostReporter
from src.usage.store import InMemoryUsageStore
from src.usage.types import ResourceType, UsageEvent


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_with(events: list[UsageEvent]) -> InMemoryUsageStore:
    store = InMemoryUsageStore()
    _run(store.record_batch(events))
    return store


# ---------------------------------------------------------------------------
# Empty / trivial
# ---------------------------------------------------------------------------

def test_empty_job_returns_zero_report():
    store = InMemoryUsageStore()
    report = _run(CostReporter(store).report_for_job("nonexistent"))
    assert report.total_usd == 0.0
    assert report.event_count == 0
    assert report.by_resource == {}
    assert report.cost_per_1k_items is None


def test_empty_job_id_returns_empty_report():
    store = InMemoryUsageStore()
    report = _run(CostReporter(store).report_for_job(""))
    assert report.total_usd == 0.0


def test_page_only_run_has_zero_cost_with_default_rates():
    """Default page rate is 0 (local fetch)."""
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PAGE, quantity=10,
        ),
    ])
    report = _run(CostReporter(store).report_for_job("j-1"))
    assert report.total_usd == 0.0
    assert report.by_resource["page"]["quantity"] == 10.0
    assert report.by_resource["page"]["usd"] == 0.0


def test_page_rate_can_be_set():
    """A deployment paying per page overrides the rate."""
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PAGE, quantity=10,
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(page=0.001))
    report = _run(reporter.report_for_job("j-1"))
    assert abs(report.total_usd - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# Token cost
# ---------------------------------------------------------------------------

def test_token_cost_with_split():
    """Input and output tokens are charged at different rates."""
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.TOKEN, quantity=5000,
            metadata={"input_tokens": 4000, "output_tokens": 1000},
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(
        token_input_per_1k=0.001,    # $0.001 per 1k input
        token_output_per_1k=0.003,   # $0.003 per 1k output
    ))
    report = _run(reporter.report_for_job("j-1"))
    # 4 * 0.001 + 1 * 0.003 = 0.007
    assert abs(report.total_usd - 0.007) < 1e-9


def test_token_cost_without_split_falls_back_to_blended():
    """Old events with no input/output metadata use the blended rate."""
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.TOKEN, quantity=5000,
            # No metadata
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(
        token_blended_per_1k=0.002,
    ))
    report = _run(reporter.report_for_job("j-1"))
    # 5 * 0.002 = 0.010
    assert abs(report.total_usd - 0.010) < 1e-9


def test_token_metadata_with_garbage_falls_back_to_blended():
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.TOKEN, quantity=1000,
            metadata={"input_tokens": "not-a-number",
                      "output_tokens": "also-bad"},
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(
        token_blended_per_1k=0.001,
    ))
    report = _run(reporter.report_for_job("j-1"))
    assert abs(report.total_usd - 0.001) < 1e-9


# ---------------------------------------------------------------------------
# Browser seconds / vision / provider credits
# ---------------------------------------------------------------------------

def test_browser_seconds_with_rate():
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.BROWSER_SECOND, quantity=22.5,
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(browser_second=0.0005))
    report = _run(reporter.report_for_job("j-1"))
    # 22.5 * 0.0005 = 0.01125
    assert abs(report.total_usd - 0.01125) < 1e-9


def test_vision_calls_with_rate():
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.VISION_CALL, quantity=3,
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(vision_call=0.0008))
    report = _run(reporter.report_for_job("j-1"))
    assert abs(report.total_usd - 0.0024) < 1e-9


def test_provider_credits_with_rate():
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PROVIDER_CREDIT, quantity=30,
            metadata={"provider_calls": 3,
                      "breakdown": {"scraperapi": 25, "scrapingant": 5}},
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(provider_credit=0.001))
    report = _run(reporter.report_for_job("j-1"))
    assert abs(report.total_usd - 0.03) < 1e-9
    # Per-provider breakdown
    assert "scraperapi" in report.by_provider
    assert "scrapingant" in report.by_provider
    assert abs(report.by_provider["scraperapi"]["usd"] - 0.025) < 1e-9
    assert abs(report.by_provider["scrapingant"]["usd"] - 0.005) < 1e-9


def test_provider_credits_without_breakdown_attributes_to_provider_field():
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PROVIDER_CREDIT, quantity=10,
            provider="scraperapi",
            # No breakdown in metadata
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(provider_credit=0.001))
    report = _run(reporter.report_for_job("j-1"))
    assert "scraperapi" in report.by_provider


# ---------------------------------------------------------------------------
# Mixed run
# ---------------------------------------------------------------------------

def test_mixed_run_totals_correctly():
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PAGE, quantity=10,
        ),
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.TOKEN, quantity=5000,
            metadata={"input_tokens": 4000, "output_tokens": 1000},
        ),
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.BROWSER_SECOND, quantity=20,
        ),
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PROVIDER_CREDIT, quantity=10,
            metadata={"breakdown": {"scraperapi": 10}},
        ),
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.VISION_CALL, quantity=2,
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(
        token_input_per_1k=0.001,
        token_output_per_1k=0.003,
        browser_second=0.0005,
        provider_credit=0.001,
        vision_call=0.0008,
        page=0.0,
    ))
    report = _run(reporter.report_for_job("j-1"))
    # token: 4*0.001 + 1*0.003 = 0.007
    # browser: 20 * 0.0005      = 0.010
    # provider: 10 * 0.001      = 0.010
    # vision: 2 * 0.0008        = 0.0016
    expected = 0.007 + 0.010 + 0.010 + 0.0016
    assert abs(report.total_usd - expected) < 1e-9
    assert report.event_count == 5


# ---------------------------------------------------------------------------
# cost_per_1k_items
# ---------------------------------------------------------------------------

def test_cost_per_1k_items_computed():
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PROVIDER_CREDIT, quantity=10,
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(provider_credit=0.001))
    report = _run(reporter.report_for_job("j-1", items_listed=500))
    # 0.01 total / 500 items * 1000 = 0.02 per 1k
    assert report.cost_per_1k_items is not None
    assert abs(report.cost_per_1k_items - 0.02) < 1e-9


def test_cost_per_1k_items_zero_when_cost_is_zero():
    """Zero cost is a legitimate answer (all free-tier)."""
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PAGE, quantity=10,
        ),
    ])
    reporter = CostReporter(store)
    report = _run(reporter.report_for_job("j-1", items_listed=100))
    assert report.cost_per_1k_items == 0.0


def test_cost_per_1k_items_none_without_items():
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.TOKEN, quantity=1000,
            metadata={"input_tokens": 1000, "output_tokens": 0},
        ),
    ])
    reporter = CostReporter(store)
    report = _run(reporter.report_for_job("j-1"))
    assert report.cost_per_1k_items is None


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

def test_report_to_dict_shape():
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PROVIDER_CREDIT, quantity=10,
            metadata={"breakdown": {"scraperapi": 10}},
        ),
    ])
    reporter = CostReporter(store, rates=CostRates(provider_credit=0.001))
    report = _run(reporter.report_for_job("j-1", items_listed=100))
    d = report.to_dict()
    assert d["total_usd"] == 0.01
    assert "by_resource" in d
    assert "by_provider" in d
    assert d["event_count"] == 1
    assert d["items_listed"] == 100
    assert d["cost_per_1k_items"] == 0.1


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_reporter_returns_empty_on_store_read_failure():
    class _BrokenStore:
        async def events_for_job(self, job_id):
            raise RuntimeError("db down")
        async def events_for_client(self, client_id, limit=1000):
            raise RuntimeError("db down")
        async def summarize(self, *a, **k):
            raise RuntimeError("db down")

    reporter = CostReporter(_BrokenStore())
    report = _run(reporter.report_for_job("j-1"))
    assert report.total_usd == 0.0
    assert report.event_count == 0


def test_client_report_aggregates_across_jobs():
    store = _store_with([
        UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PAGE, quantity=5,
        ),
        UsageEvent(
            client_id="acme", job_id="j-2",
            resource_type=ResourceType.PAGE, quantity=7,
        ),
    ])
    report = _run(CostReporter(store).report_for_client("acme"))
    assert report.by_resource["page"]["quantity"] == 12.0
    assert report.event_count == 2