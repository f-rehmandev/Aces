"""
Unit tests for the entitlement gate (§47B) and budget circuit breaker
(§38) inside PipelineRunner.

Covers:
    - Backward compatibility: no engine/tracker → no-op
    - Entitlement gate: allowed, denied, warning-propagation, fail-open
      on engine errors, correct page count in the request
    - Budget gate: soft warning once per run, hard trip stops cleanly,
      partial data preserved, per-fetch consumption
    - Combined: entitlement runs first, budget enforces later
"""
import asyncio

import pytest

from src.core.task_spec import FieldSpec, Quality, Target, TaskSpec
from src.jobs.budget import Budget, BudgetTracker
from src.pipeline_runner import PipelineRunner
from src.usage.entitlements import EntitlementDecision, OveragePolicy


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeScraper:
    def __init__(self, html="<html>SENTINEL</html>"):
        self.html = html
        self.calls = 0

    async def fetch_html(self, url, timeout=None):
        self.calls += 1
        return self.html

    async def fetch_screenshot(self, url, timeout=None):
        return b"png"


class FakeExtractor:
    def __init__(self, items=None):
        self.items = items if items is not None else [{"title": "A", "price": "$1"}]

    def extract_list(self, html, instruction):
        return list(self.items)

    def extract_from_image(self, img, instr):
        return []


class FakeEntitlementEngine:
    def __init__(
        self, allowed=True, warning="", reason="within limit",
    ):
        self.allowed = allowed
        self.warning = warning
        self.reason = reason
        self.calls: list[tuple] = []

    async def check(self, client_id, resource, requested=0.0, since=None):
        self.calls.append((client_id, resource, requested))
        return EntitlementDecision(
            allowed=self.allowed,
            client_id=client_id,
            resource=resource,
            requested=requested,
            reason=self.reason,
            warning=self.warning,
            policy=OveragePolicy.BLOCK.value,
        )


class ExplodingEntitlementEngine:
    async def check(self, *args, **kwargs):
        raise RuntimeError("entitlement store down")


def _task(urls=None, task_id="t-fixed"):
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=urls or ["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
        quality=Quality(min_records=1),
    )
    spec.task_id = task_id
    return spec


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# Backward compatibility: both gates optional
# ===========================================================================

def test_no_entitlement_engine_is_noop():
    runner = PipelineRunner(FakeScraper(), FakeExtractor())
    result = _run(runner.run(_task()))
    assert result.records
    assert not any("entitlement" in w.lower() for w in result.warnings)


def test_no_budget_tracker_is_noop():
    runner = PipelineRunner(FakeScraper(), FakeExtractor())
    result = _run(runner.run(_task()))
    assert result.records
    assert not any("budget" in w.lower() for w in result.warnings)


# ===========================================================================
# Entitlement gate
# ===========================================================================

def test_entitlement_allowed_runs_normally():
    engine = FakeEntitlementEngine(allowed=True)
    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        entitlement_engine=engine,
    )
    result = _run(runner.run(_task()))
    assert result.records
    assert len(engine.calls) == 1


def test_entitlement_asked_for_correct_client_and_resource():
    engine = FakeEntitlementEngine(allowed=True)
    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="tenant-x",
        entitlement_engine=engine,
    )
    _run(runner.run(_task()))
    client_id, resource, requested = engine.calls[0]
    assert client_id == "tenant-x"
    assert resource == "page"
    assert requested == 1.0


def test_entitlement_request_uses_url_count():
    """Requested pages must equal the number of start URLs."""
    engine = FakeEntitlementEngine(allowed=True)
    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        entitlement_engine=engine,
    )
    _run(runner.run(_task(urls=[
        "https://93.184.216.34/a",
        "https://93.184.216.35/b",
        "https://93.184.216.36/c",
    ])))
    _, _, requested = engine.calls[0]
    assert requested == 3.0


def test_entitlement_denied_returns_empty_result():
    engine = FakeEntitlementEngine(
        allowed=False, reason="over limit: 500 consumed, 1 requested",
    )
    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        entitlement_engine=engine,
    )
    result = _run(runner.run(_task()))
    assert result.records == []
    assert any(
        "entitlement denied" in w.lower() for w in result.warnings
    )
    assert any("over limit" in w for w in result.warnings)


def test_entitlement_denied_does_not_fetch():
    """The scraper must not be called when the gate refuses."""
    scraper = FakeScraper()
    engine = FakeEntitlementEngine(allowed=False, reason="over limit")
    runner = PipelineRunner(
        scraper, FakeExtractor(),
        client_id="acme",
        entitlement_engine=engine,
    )
    _run(runner.run(_task()))
    assert scraper.calls == 0


def test_entitlement_engine_exception_fails_open():
    """A broken engine warns but does not block — availability > quota."""
    scraper = FakeScraper()
    runner = PipelineRunner(
        scraper, FakeExtractor(),
        client_id="acme",
        entitlement_engine=ExplodingEntitlementEngine(),
    )
    result = _run(runner.run(_task()))
    assert result.records
    assert any(
        "entitlement check failed" in w.lower() for w in result.warnings
    )


def test_entitlement_warning_is_propagated():
    engine = FakeEntitlementEngine(
        allowed=True, warning="overage policy: WARN",
    )
    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        entitlement_engine=engine,
    )
    result = _run(runner.run(_task()))
    assert any(
        "entitlement warning" in w.lower() and "WARN" in w
        for w in result.warnings
    )


# ===========================================================================
# Budget circuit breaker
# ===========================================================================

def test_budget_under_limit_produces_no_warnings():
    tracker = BudgetTracker(Budget(max_pages=100))
    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        budget_tracker=tracker,
    )
    result = _run(runner.run(_task(urls=[
        "https://93.184.216.34/a",
        "https://93.184.216.35/b",
    ])))
    assert not any("budget" in w.lower() for w in result.warnings)


def test_budget_hard_trip_stops_cleanly():
    """max_pages=2 with 3 URLs → exactly 2 fetches, then stop."""
    scraper = FakeScraper()
    tracker = BudgetTracker(Budget(max_pages=2))
    runner = PipelineRunner(
        scraper, FakeExtractor(),
        client_id="acme",
        budget_tracker=tracker,
    )
    result = _run(runner.run(_task(urls=[
        "https://93.184.216.34/a",
        "https://93.184.216.35/b",
        "https://93.184.216.36/c",
    ])))
    # 3rd URL never fetched; circuit breaker stopped before it
    assert scraper.calls == 2
    assert any("circuit breaker" in w.lower() for w in result.warnings)


def test_budget_hard_trip_preserves_partial_data():
    scraper = FakeScraper()
    tracker = BudgetTracker(Budget(max_pages=1))
    runner = PipelineRunner(
        scraper, FakeExtractor(),
        client_id="acme",
        budget_tracker=tracker,
    )
    result = _run(runner.run(_task(urls=[
        "https://93.184.216.34/a",
        "https://93.184.216.35/b",
    ])))
    assert scraper.calls == 1
    # The one page that did fetch produced a record
    assert len(result.records) >= 1


def test_budget_consumes_one_page_per_attempt():
    """Even a failed fetch consumes budget — bandwidth is still spent."""
    class FailScraper:
        def __init__(self):
            self.calls = 0
        async def fetch_html(self, url, timeout=None):
            self.calls += 1
            raise RuntimeError("network error")
        async def fetch_screenshot(self, url, timeout=None):
            return b""

    tracker = BudgetTracker(Budget(max_pages=10))
    runner = PipelineRunner(
        FailScraper(), FakeExtractor(),
        client_id="acme",
        budget_tracker=tracker,
    )
    _run(runner.run(_task(urls=[
        "https://93.184.216.34/a",
        "https://93.184.216.35/b",
    ])))
    assert tracker.budget.pages_used == 2


def test_budget_soft_warning_emitted_at_most_once_per_run():
    """
    max_pages=10, soft fraction 0.5 → soft threshold at 5 pages.
    Run with 8 URLs so the limit is crossed and then re-checked
    several times. The warning must be emitted exactly once.
    """
    urls = [f"https://93.184.216.{34 + i}/p" for i in range(8)]
    tracker = BudgetTracker(Budget(max_pages=10, soft_limit_fraction=0.5))
    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        budget_tracker=tracker,
    )
    result = _run(runner.run(_task(urls=urls)))
    soft = [w for w in result.warnings if "soft limit" in w.lower()]
    assert len(soft) == 1, f"expected exactly one soft warning, got {soft}"


# ===========================================================================
# Combined behavior
# ===========================================================================

def test_entitlement_then_budget_ordering():
    """Entitlement checks first; budget enforces per-fetch after."""
    engine = FakeEntitlementEngine(allowed=True)
    tracker = BudgetTracker(Budget(max_pages=1))
    scraper = FakeScraper()
    runner = PipelineRunner(
        scraper, FakeExtractor(),
        client_id="acme",
        entitlement_engine=engine,
        budget_tracker=tracker,
    )
    result = _run(runner.run(_task(urls=[
        "https://93.184.216.34/a",
        "https://93.184.216.35/b",
    ])))
    # Entitlement consulted once, before fetch
    assert len(engine.calls) == 1
    # Budget then allowed exactly one fetch
    assert scraper.calls == 1
    assert any("circuit breaker" in w.lower() for w in result.warnings)


def test_entitlement_denied_skips_budget_entirely():
    """If entitlement refuses, budget is never consulted."""
    engine = FakeEntitlementEngine(allowed=False, reason="over limit")
    tracker = BudgetTracker(Budget(max_pages=10))
    scraper = FakeScraper()
    runner = PipelineRunner(
        scraper, FakeExtractor(),
        client_id="acme",
        entitlement_engine=engine,
        budget_tracker=tracker,
    )
    _run(runner.run(_task(urls=["https://93.184.216.34/a"])))
    assert scraper.calls == 0
    assert tracker.budget.pages_used == 0


def test_both_gates_absent_is_fully_transparent():
    """The old constructor signature still works and behaves identically."""
    scraper = FakeScraper()
    runner = PipelineRunner(scraper, FakeExtractor(), client_id="acme")
    result = _run(runner.run(_task()))
    assert scraper.calls == 1
    assert not any(
        "entitlement" in w.lower() or "budget" in w.lower()
        for w in result.warnings
    )