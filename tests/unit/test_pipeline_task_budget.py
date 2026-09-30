"""
Unit tests for TaskSpec.budget enforcement inside PipelineRunner (spec §38).

Covers:
    - When no budget_tracker was injected, one is built from task.budget
    - A small task.budget.max_pages trips the circuit breaker mid-run
    - The circuit-breaker warning is surfaced in the result
    - Records collected before the trip are preserved
    - An injected tracker (the test-path pattern) is used unchanged
    - A large task.budget.max_pages does not fire and produces no warning
"""
import asyncio

import pytest

from src.core.task_spec import (
    Budget as TaskBudget,
    FieldSpec,
    Quality,
    Target,
    TaskSpec,
)
from src.pipeline_runner import PipelineRunner


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeScraper:
    """Counts fetches. Returns a benign page for every URL."""

    def __init__(self):
        self.fetches = 0

    async def fetch_html(self, url, timeout=None):
        self.fetches += 1
        return "<html><body>" + ("x" * 1000) + "</body></html>"

    async def fetch_screenshot(self, url, timeout=None):
        return b""


class _FakeExtractor:
    def extract_list(self, html, instruction):
        return [{"title": "A"}]

    def extract_from_image(self, img, instr):
        return []


def _task_with_budget(
    max_pages: int,
    urls: list[str] | None = None,
    task_id: str = "t-budget",
) -> TaskSpec:
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(
            start_urls=urls or ["https://93.184.216.34/a"],
        ),
        fields=[FieldSpec(name="title")],
        quality=Quality(min_records=1),
        budget=TaskBudget(max_pages=max_pages),
    )
    spec.task_id = task_id
    return spec


# ---------------------------------------------------------------------------
# Auto-built tracker from task.budget
# ---------------------------------------------------------------------------

def test_task_budget_is_built_when_no_tracker_injected():
    """The runner must construct a tracker when none was supplied."""
    runner = PipelineRunner(_FakeScraper(), _FakeExtractor())
    assert runner.budget_tracker is None    # pre-run

    _run(runner.run(_task_with_budget(max_pages=5)))
    # A tracker was resolved and stored on the instance for this run.
    assert runner.budget_tracker is not None


def test_large_task_budget_does_not_trip():
    scraper = _FakeScraper()
    runner = PipelineRunner(scraper, _FakeExtractor())

    result = _run(runner.run(_task_with_budget(
        max_pages=100,
        urls=["https://93.184.216.34/a"],
    )))

    assert scraper.fetches == 1
    assert len(result.records) == 1
    assert not any("circuit breaker" in w.lower() for w in result.warnings)


def test_small_task_budget_trips_after_limit():
    """
    Task budget of 1 page but 3 URLs. Exactly one page must be fetched
    and the circuit-breaker warning must appear.
    """
    scraper = _FakeScraper()
    runner = PipelineRunner(scraper, _FakeExtractor())

    result = _run(runner.run(_task_with_budget(
        max_pages=1,
        urls=[
            "https://93.184.216.34/a",
            "https://93.184.216.35/b",
            "https://93.184.216.36/c",
        ],
    )))

    assert scraper.fetches == 1, f"expected 1 fetch, saw {scraper.fetches}"
    assert any("circuit breaker" in w.lower() for w in result.warnings)


def test_trip_preserves_partial_data():
    """Records from pages fetched before the trip must survive."""
    scraper = _FakeScraper()
    runner = PipelineRunner(scraper, _FakeExtractor())

    result = _run(runner.run(_task_with_budget(
        max_pages=1,
        urls=[
            "https://93.184.216.34/a",
            "https://93.184.216.35/b",
        ],
    )))

    # First page was fetched and produced a record before the trip.
    assert len(result.records) >= 1


def test_zero_task_budget_stops_immediately():
    """max_pages=0 should never allow a fetch."""
    scraper = _FakeScraper()
    runner = PipelineRunner(scraper, _FakeExtractor())

    result = _run(runner.run(_task_with_budget(
        max_pages=0,
        urls=["https://93.184.216.34/a"],
    )))

    assert scraper.fetches == 0
    assert any("circuit breaker" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Backward compatibility: injected tracker still wins
# ---------------------------------------------------------------------------

def test_injected_tracker_is_used_unchanged():
    """
    A tracker supplied at construction must be used as-is — the auto-build
    branch must not fire and overwrite it.
    """
    from src.jobs.budget import Budget, BudgetTracker

    injected = BudgetTracker(Budget(max_pages=2))
    scraper = _FakeScraper()

    runner = PipelineRunner(
        scraper, _FakeExtractor(),
        budget_tracker=injected,
    )

    # Task budget says 999, but the injected tracker says 2.
    _run(runner.run(_task_with_budget(
        max_pages=999,
        urls=[
            "https://93.184.216.34/a",
            "https://93.184.216.35/b",
            "https://93.184.216.36/c",
        ],
    )))

    # Same object, not replaced
    assert runner.budget_tracker is injected
    # The injected cap of 2 fired, not the task's 999.
    assert scraper.fetches == 2


def test_no_tracker_and_generous_task_budget_is_a_noop():
    """
    The pre-existing pattern — no injected tracker, task budget large
    enough not to fire — must not produce any budget warnings.
    """
    runner = PipelineRunner(_FakeScraper(), _FakeExtractor())
    result = _run(runner.run(_task_with_budget(
        max_pages=5_000,
        urls=["https://93.184.216.34/a"],
    )))

    assert not any("budget" in w.lower() for w in result.warnings)