"""
Unit tests for crawl mode inside PipelineRunner (§14.3A-C).

Default behavior (follow_links=False) is unchanged: one fetch per
start URL, no link discovery. When the task opts in
(`navigation.follow_links=True`), PipelineRunner drives a
CrawlFrontier: BFS discovery, depth limits, page caps, domain
policy, dedup.

Covers:
    - Default path: no discovered links are followed
    - Crawl path: BFS, depth, page cap, domain, dedup, retries-on-fail
    - _discover_links: absolute, relative, non-http schemes filtered
    - Budget gate applies per fetched page during crawl
    - Trace list contains one entry per fetched URL
"""
import asyncio

import pytest

from src.core.task_spec import (
    FieldSpec,
    Navigation,
    Quality,
    Target,
    TaskSpec,
)
from src.jobs.budget import Budget, BudgetTracker
from src.pipeline_runner import PipelineRunner


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeScraper:
    """Maps URL → HTML. Unknown URLs return a benign not-found page."""
    def __init__(self, html_map=None):
        self.html_map = html_map or {}
        self.calls: list[str] = []

    async def fetch_html(self, url, timeout=None):
        self.calls.append(url)
        return self.html_map.get(url, "<html>not found</html>")

    async def fetch_screenshot(self, url, timeout=None):
        return b"png"


class FakeExtractor:
    """Returns one record per page. Records are numbered in call order."""
    def __init__(self, one_record_per_page: bool = True):
        self.one_record_per_page = one_record_per_page
        self.calls = 0

    def extract_list(self, html, instruction):
        self.calls += 1
        if self.one_record_per_page:
            return [{"title": f"R{self.calls}"}]
        return []

    def extract_from_image(self, img, instr):
        return []


def _task(urls, *, follow_links=False, max_depth=2, max_pages=10,
          same_domain_only=True, task_id="t-crawl"):
    spec = TaskSpec(
        natural_language_prompt="crawl",
        target=Target(start_urls=urls),
        fields=[FieldSpec(name="title")],
        navigation=Navigation(
            follow_links=follow_links,
            max_depth=max_depth,
            max_pages=max_pages,
            same_domain_only=same_domain_only,
        ),
        quality=Quality(min_records=1),
    )
    spec.task_id = task_id
    return spec


def _run(coro):
    return asyncio.run(coro)


def _fetch_map_chain():
    """
    A → B, C
    B → (leaf)
    C → D
    D → (leaf)

    Depth from /a: /b=1, /c=1, /d=2
    """
    return {
        "https://93.184.216.34/a": '<html>A <a href="/b">b</a> <a href="/c">c</a></html>',
        "https://93.184.216.34/b": "<html>B</html>",
        "https://93.184.216.34/c": '<html>C <a href="/d">d</a></html>',
        "https://93.184.216.34/d": "<html>D</html>",
    }


def _urls_seen(result) -> list[str]:
    return sorted(t.url for t in result.security_traces)


# ===========================================================================
# Default behavior: no crawling
# ===========================================================================

def test_default_mode_does_not_follow_links():
    """follow_links=False → only the start URL is fetched."""
    scraper = FakeScraper(_fetch_map_chain())
    runner = PipelineRunner(scraper, FakeExtractor())
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=False,
    )))
    assert scraper.calls == ["https://93.184.216.34/a"]
    assert _urls_seen(result) == ["https://93.184.216.34/a"]


def test_default_mode_multiple_seeds_each_fetched_once():
    scraper = FakeScraper(_fetch_map_chain())
    runner = PipelineRunner(scraper, FakeExtractor())
    _run(runner.run(_task(
        ["https://93.184.216.34/a", "https://93.184.216.34/b"],
        follow_links=False,
    )))
    assert sorted(scraper.calls) == [
        "https://93.184.216.34/a",
        "https://93.184.216.34/b",
    ]


# ===========================================================================
# Crawl mode: BFS discovery
# ===========================================================================

def test_crawl_mode_follows_discovered_links():
    scraper = FakeScraper(_fetch_map_chain())
    runner = PipelineRunner(scraper, FakeExtractor())
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True, max_depth=3,
    )))
    assert set(scraper.calls) == {
        "https://93.184.216.34/a",
        "https://93.184.216.34/b",
        "https://93.184.216.34/c",
        "https://93.184.216.34/d",
    }
    assert len(result.security_traces) == 4
    assert len(result.records) == 4


def test_crawl_mode_respects_max_depth():
    """max_depth=1: /a (0) → /b (1), /c (1); /d (2) not fetched."""
    scraper = FakeScraper(_fetch_map_chain())
    runner = PipelineRunner(scraper, FakeExtractor())
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True, max_depth=1,
    )))
    urls = _urls_seen(result)
    assert urls == [
        "https://93.184.216.34/a",
        "https://93.184.216.34/b",
        "https://93.184.216.34/c",
    ]
    assert "https://93.184.216.34/d" not in urls


def test_crawl_mode_depth_zero_only_fetches_seed():
    scraper = FakeScraper(_fetch_map_chain())
    runner = PipelineRunner(scraper, FakeExtractor())
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True, max_depth=0,
    )))
    assert _urls_seen(result) == ["https://93.184.216.34/a"]


def test_crawl_mode_respects_max_pages():
    """max_pages=2 → stop after 2 fetches even if links remain."""
    scraper = FakeScraper(_fetch_map_chain())
    runner = PipelineRunner(scraper, FakeExtractor())
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True,
        max_depth=10, max_pages=2,
    )))
    # Exactly 2 pages fetched, even though frontier had more queued
    assert len(result.security_traces) == 2
    assert "https://93.184.216.34/a" in _urls_seen(result)


# ===========================================================================
# Domain policy
# ===========================================================================

def test_crawl_mode_same_domain_only_blocks_external():
    html = {
        "https://93.184.216.34/a": (
            '<html>A '
            '<a href="https://external.example/x">ext</a> '
            '<a href="/b">b</a>'
            "</html>"
        ),
        "https://93.184.216.34/b": "<html>B</html>",
        "https://external.example/x": "<html>External</html>",
    }
    scraper = FakeScraper(html)
    runner = PipelineRunner(scraper, FakeExtractor())
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True,
        max_depth=2, same_domain_only=True,
    )))
    urls = _urls_seen(result)
    assert "https://external.example/x" not in urls
    assert "https://93.184.216.34/b" in urls


def test_crawl_mode_cross_domain_allowed_when_disabled():
    html = {
        "https://93.184.216.34/a": (
            '<html>A '
            '<a href="https://external.example/x">ext</a>'
            "</html>"
        ),
        "https://external.example/x": "<html>External</html>",
    }
    scraper = FakeScraper(html)
    runner = PipelineRunner(scraper, FakeExtractor())
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True,
        max_depth=2, same_domain_only=False,
    )))
    urls = _urls_seen(result)
    assert "https://external.example/x" in urls


# ===========================================================================
# Dedup
# ===========================================================================

def test_crawl_mode_does_not_refetch_seen_urls():
    """Two seeds both linking to a common URL → common URL fetched once."""
    html = {
        "https://93.184.216.34/a": '<html>A <a href="/shared">s</a></html>',
        "https://93.184.216.34/b": '<html>B <a href="/shared">s</a></html>',
        "https://93.184.216.34/shared": "<html>Shared</html>",
    }
    scraper = FakeScraper(html)
    runner = PipelineRunner(scraper, FakeExtractor())
    _run(runner.run(_task(
        ["https://93.184.216.34/a", "https://93.184.216.34/b"],
        follow_links=True, max_depth=2,
    )))
    assert scraper.calls.count("https://93.184.216.34/shared") == 1


def test_crawl_mode_handles_cycles():
    """A → B, B → A. Neither should be fetched twice."""
    html = {
        "https://93.184.216.34/a": '<html>A <a href="/b">b</a></html>',
        "https://93.184.216.34/b": '<html>B <a href="/a">a</a></html>',
    }
    scraper = FakeScraper(html)
    runner = PipelineRunner(scraper, FakeExtractor())
    _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True, max_depth=10,
    )))
    assert scraper.calls.count("https://93.184.216.34/a") == 1
    assert scraper.calls.count("https://93.184.216.34/b") == 1


# ===========================================================================
# Link discovery helper
# ===========================================================================

def test_discover_links_extracts_absolute_urls():
    html = '<html><a href="https://x.example/a">a</a><a href="https://x.example/b">b</a></html>'
    links = PipelineRunner._discover_links(html, "https://x.example/")
    assert set(links) == {"https://x.example/a", "https://x.example/b"}


def test_discover_links_resolves_relative_urls():
    html = '<html><a href="/a">a</a><a href="../b">b</a><a href="c">c</a></html>'
    links = PipelineRunner._discover_links(html, "https://x.example/dir/page")
    assert set(links) == {
        "https://x.example/a",
        "https://x.example/b",
        "https://x.example/dir/c",
    }


def test_discover_links_ignores_non_http_schemes():
    html = (
        '<html>'
        '<a href="mailto:x@y.com">m</a>'
        '<a href="javascript:void(0)">js</a>'
        '<a href="tel:+123">t</a>'
        '<a href="https://x.example/keep">k</a>'
        "</html>"
    )
    links = PipelineRunner._discover_links(html, "https://x.example/")
    assert links == ["https://x.example/keep"]


def test_discover_links_empty_html():
    assert PipelineRunner._discover_links("", "https://x.example/") == []


def test_discover_links_no_anchors():
    html = "<html><p>no links here</p></html>"
    assert PipelineRunner._discover_links(html, "https://x.example/") == []


# ===========================================================================
# Budget integration
# ===========================================================================

def test_crawl_respects_budget_circuit_breaker():
    """A budget breaker stops the crawl mid-frontier."""
    scraper = FakeScraper(_fetch_map_chain())
    tracker = BudgetTracker(Budget(max_pages=2))
    runner = PipelineRunner(
        scraper, FakeExtractor(), budget_tracker=tracker,
    )
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True,
        max_depth=5, max_pages=10,
    )))
    # Only 2 pages fetched (budget cap)
    assert len(result.security_traces) == 2
    assert any("circuit breaker" in w.lower() for w in result.warnings)


def test_crawl_consumes_one_budget_per_page():
    scraper = FakeScraper(_fetch_map_chain())
    tracker = BudgetTracker(Budget(max_pages=100))
    runner = PipelineRunner(
        scraper, FakeExtractor(), budget_tracker=tracker,
    )
    _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True, max_depth=3,
    )))
    # 4 pages fetched → 4 consumed
    assert tracker.budget.pages_used == 4


# ===========================================================================
# Trace shape
# ===========================================================================

def test_crawl_produces_one_trace_per_fetched_url():
    scraper = FakeScraper(_fetch_map_chain())
    runner = PipelineRunner(scraper, FakeExtractor())
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True, max_depth=3,
    )))
    assert len(result.security_traces) == len(scraper.calls)
    for tr in result.security_traces:
        assert tr.url
        assert tr.ssrf_allowed  # public IPs


# ===========================================================================
# Records flow through the pipeline
# ===========================================================================

def test_crawl_records_are_produced_and_triangulated():
    """
    Multiple pages produce multiple records. With one page per domain
    and multiple domains, triangulation kicks in per identity — but
    our fake extractor gives each record a distinct title, so we get
    N distinct records.
    """
    scraper = FakeScraper(_fetch_map_chain())
    extractor = FakeExtractor(one_record_per_page=True)
    runner = PipelineRunner(scraper, extractor)
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True, max_depth=3,
    )))
    # Each of the 4 pages yielded 1 record with a distinct title.
    # No triangulation collisions → 4 output records.
    assert len(result.records) == 4


def test_crawl_with_no_records_still_completes():
    scraper = FakeScraper(_fetch_map_chain())
    extractor = FakeExtractor(one_record_per_page=False)
    runner = PipelineRunner(scraper, extractor)
    result = _run(runner.run(_task(
        ["https://93.184.216.34/a"], follow_links=True, max_depth=3,
    )))
    assert result.records == []
    # But every page was still fetched
    assert len(result.security_traces) == 4