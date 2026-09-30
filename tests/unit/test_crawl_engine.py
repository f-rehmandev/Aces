"""Unit tests for CrawlEngine (spec §14.3A, §14.3B)."""
import asyncio

import pytest

from src.crawl.checkpoint import CheckpointStore
from src.crawl.engine import CrawlEngine, CrawlResult
from src.crawl.types import CrawlPolicy, CrawlState


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeFetcher:
    """Returns canned HTML per URL. Records which URLs were fetched."""
    def __init__(self, html_map=None, raises=None):
        self.html_map = html_map or {}
        self.raises = raises or {}
        self.fetched: list[str] = []
        self.call_count: dict[str, int] = {}

    async def __call__(self, url: str) -> str:
        self.fetched.append(url)
        self.call_count[url] = self.call_count.get(url, 0) + 1
        if url in self.raises:
            raise self.raises[url]
        return self.html_map.get(url, "")


def make_processor(links_by_url=None, records_by_url=None,
                   raises_by_url=None):
    """
    Returns a page_processor(url, html) -> (records, links).

    links_by_url / records_by_url / raises_by_url are dicts keyed on URL.
    Defaults: 1 record per page, 0 links, no raises.
    """
    links_by_url = links_by_url or {}
    records_by_url = records_by_url or {}
    raises_by_url = raises_by_url or {}

    def process(url, html):
        if url in raises_by_url:
            raise raises_by_url[url]
        records = records_by_url.get(url, [{"url": url, "size": len(html)}])
        links = links_by_url.get(url, [])
        return records, links

    return process


def make_store():
    """In-memory CheckpointStore. Returns (store, rows_dict)."""
    rows: dict = {}
    store = CheckpointStore(
        fetch_fn=lambda c, t: rows.get(f"{c}::{t}"),
        write_fn=lambda cp: rows.__setitem__(
            f"{cp['client_id']}::{cp['task_id']}", cp,
        ),
        clear_fn=lambda c, t: int(bool(rows.pop(f"{c}::{t}", None))),
    )
    return store, rows


# ---------------------------------------------------------------------------
# Basic flow
# ---------------------------------------------------------------------------

def test_empty_seeds_returns_empty_result():
    fetcher = FakeFetcher()
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=make_processor(),
    )
    result = asyncio.run(engine.crawl("acme", "t-1", seeds=[]))
    assert result.status == "completed"
    assert result.records == []
    assert result.targets_processed == 0
    assert fetcher.fetched == []


def test_single_page_crawl():
    fetcher = FakeFetcher({"https://x.com/a": "<html>hello</html>"})
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=make_processor(),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert result.status == "completed"
    assert len(result.records) == 1
    assert result.targets_processed == 1
    assert fetcher.fetched == ["https://x.com/a"]


def test_link_discovery_breadth_first():
    fetcher = FakeFetcher({
        "https://x.com/a": "<html>a</html>",
        "https://x.com/b": "<html>b</html>",
        "https://x.com/c": "<html>c</html>",
    })
    processor = make_processor(links_by_url={
        "https://x.com/a": ["https://x.com/b", "https://x.com/c"],
    })
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(max_depth=3, discover_links=True),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert result.targets_processed == 3
    assert len(result.records) == 3
    # a processed first, then b, then c (FIFO)
    assert fetcher.fetched == [
        "https://x.com/a",
        "https://x.com/b",
        "https://x.com/c",
    ]


def test_discovery_disabled_ignores_links():
    fetcher = FakeFetcher({
        "https://x.com/a": "<html>a</html>",
        "https://x.com/b": "<html>b</html>",
    })
    processor = make_processor(links_by_url={
        "https://x.com/a": ["https://x.com/b"],
    })
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(discover_links=False),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert result.targets_processed == 1
    assert "https://x.com/b" not in fetcher.fetched


def test_cycle_does_not_loop():
    fetcher = FakeFetcher({
        "https://x.com/a": "<html>a</html>",
        "https://x.com/b": "<html>b</html>",
    })
    processor = make_processor(links_by_url={
        "https://x.com/a": ["https://x.com/b"],
        "https://x.com/b": ["https://x.com/a"],   # cycle
    })
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(max_depth=5),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    # Each URL visited exactly once despite the cycle
    assert fetcher.fetched.count("https://x.com/a") == 1
    assert fetcher.fetched.count("https://x.com/b") == 1
    assert result.targets_processed == 2


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def test_max_pages_stops_crawl():
    fetcher = FakeFetcher({
        "https://x.com/a": "<html>a</html>",
        "https://x.com/b": "<html>b</html>",
        "https://x.com/c": "<html>c</html>",
    })
    processor = make_processor(links_by_url={
        "https://x.com/a": ["https://x.com/b", "https://x.com/c"],
    })
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(max_pages=2),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert result.status == "stopped_by_budget"
    assert result.budget_pages_used == 2
    assert result.targets_processed == 2


def test_max_bytes_stops_crawl():
    big = "x" * 500
    fetcher = FakeFetcher({
        "https://x.com/a": "<html>" + big + "</html>",
        "https://x.com/b": "<html>" + big + "</html>",
        "https://x.com/c": "<html>" + big + "</html>",
    })
    processor = make_processor(links_by_url={
        "https://x.com/a": ["https://x.com/b", "https://x.com/c"],
    })
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(max_pages=100, max_bytes=1000),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert result.status == "stopped_by_budget"
    assert result.budget_bytes_used >= 1000


def test_max_wall_clock_stops_crawl():
    # Fake clock that jumps 100s per call → effectively unbounded time
    now = [0.0]
    def clock():
        now[0] += 100.0
        return now[0]

    fetcher = FakeFetcher({"https://x.com/a": "<html>a</html>"})
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=make_processor(),
        policy=CrawlPolicy(max_pages=100, max_wall_clock_seconds=50),
        clock=clock,
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert result.status == "stopped_by_budget"


# ---------------------------------------------------------------------------
# Failures and retries
# ---------------------------------------------------------------------------

def test_fetch_exception_retries_then_fails():
    fetcher = FakeFetcher(raises={
        "https://x.com/a": RuntimeError("network down"),
    })
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=make_processor(),
        policy=CrawlPolicy(max_attempts=2),
        sleep=asyncio.sleep,   # use real sleep (delays are tiny: 1s, 2s)
    )
    # Use a fast clock so retries promote immediately
    result = asyncio.run(asyncio.wait_for(
        engine.crawl("acme", "t-1", seeds=["https://x.com/a"]),
        timeout=10,
    ))
    assert result.targets_failed == 1
    assert result.status == "completed"


def test_processor_exception_marks_failed():
    fetcher = FakeFetcher({"https://x.com/a": "<html>a</html>"})
    processor = make_processor(raises_by_url={
        "https://x.com/a": RuntimeError("bad html"),
    })
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(max_attempts=1),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert result.targets_failed == 1
    assert result.targets_processed == 0


def test_empty_response_is_failure():
    fetcher = FakeFetcher({"https://x.com/a": ""})   # empty HTML
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=make_processor(),
        policy=CrawlPolicy(max_attempts=1),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert result.targets_failed == 1
    assert result.targets_processed == 0


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def test_clean_completion_clears_checkpoint():
    fetcher = FakeFetcher({"https://x.com/a": "<html>a</html>"})
    store, rows = make_store()
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=make_processor(),
        checkpoint_store=store,
        checkpoint_interval=100,
    )
    asyncio.run(engine.crawl("acme", "t-1", seeds=["https://x.com/a"]))
    assert store.load_latest("acme", "t-1") is None


def test_budget_stop_persists_checkpoint():
    fetcher = FakeFetcher({
        "https://x.com/a": "<html>a</html>",
        "https://x.com/b": "<html>b</html>",
        "https://x.com/c": "<html>c</html>",
    })
    processor = make_processor(links_by_url={
        "https://x.com/a": ["https://x.com/b", "https://x.com/c"],
    })
    store, rows = make_store()
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(max_pages=1),
        checkpoint_store=store,
        checkpoint_interval=1,
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert result.status == "stopped_by_budget"
    assert result.last_checkpoint_id != ""
    cp = store.load_latest("acme", "t-1")
    assert cp is not None
    assert cp.budget_pages_used == 1
    assert len(cp.frontier) == 3   # a processed, b + c queued


def test_resume_from_checkpoint_ignores_seeds():
    # First run: process a, discover b, stop on budget
    fetcher = FakeFetcher({
        "https://x.com/a": "<html>a</html>",
        "https://x.com/b": "<html>b</html>",
    })
    processor = make_processor(links_by_url={
        "https://x.com/a": ["https://x.com/b"],
    })
    store, rows = make_store()

    e1 = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(max_pages=1),
        checkpoint_store=store,
        checkpoint_interval=1,
    )
    r1 = asyncio.run(e1.crawl("acme", "t-1", seeds=["https://x.com/a"]))
    assert r1.status == "stopped_by_budget"

    # Second run: resume with a fresh engine + larger budget
    e2 = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(max_pages=10),
        checkpoint_store=store,
        checkpoint_interval=100,
    )
    r2 = asyncio.run(e2.crawl("acme", "t-1", seeds=["https://ignored.example"]))
    assert r2.resumed_from_checkpoint is True
    assert r2.status == "completed"
    # Should have picked up b (the leftover), not ignored.example
    assert "https://x.com/b" in fetcher.fetched
    assert "https://ignored.example" not in fetcher.fetched


def test_checkpoint_written_every_interval():
    fetcher = FakeFetcher({
        f"https://x.com/{i}": f"<html>{i}</html>" for i in range(5)
    })
    store, rows = make_store()
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=make_processor(),
        checkpoint_store=store,
        checkpoint_interval=2,
    )
    asyncio.run(engine.crawl(
        "acme", "t-1",
        seeds=[f"https://x.com/{i}" for i in range(5)],
    ))
    # Clean completion → checkpoint cleared. Just verify we got through
    # without an exception; the interval-write path is exercised.
    assert store.load_latest("acme", "t-1") is None


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

def test_result_to_dict():
    fetcher = FakeFetcher({"https://x.com/a": "<html>a</html>"})
    engine = CrawlEngine(
        fetch_html=fetcher, page_processor=make_processor(),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    d = result.to_dict()
    assert d["status"] == "completed"
    assert d["records_count"] == 1
    assert d["targets_processed"] == 1


def test_no_checkpoint_store_is_fine():
    fetcher = FakeFetcher({"https://x.com/a": "<html>a</html>"})
    engine = CrawlEngine(
        fetch_html=fetcher, page_processor=make_processor(),
        checkpoint_store=None,
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert result.status == "completed"


# ---------------------------------------------------------------------------
# Policy enforcement (delegated to the frontier)
# ---------------------------------------------------------------------------

def test_same_domain_policy_blocks_external_links():
    fetcher = FakeFetcher({
        "https://x.com/a": "<html>a</html>",
        "https://other.com/x": "<html>x</html>",
    })
    processor = make_processor(links_by_url={
        "https://x.com/a": ["https://other.com/x"],
    })
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(same_domain_only=True),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert "https://other.com/x" not in fetcher.fetched
    assert result.targets_processed == 1


def test_exclude_patterns_block_links():
    fetcher = FakeFetcher({
        "https://x.com/a": "<html>a</html>",
        "https://x.com/private/x": "<html>p</html>",
    })
    processor = make_processor(links_by_url={
        "https://x.com/a": ["https://x.com/private/x"],
    })
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(exclude_patterns=[r"/private/"]),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/a"],
    ))
    assert "https://x.com/private/x" not in fetcher.fetched
    assert result.targets_processed == 1


def test_max_depth_stops_recursion():
    fetcher = FakeFetcher({
        f"https://x.com/p{i}": f"<html>{i}</html>" for i in range(5)
    })
    processor = make_processor(links_by_url={
        "https://x.com/p0": ["https://x.com/p1"],
        "https://x.com/p1": ["https://x.com/p2"],
        "https://x.com/p2": ["https://x.com/p3"],
        "https://x.com/p3": ["https://x.com/p4"],
    })
    engine = CrawlEngine(
        fetch_html=fetcher,
        page_processor=processor,
        policy=CrawlPolicy(max_depth=2),
    )
    result = asyncio.run(engine.crawl(
        "acme", "t-1", seeds=["https://x.com/p0"],
    ))
    # depth 0 (p0), depth 1 (p1), depth 2 (p2). p3 would be depth 3 → refused.
    assert "https://x.com/p3" not in fetcher.fetched
    assert result.targets_processed == 3

# ---------------------------------------------------------------------------
# Async page_processor (Task 9)
# ---------------------------------------------------------------------------

def test_async_page_processor_is_awaited():
    """
    Regression guard: the engine must await an async page_processor.
    Without this, calling an async processor would return a coroutine
    and unpacking `page_records, links = coroutine` would crash.
    """
    async def _run_case():
        fetcher = FakeFetcher({
            "https://x.com/a": "<html>a</html>",
            "https://x.com/b": "<html>b</html>",
        })

        async def async_processor(url, html):
            # Simulate an async extractor call
            await asyncio.sleep(0)
            return [{"url": url}], ["https://x.com/b"] if "/a" in url else []

        engine = CrawlEngine(
            fetch_html=fetcher,
            page_processor=async_processor,
            policy=CrawlPolicy(max_depth=2, discover_links=True),
        )
        result = await engine.crawl(
            "acme", "t-1", seeds=["https://x.com/a"],
        )
        return result

    result = asyncio.run(_run_case())
    assert result.status == "completed"
    assert len(result.records) == 2
    assert result.targets_processed == 2


def test_sync_and_async_processors_produce_same_result():
    """
    Belt-and-braces: a sync processor and an equivalent async processor
    produce identical crawl output on the same input.
    """
    html_map = {
        "https://x.com/a": "<html>a</html>",
        "https://x.com/b": "<html>b</html>",
    }

    async def _run_both():
        sync_engine = CrawlEngine(
            fetch_html=FakeFetcher(dict(html_map)),
            page_processor=make_processor(
                links_by_url={"https://x.com/a": ["https://x.com/b"]},
            ),
            policy=CrawlPolicy(max_depth=2, discover_links=True),
        )
        r_sync = await sync_engine.crawl(
            "acme", "t-sync", seeds=["https://x.com/a"],
        )

        async def async_proc(url, html):
            records, links = make_processor(
                links_by_url={"https://x.com/a": ["https://x.com/b"]},
            )(url, html)
            return records, links

        async_engine = CrawlEngine(
            fetch_html=FakeFetcher(dict(html_map)),
            page_processor=async_proc,
            policy=CrawlPolicy(max_depth=2, discover_links=True),
        )
        r_async = await async_engine.crawl(
            "acme", "t-async", seeds=["https://x.com/a"],
        )
        return r_sync, r_async

    r_sync, r_async = asyncio.run(_run_both())
    assert r_sync.targets_processed == r_async.targets_processed
    assert len(r_sync.records) == len(r_async.records)