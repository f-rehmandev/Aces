"""Unit tests for CrawlFrontier (spec §14.3A)."""
import pytest

from src.crawl.frontier import CrawlFrontier
from src.crawl.types import CrawlPolicy, CrawlState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def frontier():
    return CrawlFrontier()


# ---------------------------------------------------------------------------
# Empty / add basics
# ---------------------------------------------------------------------------

def test_empty_frontier(frontier):
    assert len(frontier) == 0
    assert frontier.next_target() is None
    assert not frontier.has_pending()


def test_add_returns_queued_target(frontier):
    t = frontier.add("https://example.com/a")
    assert t is not None
    assert t.url == "https://example.com/a"
    assert t.state == CrawlState.QUEUED
    assert len(frontier) == 1


def test_add_canonicalizes_url(frontier):
    t = frontier.add("HTTPS://EXAMPLE.COM/a?utm_source=x#frag")
    assert t.url == "https://example.com/a"


def test_add_deduplicates_after_canonicalization(frontier):
    assert frontier.add("https://example.com/a") is not None
    assert frontier.add("https://example.com/a?utm_source=x") is None
    assert len(frontier) == 1


def test_add_empty_returns_none(frontier):
    assert frontier.add("") is None
    assert frontier.add(None) is None


def test_add_rejects_non_http_scheme(frontier):
    assert frontier.add("ftp://example.com/a") is None
    assert frontier.add("file:///etc/passwd") is None


def test_add_many_deduplicates(frontier):
    added = frontier.add_many([
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/a",
    ])
    assert len(added) == 2
    assert len(frontier) == 2


def test_contains_uses_canonical_form(frontier):
    frontier.add("https://example.com/a")
    assert "https://example.com/a" in frontier
    assert "https://example.com/a?utm_source=x" in frontier
    assert "https://example.com/b" not in frontier


def test_get_returns_matching_target(frontier):
    frontier.add("https://example.com/a")
    assert frontier.get("https://example.com/a") is not None
    assert frontier.get("https://example.com/never-added") is None


# ---------------------------------------------------------------------------
# Policy enforcement
# ---------------------------------------------------------------------------

def test_add_rejects_different_domain_by_default(frontier):
    frontier.add("https://example.com/start")
    assert frontier.add(
        "https://other.com/x",
        parent_url="https://example.com/start",
    ) is None


def test_add_allows_different_domain_when_policy_disables_it():
    f = CrawlFrontier(policy=CrawlPolicy(same_domain_only=False))
    f.add("https://example.com/start")
    t = f.add("https://other.com/x", parent_url="https://example.com/start")
    assert t is not None


def test_add_honors_exclude_patterns():
    f = CrawlFrontier(policy=CrawlPolicy(exclude_patterns=[r"/private/"]))
    assert f.add("https://example.com/private/x") is None
    assert f.add("https://example.com/public/x") is not None


def test_add_honors_include_patterns():
    f = CrawlFrontier(policy=CrawlPolicy(include_patterns=[r"/products/"]))
    assert f.add("https://example.com/about") is None
    assert f.add("https://example.com/products/x") is not None


def test_add_honors_depth_limit():
    f = CrawlFrontier(policy=CrawlPolicy(max_depth=2))
    assert f.add("https://example.com/a", depth=2) is not None
    assert f.add("https://example.com/b", depth=3) is None


# ---------------------------------------------------------------------------
# next_target
# ---------------------------------------------------------------------------

def test_next_target_marks_fetching(frontier):
    t = frontier.add("https://example.com/a")
    got = frontier.next_target()
    assert got is t
    assert t.state == CrawlState.FETCHING
    assert t.attempts == 1
    assert t.last_attempt_at


def test_next_target_fifo_order(frontier):
    a = frontier.add("https://example.com/a")
    b = frontier.add("https://example.com/b")
    c = frontier.add("https://example.com/c")
    assert frontier.next_target() is a
    assert frontier.next_target() is b
    assert frontier.next_target() is c


def test_next_target_returns_none_when_all_fetching(frontier):
    frontier.add("https://example.com/a")
    frontier.next_target()
    assert frontier.next_target() is None


def test_next_target_returns_none_when_empty(frontier):
    assert frontier.next_target() is None


# ---------------------------------------------------------------------------
# Terminal transitions
# ---------------------------------------------------------------------------

def test_mark_processed(frontier):
    t = frontier.add("https://example.com/a")
    frontier.next_target()
    frontier.mark_processed(t, records_extracted=10)
    assert t.state == CrawlState.PROCESSED
    assert t.records_extracted == 10


def test_mark_failed(frontier):
    t = frontier.add("https://example.com/a")
    frontier.next_target()
    frontier.mark_failed(t, error="404")
    assert t.state == CrawlState.FAILED
    assert t.error == "404"


def test_mark_skipped(frontier):
    t = frontier.add("https://example.com/a")
    frontier.next_target()
    frontier.mark_skipped(t, reason="robots.txt")
    assert t.state == CrawlState.SKIPPED
    assert t.notes == "robots.txt"


def test_mark_policy_refused(frontier):
    t = frontier.add("https://example.com/a")
    frontier.next_target()
    frontier.mark_policy_refused(t, reason="compliance")
    assert t.state == CrawlState.POLICY_REFUSED


@pytest.mark.parametrize("mark", [
    "mark_processed", "mark_failed", "mark_skipped", "mark_policy_refused",
])
def test_marking_terminal_twice_raises(frontier, mark):
    t = frontier.add("https://example.com/a")
    frontier.next_target()
    getattr(frontier, mark)(t)
    with pytest.raises(ValueError):
        getattr(frontier, mark)(t)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

def test_retry_moves_to_retry_wait():
    f = CrawlFrontier(policy=CrawlPolicy(max_attempts=3))
    t = f.add("https://example.com/a")
    f.next_target()
    f.mark_retry(t, error="timeout", retry_after_seconds=100)
    assert t.state == CrawlState.RETRY_WAIT
    assert t.error == "timeout"


def test_retry_promotes_after_wait_with_fake_clock():
    now = [0.0]
    f = CrawlFrontier(
        policy=CrawlPolicy(max_attempts=3),
        clock=lambda: now[0],
    )
    t = f.add("https://example.com/a")
    f.next_target()
    f.mark_retry(t, retry_after_seconds=10)
    assert t.state == CrawlState.RETRY_WAIT

    assert f.next_target() is None

    now[0] = 11.0
    got = f.next_target()
    assert got is t
    assert t.state == CrawlState.FETCHING
    assert t.attempts == 2


def test_retry_exhaustion_marks_failed():
    f = CrawlFrontier(policy=CrawlPolicy(max_attempts=2))
    t = f.add("https://example.com/a")

    f.next_target()
    f.mark_retry(t, retry_after_seconds=0)
    f.next_target()
    f.mark_retry(t)
    assert t.state == CrawlState.FAILED


def test_retry_multiple_targets_promote_independently():
    now = [0.0]
    f = CrawlFrontier(
        policy=CrawlPolicy(max_attempts=5),
        clock=lambda: now[0],
    )
    a = f.add("https://example.com/a")
    b = f.add("https://example.com/b")
    f.next_target()
    f.mark_retry(a, retry_after_seconds=100)
    f.next_target()
    f.mark_retry(b, retry_after_seconds=10)

    now[0] = 50.0
    assert f.next_target() is b
    assert b.state == CrawlState.FETCHING

    now[0] = 150.0
    assert f.next_target() is a


# ---------------------------------------------------------------------------
# Stats / pending
# ---------------------------------------------------------------------------

def test_stats_counts_by_state(frontier):
    a = frontier.add("https://example.com/a")
    b = frontier.add("https://example.com/b")
    frontier.add("https://example.com/c")
    frontier.next_target()
    frontier.mark_processed(a)
    frontier.next_target()
    frontier.mark_failed(b, error="x")

    stats = frontier.stats()
    assert stats["PROCESSED"] == 1
    assert stats["FAILED"] == 1
    assert stats["QUEUED"] == 1


def test_has_pending_true_while_queued(frontier):
    frontier.add("https://example.com/a")
    assert frontier.has_pending()


def test_has_pending_true_while_retry_wait():
    f = CrawlFrontier(policy=CrawlPolicy(max_attempts=3))
    t = f.add("https://example.com/a")
    f.next_target()
    f.mark_retry(t, retry_after_seconds=100)
    assert f.has_pending()


def test_has_pending_false_when_all_terminal(frontier):
    t = frontier.add("https://example.com/a")
    frontier.next_target()
    frontier.mark_processed(t)
    assert not frontier.has_pending()


# ---------------------------------------------------------------------------
# Checkpoint restore
# ---------------------------------------------------------------------------

def test_snapshot_preserves_order():
    f = CrawlFrontier()
    f.add_many([
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ])
    urls = [t.url for t in f.snapshot()]
    assert urls == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]


def test_load_from_checkpoint_resets_fetching_to_queued():
    f = CrawlFrontier()
    f.add("https://example.com/a")
    f.add("https://example.com/b")
    f.next_target()

    f2 = CrawlFrontier()
    f2.load_from_checkpoint(f.snapshot())
    assert len(f2) == 2
    assert f2.get("https://example.com/a").state == CrawlState.QUEUED
    assert f2.get("https://example.com/b").state == CrawlState.QUEUED


def test_load_from_checkpoint_replaces_existing():
    f = CrawlFrontier()
    f.add("https://example.com/old")
    f.load_from_checkpoint([])
    assert len(f) == 0


def test_load_from_checkpoint_retains_terminal_states():
    f = CrawlFrontier()
    a = f.add("https://example.com/a")
    f.next_target()
    f.mark_processed(a, records_extracted=3)

    f2 = CrawlFrontier()
    f2.load_from_checkpoint(f.snapshot())
    assert f2.get("https://example.com/a").state == CrawlState.PROCESSED
    assert f2.get("https://example.com/a").records_extracted == 3