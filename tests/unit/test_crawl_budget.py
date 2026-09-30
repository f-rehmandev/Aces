"""Unit tests for crawl budget (spec §14.1, §14.3)."""
import pytest

from src.navigation.budget import CrawlBudget


# --- no limits ----------------------------------------------------------

def test_no_limits_always_continues():
    b = CrawlBudget()
    for _ in range(100):
        b.consume_page(1000)
    ok, reason = b.can_continue()
    assert ok and reason == "within_budget"


# --- pages --------------------------------------------------------------

def test_max_pages_stops_exactly_at_limit():
    b = CrawlBudget(max_pages=2)
    b.consume_page()
    assert b.can_continue()[0]
    b.consume_page()
    ok, reason = b.can_continue()
    assert not ok and reason == "max_pages_reached"


def test_max_pages_zero_immediately_stops():
    b = CrawlBudget(max_pages=0)
    ok, reason = b.can_continue()
    assert not ok and reason == "max_pages_reached"


# --- bytes --------------------------------------------------------------

def test_max_bytes_stops_after_crossing():
    b = CrawlBudget(max_bytes=150)
    b.consume_page(100)
    assert b.can_continue()[0]
    b.consume_page(60)   # total 160
    ok, reason = b.can_continue()
    assert not ok and reason == "max_bytes_reached"


def test_consume_page_with_zero_bytes_still_counts_as_page():
    b = CrawlBudget(max_pages=1)
    b.consume_page()
    assert b.pages_fetched == 1
    assert b.bytes_fetched == 0


def test_negative_byte_count_raises():
    b = CrawlBudget()
    with pytest.raises(ValueError):
        b.consume_page(-1)


# --- wall clock ---------------------------------------------------------

def test_max_wall_clock_stops_when_elapsed():
    fake = [100.0]
    b = CrawlBudget(max_wall_clock_seconds=5, now_fn=lambda: fake[0])
    assert b.can_continue()[0]
    fake[0] = 103.0
    assert b.can_continue()[0]
    fake[0] = 105.0
    ok, reason = b.can_continue()
    assert not ok and reason == "max_wall_clock_reached"


def test_elapsed_uses_injected_now():
    fake = [10.0]
    b = CrawlBudget(now_fn=lambda: fake[0])
    fake[0] = 13.5
    assert abs(b.elapsed_seconds() - 3.5) < 1e-9


# --- precedence ---------------------------------------------------------

def test_pages_checked_before_bytes():
    # If both limits would be exceeded, pages wins (first check in code).
    b = CrawlBudget(max_pages=1, max_bytes=10)
    b.consume_page(100)
    ok, reason = b.can_continue()
    assert not ok and reason == "max_pages_reached"


# --- snapshot -----------------------------------------------------------

def test_snapshot_reports_state_and_limits():
    b = CrawlBudget(max_pages=10, max_bytes=1000, max_wall_clock_seconds=60)
    b.consume_page(200)
    snap = b.snapshot()
    assert snap["pages_fetched"] == 1
    assert snap["bytes_fetched"] == 200
    assert snap["limits"]["max_pages"] == 10
    assert snap["limits"]["max_bytes"] == 1000
    assert snap["limits"]["max_wall_clock_seconds"] == 60
    assert snap["elapsed_seconds"] >= 0