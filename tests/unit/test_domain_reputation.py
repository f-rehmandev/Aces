"""Unit tests for domain reputation (spec §20)."""
import pytest

from src.discovery.domain_reputation import (
    DomainReputationStore, DomainStats, apply_reputation,
    NEUTRAL_SCORE, build_supabase_store,
)


# ---------------------------------------------------------------------------
# In-memory store fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    rows: dict[str, dict] = {}
    def fetch(d): return rows.get(d)
    def write(d, stats): rows[d] = stats
    return DomainReputationStore(fetch_fn=fetch, write_fn=write)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def test_unknown_domain_is_neutral(store):
    assert store.score("nope.com") == NEUTRAL_SCORE


def test_successful_domain_scores_high(store):
    store.record_outcome("good.com", records_produced=5)
    assert store.score("good.com") > NEUTRAL_SCORE


def test_failing_domain_scores_low(store):
    for _ in range(3):
        store.record_outcome("bad.com", records_produced=0)
    assert store.score("bad.com") < NEUTRAL_SCORE


def test_productive_domain_scores_higher_than_barely_working(store):
    store.record_outcome("rich.com", records_produced=20)
    store.record_outcome("meh.com", records_produced=1)
    assert store.score("rich.com") > store.score("meh.com")


def test_score_clamped_between_zero_and_one(store):
    for _ in range(50):
        store.record_outcome("great.com", records_produced=100)
    assert 0.0 <= store.score("great.com") <= 1.0

    for _ in range(50):
        store.record_outcome("bad.com", records_produced=0)
    assert 0.0 <= store.score("bad.com") <= 1.0


def test_url_scoring_extracts_domain(store):
    store.record_outcome("shop.example", records_produced=10)
    assert store.score_url("https://shop.example/p/x") == store.score("shop.example")


def test_url_scoring_unparseable_url_returns_neutral(store):
    assert store.score_url("") == NEUTRAL_SCORE
    assert store.score_url(":::: not a url") == NEUTRAL_SCORE


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def test_record_run_batches_by_domain(store):
    store.record_run({
        "https://a.com/x": 3,
        "https://a.com/y": 2,
        "https://b.com/z": 0,
    })
    assert store.get_stats("a.com").attempts == 1
    assert store.get_stats("a.com").total_records == 5
    assert store.get_stats("a.com").successful_runs == 1
    assert store.get_stats("b.com").attempts == 1
    assert store.get_stats("b.com").successful_runs == 0


def test_record_outcome_increments_counters(store):
    store.record_outcome("x.com", 5)
    store.record_outcome("x.com", 3)
    stats = store.get_stats("x.com")
    assert stats.attempts == 2
    assert stats.successful_runs == 2
    assert stats.total_records == 8


def test_record_failure_increments_attempts_not_successes(store):
    store.record_outcome("x.com", 0)
    stats = store.get_stats("x.com")
    assert stats.attempts == 1
    assert stats.successful_runs == 0


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_broken_fetch_returns_neutral():
    def boom(d): raise RuntimeError("db down")
    store = DomainReputationStore(fetch_fn=boom, write_fn=lambda d, s: None)
    assert store.score("anything.com") == NEUTRAL_SCORE


def test_broken_write_does_not_raise():
    calls = []
    def fetch(d): return None
    def broken_write(d, s): raise RuntimeError("read-only replica")
    store = DomainReputationStore(fetch_fn=fetch, write_fn=broken_write)
    # Must not raise
    store.record_outcome("x.com", 5)


# ---------------------------------------------------------------------------
# apply_reputation
# ---------------------------------------------------------------------------

def test_apply_reputation_promotes_good_domains(store):
    store.record_outcome("good.com", 5)
    for _ in range(3):
        store.record_outcome("bad.com", 0)

    urls = ["https://good.com/x", "https://bad.com/y", "https://new.com/z"]
    base = [10.0, 10.0, 10.0]
    ranked = apply_reputation(urls, base, store)
    order = [u for u, _ in ranked]
    assert order.index("https://good.com/x") < order.index("https://new.com/z")
    assert order.index("https://new.com/z") < order.index("https://bad.com/y")


def test_apply_reputation_respects_base_scores(store):
    # No history → neutral → base score decides
    urls = ["https://a.com/x", "https://b.com/y"]
    base = [20.0, 5.0]
    ranked = apply_reputation(urls, base, store)
    assert ranked[0][0] == "https://a.com/x"


def test_apply_reputation_survives_broken_store():
    def boom(d): raise RuntimeError("down")
    broken = DomainReputationStore(fetch_fn=boom, write_fn=lambda d, s: None)
    urls = ["https://a.com/x"]
    ranked = apply_reputation(urls, [10.0], broken)
    assert len(ranked) == 1