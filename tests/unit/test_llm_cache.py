"""Unit tests for the LLM response cache (spec §42)."""
import time

import pytest

from src.llm.cache import (
    CacheEntry,
    DEFAULT_TTL_SECONDS,
    LLMCache,
    _canonicalize_options,
    _normalize,
    compute_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache():
    """Fresh in-memory cache per test."""
    c = LLMCache(db_path=":memory:", ttl_seconds=3600, max_entries=100)
    yield c
    c.close()


@pytest.fixture
def fake_clock():
    """Mutable clock. Tests advance `clock['now']` to travel in time."""
    return {"now": 1000.0}


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------

def test_key_is_deterministic():
    assert compute_key("hello world") == compute_key("hello world")


def test_key_differs_on_different_prompts():
    assert compute_key("hello") != compute_key("goodbye")


def test_key_normalizes_whitespace():
    a = compute_key("hello   world")
    b = compute_key("  hello world  ")
    c = compute_key("hello\nworld")
    assert a == b == c


def test_key_differs_on_model():
    assert compute_key("hi", model="gemini") != compute_key("hi", model="groq")


def test_key_differs_on_options():
    a = compute_key("hi", options={"temperature": 0.1})
    b = compute_key("hi", options={"temperature": 0.9})
    assert a != b


def test_key_ignores_none_options():
    a = compute_key("hi", options={"temperature": 0.1, "top_p": None})
    b = compute_key("hi", options={"temperature": 0.1})
    assert a == b


def test_key_differs_on_template_version():
    assert (compute_key("hi", prompt_template_version="v1")
            != compute_key("hi", prompt_template_version="v2"))


def test_key_differs_on_dom_hash():
    assert (compute_key("hi", dom_hash="abc")
            != compute_key("hi", dom_hash="def"))


def test_normalize_helper():
    assert _normalize("  a   b  c  ") == "a b c"
    assert _normalize("") == ""
    assert _normalize(None) == ""


def test_canonicalize_options_drops_none():
    assert _canonicalize_options({"a": 1, "b": None}) == {"a": 1}


# ---------------------------------------------------------------------------
# Basic put / get
# ---------------------------------------------------------------------------

def test_get_returns_none_for_missing(cache):
    assert cache.get("nope") is None


def test_put_then_get(cache):
    cache.put("k1", "the prompt", "the response", provider="gemini")
    entry = cache.get("k1")
    assert isinstance(entry, CacheEntry)
    assert entry.response_text == "the response"
    assert entry.provider == "gemini"
    assert entry.prompt == "the prompt"


def test_put_overwrites_existing(cache):
    cache.put("k1", "p", "first")
    cache.put("k1", "p", "second")
    assert cache.get("k1").response_text == "second"


def test_get_increments_hit_count(cache):
    cache.put("k1", "p", "r")
    assert cache.get("k1").hit_count == 1
    assert cache.get("k1").hit_count == 2
    assert cache.get("k1").hit_count == 3
    assert cache.stats()["hits"] == 3


def test_miss_then_hit_stats(cache):
    cache.get("nope")
    cache.put("k1", "p", "r")
    cache.get("k1")
    s = cache.stats()
    assert s["misses"] == 1
    assert s["hits"] == 1
    assert s["hit_rate"] == 0.5


def test_entry_age_is_computed(cache):
    cache.put("k1", "p", "r")
    assert cache.get("k1").age_seconds >= 0.0


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

def test_ttl_expiration(fake_clock):
    c = LLMCache(
        db_path=":memory:", ttl_seconds=10,
        clock=lambda: fake_clock["now"],
    )
    try:
        c.put("k1", "p", "r")
        assert c.get("k1") is not None

        fake_clock["now"] += 11
        assert c.get("k1") is None
        assert c.stats()["expired"] >= 1
    finally:
        c.close()


def test_ttl_just_under_keeps_entry(fake_clock):
    c = LLMCache(
        db_path=":memory:", ttl_seconds=10,
        clock=lambda: fake_clock["now"],
    )
    try:
        c.put("k1", "p", "r")
        fake_clock["now"] += 9
        assert c.get("k1") is not None
    finally:
        c.close()


def test_evict_expired_sweeps(fake_clock):
    c = LLMCache(
        db_path=":memory:", ttl_seconds=10,
        clock=lambda: fake_clock["now"],
    )
    try:
        c.put("k1", "p", "r")
        fake_clock["now"] += 100
        assert c.evict_expired() == 1
        assert c.evict_expired() == 0
    finally:
        c.close()


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

def test_lru_eviction_on_entry_count(fake_clock):
    c = LLMCache(
        db_path=":memory:", ttl_seconds=99999, max_entries=3,
        clock=lambda: fake_clock["now"],
    )
    try:
        for i in range(3):
            fake_clock["now"] = 1000 + i
            c.put(f"k{i}", f"p{i}", f"r{i}")

        # Touch k0 so it becomes most-recently-used.
        fake_clock["now"] = 1010
        c.get("k0")

        # Add a 4th; the LRU entry (k1) should be evicted.
        fake_clock["now"] = 1020
        c.put("k3", "p3", "r3")

        assert c.get("k0") is not None
        assert c.get("k1") is None
        assert c.get("k2") is not None
        assert c.get("k3") is not None
        assert c.stats()["evicted"] >= 1
    finally:
        c.close()


def test_lru_eviction_on_byte_cap(fake_clock):
    c = LLMCache(
        db_path=":memory:", ttl_seconds=99999, max_entries=100,
        max_bytes=50,
        clock=lambda: fake_clock["now"],
    )
    try:
        fake_clock["now"] = 1000
        c.put("k1", "p", "x" * 30)   # 30 bytes
        fake_clock["now"] = 1001
        c.put("k2", "p", "y" * 30)   # +30 = 60 > 50 → evict k1

        assert c.get("k1") is None
        assert c.get("k2") is not None
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Clear / stats
# ---------------------------------------------------------------------------

def test_clear_removes_everything(cache):
    cache.put("k1", "p", "r")
    cache.put("k2", "p", "r")
    assert cache.clear() == 2
    assert cache.stats()["entry_count"] == 0


def test_stats_shape(cache):
    cache.put("k1", "p", "r")
    s = cache.stats()
    for key in ("entry_count", "total_bytes", "hits", "misses",
                "expired", "evicted", "hit_rate",
                "ttl_seconds", "max_entries", "max_bytes"):
        assert key in s
    assert s["entry_count"] == 1
    assert s["hit_rate"] == 0.0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_put_after_close_is_silent():
    c = LLMCache(db_path=":memory:")
    c.close()
    # Should not raise — put swallows storage errors.
    c.put("k1", "p", "r")


def test_get_after_close_returns_none_or_raises():
    """
    get() doesn't wrap its sqlite call, so a closed connection raises
    ProgrammingError. Document this so a future refactor makes it
    consistent if desired.
    """
    import sqlite3
    c = LLMCache(db_path=":memory:")
    c.close()
    with pytest.raises(sqlite3.ProgrammingError):
        c.get("k1")