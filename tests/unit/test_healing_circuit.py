"""Unit tests for the healing circuit breaker (spec §17.2.1)."""
import pytest

from src.healing.circuit import (
    HealingCircuitBreaker, CircuitSnapshot, DEFAULT_MAX_PER_DOMAIN,
)


def test_default_max():
    assert DEFAULT_MAX_PER_DOMAIN == 3


def test_invalid_max_raises():
    with pytest.raises(ValueError):
        HealingCircuitBreaker(max_per_domain=-1)


def test_first_attempts_allowed():
    cb = HealingCircuitBreaker(max_per_domain=3)
    allowed, reason = cb.can_attempt("shop.example")
    assert allowed
    assert reason == "under_limit"


def test_third_attempt_trips():
    cb = HealingCircuitBreaker(max_per_domain=3)
    cb.record_attempt("shop.example")
    cb.record_attempt("shop.example")
    cb.record_attempt("shop.example")
    allowed, reason = cb.can_attempt("shop.example")
    assert not allowed
    assert "healing loop detected" in reason


def test_independent_domains():
    cb = HealingCircuitBreaker(max_per_domain=3)
    cb.record_attempt("a.example")
    cb.record_attempt("a.example")
    cb.record_attempt("a.example")
    assert not cb.can_attempt("a.example")[0]
    assert cb.can_attempt("b.example")[0] is True


def test_case_insensitive():
    cb = HealingCircuitBreaker(max_per_domain=1)
    cb.record_attempt("SHOP.EXAMPLE")
    assert not cb.can_attempt("shop.example")[0]


def test_max_zero_immediately_trips():
    cb = HealingCircuitBreaker(max_per_domain=0)
    allowed, reason = cb.can_attempt("x.example")
    assert not allowed
    assert "healing loop" in reason


def test_snapshot_shape():
    cb = HealingCircuitBreaker(max_per_domain=3)
    cb.record_attempt("shop.example")
    snap = cb.snapshot("shop.example")
    assert isinstance(snap, CircuitSnapshot)
    assert snap.domain == "shop.example"
    assert snap.healing_attempts == 1
    assert not snap.tripped


def test_snapshot_to_dict():
    cb = HealingCircuitBreaker()
    cb.record_attempt("x.example")
    d = cb.snapshot("x.example").to_dict()
    assert d["healing_attempts"] == 1
    assert d["tripped"] is False


def test_attempts_for():
    cb = HealingCircuitBreaker()
    cb.record_attempt("x.example")
    cb.record_attempt("x.example")
    assert cb.attempts_for("x.example") == 2
    assert cb.attempts_for("y.example") == 0


def test_reset_clears_state():
    cb = HealingCircuitBreaker(max_per_domain=2)
    cb.record_attempt("x.example")
    cb.record_attempt("x.example")
    assert cb.is_tripped("x.example")
    cb.reset()
    assert not cb.is_tripped("x.example")
    assert cb.attempts_for("x.example") == 0


def test_record_returns_new_count():
    cb = HealingCircuitBreaker()
    assert cb.record_attempt("x.example") == 1
    assert cb.record_attempt("x.example") == 2