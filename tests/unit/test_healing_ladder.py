"""Unit tests for the healing ladder orchestrator (spec §17)."""
import pytest

from src.healing.types import FailureContext, FailureKind, LadderRung
from src.healing.circuit import HealingCircuitBreaker
from src.healing.ladder import HealingLadder, make_ladder


def ctx(**kw):
    base = dict(
        url="https://shop.example/x", domain="shop.example",
        failure_kind=FailureKind.SELECTOR, field_name="price",
    )
    base.update(kw)
    return FailureContext(**base)


# --- rung 1 -----------------------------------------------------------

def test_rung1_success_stops_ladder():
    called = {"llm": 0, "vision": 0}
    lad = HealingLadder(
        deterministic_fn=lambda c: [{"price": "$5"}],
        text_llm_fn=lambda c: called.update(llm=called["llm"] + 1) or [],
        vision_fn=lambda c: called.update(vision=called["vision"] + 1) or [],
    )
    res = lad.attempt(ctx())
    assert res.recovered
    assert res.rung_used == LadderRung.DETERMINISTIC_FALLBACK
    assert called["llm"] == 0
    assert called["vision"] == 0


def test_rung1_raises_treated_as_failed():
    def raises(c): raise RuntimeError("boom")
    lad = HealingLadder(
        deterministic_fn=raises,
        text_llm_fn=lambda c: [{"x": 1}],
    )
    res = lad.attempt(ctx())
    assert res.recovered
    assert res.rung_used == LadderRung.TEXT_LLM
    assert any("RuntimeError" in a.reason for a in res.attempts)


# --- rung 2 -----------------------------------------------------------

def test_rung2_used_when_rung1_fails():
    lad = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=lambda c: [{"price": "$5"}],
        vision_fn=lambda c: [{"price": "$5"}],
    )
    res = lad.attempt(ctx())
    assert res.recovered
    assert res.rung_used == LadderRung.TEXT_LLM


# --- rung 3 -----------------------------------------------------------

def test_rung3_used_when_rung1_2_fail():
    lad = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=lambda c: [],
        vision_fn=lambda c: [{"price": "$5"}],
    )
    res = lad.attempt(ctx())
    assert res.recovered
    assert res.rung_used == LadderRung.VISION


# --- rung 4 (graceful) -----------------------------------------------

def test_all_rungs_fail_graceful_failure():
    lad = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=lambda c: [],
        vision_fn=lambda c: [],
    )
    res = lad.attempt(ctx(partial_records=[{"title": "A"}, {"title": "B"}]))
    assert not res.recovered
    assert res.rung_used is None
    assert res.gave_up_reason == "all rungs exhausted"
    assert "Partial records: 2" in res.warning


def test_disabled_rungs_are_marked_not_attempted():
    lad = HealingLadder(deterministic_fn=None, text_llm_fn=None, vision_fn=None)
    res = lad.attempt(ctx())
    non_graceful = [a for a in res.attempts if a.rung != LadderRung.GRACEFUL_FAILURE]
    assert all(not a.attempted for a in non_graceful)
    assert all("disabled" in a.reason for a in non_graceful)


# --- terminal failures ------------------------------------------------

@pytest.mark.parametrize("kind", [
    FailureKind.ACCESS_BLOCK,
    FailureKind.AUTHORIZATION,
    FailureKind.COMPLIANCE,
    FailureKind.SECURITY,
    FailureKind.BUDGET,
])
def test_terminal_failure_skips_all_rungs(kind):
    called = {"count": 0}
    def bump(c): called["count"] += 1; return [{"x": 1}]
    lad = HealingLadder(deterministic_fn=bump, text_llm_fn=bump, vision_fn=bump)
    res = lad.attempt(ctx(failure_kind=kind))
    assert not res.recovered
    assert "terminal" in res.gave_up_reason
    assert called["count"] == 0


def test_non_extraction_failure_skips():
    lad = HealingLadder(deterministic_fn=lambda c: [{"x": 1}])
    res = lad.attempt(ctx(failure_kind=FailureKind.NETWORK))
    assert not res.recovered
    assert "not an extraction failure" in res.gave_up_reason


# --- circuit breaker --------------------------------------------------

def test_circuit_breaker_skips_llm_rungs_after_trip():
    cb = HealingCircuitBreaker(max_per_domain=3)
    lad = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=lambda c: [],
        vision_fn=lambda c: [],
        circuit_breaker=cb,
    )
    for _ in range(3):
        lad.attempt(ctx())
    # 4th attempt hits the tripped circuit
    res = lad.attempt(ctx())
    skipped = [a for a in res.attempts if not a.attempted]
    assert any(a.rung == LadderRung.TEXT_LLM for a in skipped)
    assert any(a.rung == LadderRung.VISION for a in skipped)
    assert "circuit tripped" in res.warning


def test_circuit_breaker_does_not_block_rung1():
    cb = HealingCircuitBreaker(max_per_domain=1)
    # Trip it
    cb.record_attempt("shop.example")
    lad = HealingLadder(
        deterministic_fn=lambda c: [{"x": 1}],
        circuit_breaker=cb,
    )
    res = lad.attempt(ctx())
    assert res.recovered
    assert res.rung_used == LadderRung.DETERMINISTIC_FALLBACK


# --- cost tracking ----------------------------------------------------

def test_rung1_cost_is_zero():
    lad = HealingLadder(deterministic_fn=lambda c: [{"x": 1}])
    res = lad.attempt(ctx())
    assert res.total_cost_estimate == 0.0


def test_vision_cost_is_highest():
    lad = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=lambda c: [],
        vision_fn=lambda c: [{"x": 1}],
    )
    res = lad.attempt(ctx())
    assert res.total_cost_estimate > 0


# --- factory ----------------------------------------------------------

def test_make_ladder_default_only_rung1():
    lad = make_ladder()
    res = lad.attempt(ctx())
    assert not res.recovered
    assert any(a.rung == LadderRung.DETERMINISTIC_FALLBACK for a in res.attempts)


# --- serialization ----------------------------------------------------

def test_result_to_dict():
    lad = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=lambda c: [{"a": 1}],
    )
    res = lad.attempt(ctx())
    d = res.to_dict()
    assert d["recovered"] is True
    assert d["rung_used"] == "rung2_text_llm"
    assert d["records_count"] == 1