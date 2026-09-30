"""Unit tests for self-healing types (spec §17)."""
import pytest

from src.healing.types import (
    FailureKind, LadderRung, FailureContext, RungAttempt, HealingResult,
    EXTRACTION_FAILURES, TERMINAL_FAILURES, RUNG_ORDER, RUNG_COST,
    domain_of,
)


# --- enum membership --------------------------------------------------

def test_extraction_failures_subset():
    assert FailureKind.SELECTOR in EXTRACTION_FAILURES
    assert FailureKind.EMPTY in EXTRACTION_FAILURES
    assert FailureKind.SCHEMA in EXTRACTION_FAILURES


def test_terminal_failures_subset():
    assert FailureKind.ACCESS_BLOCK in TERMINAL_FAILURES
    assert FailureKind.AUTHORIZATION in TERMINAL_FAILURES
    assert FailureKind.COMPLIANCE in TERMINAL_FAILURES


def test_extraction_and_terminal_do_not_overlap():
    assert not (EXTRACTION_FAILURES & TERMINAL_FAILURES)


def test_rung_order_is_cheapest_first():
    assert RUNG_ORDER[0] == LadderRung.DETERMINISTIC_FALLBACK
    assert RUNG_ORDER[-1] == LadderRung.GRACEFUL_FAILURE


def test_rung_costs_increasing():
    costs = [RUNG_COST[r] for r in RUNG_ORDER[:3]]   # rung 4 is a give-up
    assert costs[0] <= costs[1] <= costs[2]


# --- FailureContext ---------------------------------------------------

def test_failure_context_domain_extraction():
    ctx = FailureContext(
        url="https://shop.example/cat/x",
        domain=domain_of("https://shop.example/cat/x"),
    )
    assert ctx.domain == "shop.example"


def test_failure_context_to_dict_masks_html():
    ctx = FailureContext(url="https://x", html="x" * 5000)
    d = ctx.to_dict()
    assert "chars" in d["html"]


def test_failure_context_serializes_enum():
    ctx = FailureContext(url="https://x", failure_kind=FailureKind.SELECTOR)
    d = ctx.to_dict()
    assert d["failure_kind"] == "selector"


# --- RungAttempt ------------------------------------------------------

def test_rung_attempt_to_dict_serializes_enum():
    a = RungAttempt(
        rung=LadderRung.TEXT_LLM, succeeded=True, records_recovered=3,
    )
    d = a.to_dict()
    assert d["rung"] == "rung2_text_llm"
    assert d["records_recovered"] == 3


# --- HealingResult ----------------------------------------------------

def test_healing_result_records_count():
    r = HealingResult(
        url="https://x", domain="x", recovered=True,
        rung_used=LadderRung.DETERMINISTIC_FALLBACK,
        records=[{"a": 1}, {"b": 2}],
    )
    assert len(r.records) == 2


def test_healing_result_cost_estimate_zero_for_rung1():
    r = HealingResult(
        url="https://x", domain="x", recovered=True,
        rung_used=LadderRung.DETERMINISTIC_FALLBACK,
        attempts=[RungAttempt(LadderRung.DETERMINISTIC_FALLBACK,
                              succeeded=True, records_recovered=1)],
    )
    assert r.total_cost_estimate == 0.0


def test_healing_result_cost_estimate_with_vision():
    r = HealingResult(
        url="https://x", domain="x", recovered=True,
        rung_used=LadderRung.VISION,
        attempts=[
            RungAttempt(LadderRung.DETERMINISTIC_FALLBACK, succeeded=False),
            RungAttempt(LadderRung.TEXT_LLM, succeeded=False),
            RungAttempt(LadderRung.VISION, succeeded=True, records_recovered=2),
        ],
    )
    # Vision is the most expensive rung
    assert r.total_cost_estimate > 0


def test_healing_result_to_dict_shape():
    r = HealingResult(
        url="https://x", domain="x", recovered=False,
        gave_up_reason="all rungs failed",
    )
    d = r.to_dict()
    assert d["recovered"] is False
    assert d["rung_used"] is None
    assert d["gave_up_reason"] == "all rungs failed"