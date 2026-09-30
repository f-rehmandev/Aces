"""Unit tests for the evolution engine (spec §19)."""
import time

import pytest

from src.evolution.types import (
    CandidateStatus, CandidateStrategy, EvolutionOutcome, EvolutionResult,
    Experience, PromotionGate, RollbackTrigger, SandboxResult,
)
from src.evolution.engine import EvolutionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_sandbox(candidate):
    return SandboxResult(
        records_extracted=50, field_completeness=0.95,
        schema_validation_pass_rate=1.0, duplicate_rate=0.0,
        cost_usd=0.02, latency_seconds=10,
    )


def _baseline_experiences(domain="shop.example", n=3):
    return [
        Experience(
            run_id=f"r{i}", task_id="t", domain=domain,
            records_extracted=50, field_completeness=0.90,
            schema_validation_pass_rate=1.0, duplicate_rate=0.0,
            cost_usd=0.02, latency_seconds=10,
        )
        for i in range(n)
    ]


def _engine(sandbox=_ok_sandbox, promoter=None, rollbacker=None, clock=None):
    promoted = []
    if promoter is None:
        def promoter(c):
            sid = f"sid-{len(promoted)}"
            promoted.append(sid)
            return sid
    return EvolutionEngine(
        sandbox_runner=sandbox, promoter=promoter,
        rollbacker=rollbacker, clock=clock or time.monotonic,
    )


# ---------------------------------------------------------------------------
# Experience recording
# ---------------------------------------------------------------------------

def test_experience_recording_and_retrieval():
    e = _engine()
    e.record_experience(Experience(run_id="1", task_id="t", domain="a"))
    e.record_experience(Experience(run_id="2", task_id="t", domain="b"))
    assert len(e.experiences()) == 2
    assert len(e.experiences_for_domain("a")) == 1


def test_failure_clusters():
    e = _engine()
    for i in range(3):
        e.record_experience(Experience(run_id=str(i), task_id="t",
                                        domain="x", failure_class="selector"))
    e.record_experience(Experience(run_id="ok", task_id="t", domain="x"))
    clusters = e.failure_clusters("x")
    assert clusters == {"selector": 3}


# ---------------------------------------------------------------------------
# Candidate proposals
# ---------------------------------------------------------------------------

def test_propose_candidates_for_observed_failures():
    e = _engine()
    e.record_experience(Experience(run_id="1", task_id="t",
                                    domain="x", failure_class="selector"))
    proposals = e.propose_candidates("x", {
        "selector": lambda: {"fallback": [".a"]},
        "timeout": lambda: {"timeout_ms": 30000},
    })
    assert len(proposals) == 1
    assert proposals[0].configuration == {"fallback": [".a"]}


def test_propose_candidates_no_failures_returns_empty():
    e = _engine()
    e.record_experience(Experience(run_id="1", task_id="t", domain="x"))
    proposals = e.propose_candidates("x", {"selector": lambda: {}})
    assert proposals == []


# ---------------------------------------------------------------------------
# Promotion — happy path
# ---------------------------------------------------------------------------

def test_promotion_happy_path():
    e = _engine()
    for exp in _baseline_experiences():
        e.record_experience(exp)
    c = CandidateStrategy(domain="shop.example", configuration={"a": 1})
    result = e.evaluate(c, baseline_runs=3)
    assert result.outcome == EvolutionOutcome.PROMOTED
    assert c.status == CandidateStatus.PROMOTED
    assert e.promoted_strategy_for("shop.example") is not None


def test_promotion_records_baseline_and_candidate_means():
    e = _engine()
    for exp in _baseline_experiences():
        e.record_experience(exp)
    c = CandidateStrategy(domain="shop.example", configuration={})
    result = e.evaluate(c, baseline_runs=3)
    assert result.baseline_mean == pytest.approx(0.90)
    assert result.candidate_mean == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Promotion gate rejections (§19.3)
# ---------------------------------------------------------------------------

def test_gate_rejects_low_records():
    def small(candidate):
        return SandboxResult(records_extracted=5, schema_validation_pass_rate=1.0)
    e = _engine(sandbox=small)
    result = e.evaluate(CandidateStrategy(domain="x", configuration={}),
                        baseline_runs=3)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "records" in result.reason


def test_gate_rejects_low_schema_pass():
    def schema_fail(candidate):
        return SandboxResult(records_extracted=50,
                             schema_validation_pass_rate=0.8)
    e = _engine(sandbox=schema_fail)
    result = e.evaluate(CandidateStrategy(domain="x", configuration={}),
                        baseline_runs=3)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "schema" in result.reason


def test_gate_rejects_insufficient_runs():
    e = _engine()
    result = e.evaluate(CandidateStrategy(domain="x", configuration={}),
                        baseline_runs=2)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "run" in result.reason.lower()


def test_gate_rejects_compliance_flags():
    def flagged(candidate):
        return SandboxResult(
            records_extracted=50, schema_validation_pass_rate=1.0,
            compliance_flags=["access_control_detected"],
        )
    e = _engine(sandbox=flagged)
    result = e.evaluate(CandidateStrategy(domain="x", configuration={}),
                        baseline_runs=3)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "compliance" in result.reason.lower()


def test_gate_rejects_field_completeness_drop():
    def worse(candidate):
        return SandboxResult(
            records_extracted=50, field_completeness=0.5,
            schema_validation_pass_rate=1.0,
        )
    e = _engine(sandbox=worse)
    for exp in _baseline_experiences():
        e.record_experience(exp)
    result = e.evaluate(CandidateStrategy(domain="shop.example", configuration={}),
                        baseline_runs=3)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "field completeness" in result.reason.lower()


def test_gate_rejects_duplicate_rise():
    def dup(candidate):
        return SandboxResult(
            records_extracted=50, field_completeness=0.95,
            schema_validation_pass_rate=1.0,
            duplicate_rate=0.5,
        )
    e = _engine(sandbox=dup)
    for exp in _baseline_experiences():
        e.record_experience(exp)
    result = e.evaluate(CandidateStrategy(domain="shop.example", configuration={}),
                        baseline_runs=3)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "duplicate" in result.reason.lower()


def test_gate_rejects_cost_blowup():
    def expensive(candidate):
        return SandboxResult(
            records_extracted=50, field_completeness=0.95,
            schema_validation_pass_rate=1.0,
            cost_usd=1.0,
        )
    e = _engine(sandbox=expensive)
    for exp in _baseline_experiences():
        e.record_experience(exp)
    result = e.evaluate(CandidateStrategy(domain="shop.example", configuration={}),
                        baseline_runs=3)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "cost" in result.reason.lower()


def test_gate_rejects_latency_blowup():
    def slow(candidate):
        return SandboxResult(
            records_extracted=50, field_completeness=0.95,
            schema_validation_pass_rate=1.0,
            latency_seconds=500,
        )
    e = _engine(sandbox=slow)
    for exp in _baseline_experiences():
        e.record_experience(exp)
    result = e.evaluate(CandidateStrategy(domain="shop.example", configuration={}),
                        baseline_runs=3)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "latency" in result.reason.lower()

# ---------------------------------------------------------------------------
# Sandbox errors
# ---------------------------------------------------------------------------

def test_sandbox_exception_rejects():
    def raises(candidate):
        raise RuntimeError("boom")
    e = _engine(sandbox=raises)
    result = e.evaluate(CandidateStrategy(domain="x", configuration={}),
                        baseline_runs=1)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "boom" in result.reason


def test_sandbox_error_field_rejects():
    def errored(candidate):
        return SandboxResult(error="network down")
    e = _engine(sandbox=errored)
    result = e.evaluate(CandidateStrategy(domain="x", configuration={}),
                        baseline_runs=1)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "network down" in result.reason


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def test_rollback_on_success_rate_drop():
    rolled_back: list[tuple[str, str]] = []
    e = _engine(rollbacker=lambda d, r: rolled_back.append((d, r)))
    reason = e.check_rollback(
        "shop.example",
        current_success_rate=0.5,
        seven_day_baseline_success_rate=0.95,
        current_quality_score=0.9,
    )
    assert reason is not None
    assert "success rate" in reason
    assert rolled_back[0][0] == "shop.example"


def test_rollback_on_quality_floor():
    rolled_back: list[tuple[str, str]] = []
    e = _engine(rollbacker=lambda d, r: rolled_back.append((d, r)))
    reason = e.check_rollback(
        "shop.example",
        current_success_rate=0.95,
        seven_day_baseline_success_rate=0.95,
        current_quality_score=0.3,
    )
    assert reason is not None
    assert "quality" in reason


def test_no_rollback_when_healthy():
    e = _engine(rollbacker=lambda d, r: None)
    reason = e.check_rollback(
        "shop.example",
        current_success_rate=0.95,
        seven_day_baseline_success_rate=0.95,
        current_quality_score=0.9,
    )
    assert reason is None


def test_rollback_cooldown_prevents_double_rollback():
    clock_val = [0.0]
    e = _engine(
        rollbacker=lambda d, r: None,
        clock=lambda: clock_val[0],
    )
    reason = e.check_rollback(
        "shop.example", current_success_rate=0.5,
        seven_day_baseline_success_rate=0.95, current_quality_score=0.9,
    )
    assert reason is not None
    # Immediately after, cooldown blocks another rollback
    reason2 = e.check_rollback(
        "shop.example", current_success_rate=0.5,
        seven_day_baseline_success_rate=0.95, current_quality_score=0.9,
    )
    assert reason2 is None


def test_rolled_back_candidate_status_updated():
    e = _engine(rollbacker=lambda d, r: None)
    c = CandidateStrategy(domain="x", configuration={})
    c.status = CandidateStatus.PROMOTED
    e._candidates.append(c)
    e.check_rollback("x", current_success_rate=0.1,
                     seven_day_baseline_success_rate=1.0,
                     current_quality_score=0.9)
    assert c.status == CandidateStatus.ROLLED_BACK


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def test_candidates_returns_copy():
    e = _engine()
    e.propose_candidates("x", {"selector": lambda: {}})
    candidates = e.candidates()
    candidates.append("garbage")
    assert len(e.candidates()) == 0


def test_promoted_strategy_for_unknown_domain_returns_none():
    e = _engine()
    assert e.promoted_strategy_for("nope") is None