"""Unit tests for alert rules + engine (spec §30A)."""
import pytest

from src.alerts.rules import (
    AlertRule, AlertEvent, ConditionType, Severity,
    evaluate_condition, ALL_CONDITIONS,
)
from src.alerts.engine import AlertEngine, build_default_engine


def _clock_pair(start: float = 0.0):
    val = [start]
    return val, (lambda: val[0])


# --- rule validation --------------------------------------------------

def test_valid_rule():
    ok, _ = AlertRule(condition_type=ConditionType.JOB_FAILED.value).validates()
    assert ok


def test_invalid_condition_rejected():
    ok, reason = AlertRule(condition_type="nope").validates()
    assert not ok and "condition_type" in reason


def test_invalid_severity_rejected():
    ok, reason = AlertRule(
        condition_type=ConditionType.JOB_FAILED.value,
        severity="panic",
    ).validates()
    assert not ok and "severity" in reason


def test_condition_requiring_threshold_rejected_without_one():
    ok, reason = AlertRule(
        condition_type=ConditionType.QUALITY_BELOW.value,
    ).validates()
    assert not ok and "threshold" in reason


def test_all_conditions_are_known():
    for c in ConditionType:
        assert c.value in ALL_CONDITIONS


# --- predicates ------------------------------------------------------

def test_job_failed_fires():
    r = AlertRule(condition_type=ConditionType.JOB_FAILED.value)
    assert evaluate_condition(r, AlertEvent(kind="job_failed", job_id="j"), {}) is not None


def test_job_failed_does_not_fire_on_completed():
    r = AlertRule(condition_type=ConditionType.JOB_FAILED.value)
    assert evaluate_condition(r, AlertEvent(kind="job_completed"), {}) is None


def test_quality_below_fires_and_not_fires():
    r = AlertRule(condition_type=ConditionType.QUALITY_BELOW.value, threshold=0.8)
    assert evaluate_condition(
        r, AlertEvent(kind="job_completed", payload={"quality_score": 0.5}), {}) is not None
    assert evaluate_condition(
        r, AlertEvent(kind="job_completed", payload={"quality_score": 0.9}), {}) is None


def test_record_count_drop_fires():
    r = AlertRule(condition_type=ConditionType.RECORD_COUNT_DROP.value, threshold=100)
    assert evaluate_condition(
        r, AlertEvent(kind="job_completed", payload={"records": 50}), {}) is not None


def test_record_count_rise_fires():
    r = AlertRule(condition_type=ConditionType.RECORD_COUNT_RISE.value, threshold=100)
    assert evaluate_condition(
        r, AlertEvent(kind="job_completed", payload={"records": 500}), {}) is not None


def test_budget_threshold_fires_at_pct():
    r = AlertRule(condition_type=ConditionType.BUDGET_THRESHOLD.value, threshold=0.8)
    assert evaluate_condition(
        r, AlertEvent(kind="budget_event", payload={"pct": 0.9}), {}) is not None
    assert evaluate_condition(
        r, AlertEvent(kind="budget_event", payload={"pct": 0.5}), {}) is None


def test_source_conflict_spike_fires():
    r = AlertRule(condition_type=ConditionType.SOURCE_CONFLICT_SPIKE.value, threshold=5)
    assert evaluate_condition(
        r, AlertEvent(kind="job_completed", payload={"conflicts": 8}), {}) is not None


def test_unexpected_removals_fires():
    r = AlertRule(condition_type=ConditionType.UNEXPECTED_REMOVALS.value, threshold=10)
    assert evaluate_condition(
        r, AlertEvent(kind="change_detected", payload={"removed": 15}), {}) is not None
    assert evaluate_condition(
        r, AlertEvent(kind="change_detected", payload={"removed": 3}), {}) is None


def test_self_healing_event_fires():
    r = AlertRule(condition_type=ConditionType.SELF_HEALING_EVENT.value)
    event = AlertEvent(
        kind="self_healing_event",
        payload={"rung": "rung2_text_llm"},
    )
    msg = evaluate_condition(r, event, {})
    assert msg is not None and "rung2_text_llm" in msg

def test_repeated_job_failure_uses_state():
    r = AlertRule(condition_type=ConditionType.REPEATED_JOB_FAILURE.value, threshold=3)
    assert evaluate_condition(
        r, AlertEvent(kind="job_failed"), {"recent_failures": [1, 2, 3]},
    ) is not None
    assert evaluate_condition(
        r, AlertEvent(kind="job_failed"), {"recent_failures": [1]},
    ) is None


# --- engine: cooldown -------------------------------------------------

def test_cooldown_suppresses_repeat():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock)
    e.add_rule(AlertRule(
        condition_type=ConditionType.JOB_FAILED.value, cooldown_seconds=60,
    ))
    assert e.evaluate(AlertEvent(kind="job_failed", task_id="t", job_id="j")) != []
    assert e.evaluate(AlertEvent(kind="job_failed", task_id="t", job_id="j")) == []


def test_cooldown_expires():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock)
    e.add_rule(AlertRule(
        condition_type=ConditionType.JOB_FAILED.value, cooldown_seconds=60,
    ))
    e.evaluate(AlertEvent(kind="job_failed", task_id="t"))
    val[0] = 61.0
    assert e.evaluate(AlertEvent(kind="job_failed", task_id="t")) != []


def test_cooldown_zero_never_suppresses():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock)
    e.add_rule(AlertRule(
        condition_type=ConditionType.JOB_FAILED.value, cooldown_seconds=0,
    ))
    for _ in range(5):
        assert e.evaluate(AlertEvent(kind="job_failed", task_id="t")) != []


# --- engine: rule scoping --------------------------------------------

def test_task_scoped_rule_only_fires_for_matching_task():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock)
    e.add_rule(AlertRule(
        condition_type=ConditionType.JOB_FAILED.value,
        scope="task", scope_id="t-A",
    ))
    assert e.evaluate(AlertEvent(kind="job_failed", task_id="t-A"))
    assert not e.evaluate(AlertEvent(kind="job_failed", task_id="t-B"))


def test_client_scoped_rule_scopes_by_client_id():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock)
    e.add_rule(AlertRule(
        condition_type=ConditionType.JOB_FAILED.value,
        scope="client", scope_id="acme",
    ))
    assert e.evaluate(AlertEvent(kind="job_failed", client_id="acme", task_id="t"))
    assert not e.evaluate(AlertEvent(kind="job_failed", client_id="other", task_id="t"))


def test_disabled_rule_never_fires():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock)
    e.add_rule(AlertRule(
        condition_type=ConditionType.JOB_FAILED.value, enabled=False,
    ))
    assert e.evaluate(AlertEvent(kind="job_failed", task_id="t")) == []


# --- engine: repeated-failure window --------------------------------

def test_repeated_failure_fires_on_third():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock, failure_window_seconds=600)
    e.add_rule(AlertRule(
        condition_type=ConditionType.REPEATED_JOB_FAILURE.value,
        threshold=3, cooldown_seconds=0,
    ))
    assert e.evaluate(AlertEvent(kind="job_failed", task_id="t")) == []
    assert e.evaluate(AlertEvent(kind="job_failed", task_id="t")) == []
    assert e.evaluate(AlertEvent(kind="job_failed", task_id="t")) != []


def test_repeated_failure_window_resets():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock, failure_window_seconds=600)
    e.add_rule(AlertRule(
        condition_type=ConditionType.REPEATED_JOB_FAILURE.value,
        threshold=3, cooldown_seconds=0,
    ))
    e.evaluate(AlertEvent(kind="job_failed", task_id="t"))
    e.evaluate(AlertEvent(kind="job_failed", task_id="t"))
    # >600s later
    val[0] = 700.0
    # The old failures are gone -> this becomes the first failure in window
    assert e.evaluate(AlertEvent(kind="job_failed", task_id="t")) == []


# --- evaluate_many ---------------------------------------------------

def test_evaluate_many_returns_all_fired():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock)
    e.add_rule(AlertRule(
        condition_type=ConditionType.JOB_FAILED.value, cooldown_seconds=0,
    ))
    events = [
        AlertEvent(kind="job_failed", task_id="t1"),
        AlertEvent(kind="job_failed", task_id="t2"),
        AlertEvent(kind="job_completed"),
    ]
    fired = e.evaluate_many(events)
    assert len(fired) == 2


# --- dedup key stability ---------------------------------------------

def test_dedup_key_is_stable_for_same_scope():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock)
    r = AlertRule(condition_type=ConditionType.JOB_FAILED.value, cooldown_seconds=0)
    e.add_rule(r)
    f1 = e.evaluate(AlertEvent(kind="job_failed", task_id="t", job_id="j"))[0]
    f2 = e.evaluate(AlertEvent(kind="job_failed", task_id="t", job_id="j"))[0]
    assert f1.dedup_key == f2.dedup_key


def test_dedup_key_differs_across_tasks():
    val, clock = _clock_pair()
    e = AlertEngine(clock=clock)
    e.add_rule(AlertRule(
        condition_type=ConditionType.JOB_FAILED.value, cooldown_seconds=0,
    ))
    f1 = e.evaluate(AlertEvent(kind="job_failed", task_id="t1"))[0]
    f2 = e.evaluate(AlertEvent(kind="job_failed", task_id="t2"))[0]
    assert f1.dedup_key != f2.dedup_key


# --- rule management -------------------------------------------------

def test_add_invalid_rule_raises():
    e = AlertEngine()
    with pytest.raises(ValueError):
        e.add_rule(AlertRule(condition_type="nope"))


def test_rules_returns_copy():
    e = AlertEngine()
    r = AlertRule(condition_type=ConditionType.JOB_FAILED.value)
    e.add_rule(r)
    rules = e.rules()
    rules.append("garbage")
    assert len(e.rules()) == 1


# --- default engine --------------------------------------------------

def test_default_engine_fires_on_job_failed():
    e = build_default_engine()
    assert e.evaluate(AlertEvent(kind="job_failed", task_id="t"))


def test_default_engine_fires_on_circuit_breaker():
    e = build_default_engine()
    fired = e.evaluate(AlertEvent(kind="circuit_breaker_trip",
                                   payload={"reason": "max_usd"}))
    assert any(f.condition_type == "circuit_breaker_trip" for f in fired)