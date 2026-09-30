"""
Alert rules — spec §30A.

An AlertRule is a scope + condition + severity + cooldown + targets.
This file defines the data model and the condition predicates. The engine
that evaluates and deduplicates lives in `engine.py`.

Conditions supported (§30A):
    job_failed              — a job entered FAILED state
    repeated_job_failure    — N failures on the same task within a window
    record_count_drop       — record count fell below a threshold
    record_count_rise       — record count rose above a threshold
    quality_below           — quality score below a threshold
    confidence_collapse     — mean confidence below a threshold
    schema_change           — a record's field set changed
    source_conflict_spike   — conflict count exceeded a threshold
    budget_threshold        — % of budget consumed passed a threshold
    circuit_breaker_trip    — circuit breaker tripped on a job
    self_healing_event      — a healing rung fired
    unexpected_removals     — a diff had many REMOVED records
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ConditionType(str, Enum):
    JOB_FAILED = "job_failed"
    REPEATED_JOB_FAILURE = "repeated_job_failure"
    RECORD_COUNT_DROP = "record_count_drop"
    RECORD_COUNT_RISE = "record_count_rise"
    QUALITY_BELOW = "quality_below"
    CONFIDENCE_COLLAPSE = "confidence_collapse"
    SCHEMA_CHANGE = "schema_change"
    SOURCE_CONFLICT_SPIKE = "source_conflict_spike"
    BUDGET_THRESHOLD = "budget_threshold"
    CIRCUIT_BREAKER_TRIP = "circuit_breaker_trip"
    SELF_HEALING_EVENT = "self_healing_event"
    UNEXPECTED_REMOVALS = "unexpected_removals"


ALL_CONDITIONS = {c.value for c in ConditionType}


@dataclass
class AlertRule:
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scope: str = "client"              # client | project | task | domain | job
    scope_id: str = ""                 # e.g. task_id, domain, etc.
    condition_type: str = ""
    threshold: Optional[float] = None
    expression: str = ""               # free-form, reserved
    severity: str = Severity.WARNING.value
    cooldown_seconds: int = 300
    notification_targets: list[str] = field(default_factory=list)
    enabled: bool = True
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AlertRule":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def validates(self) -> tuple[bool, str]:
        """Return (ok, reason). Used by rule-creation validators."""
        if self.condition_type not in ALL_CONDITIONS:
            return False, f"unknown condition_type: {self.condition_type!r}"
        if self.severity not in {s.value for s in Severity}:
            return False, f"unknown severity: {self.severity!r}"
        if self.scope not in ("client", "project", "task", "domain", "job"):
            return False, f"unknown scope: {self.scope!r}"
        # Conditions that require a threshold must have one set
        needs_threshold = {
            ConditionType.RECORD_COUNT_DROP.value,
            ConditionType.RECORD_COUNT_RISE.value,
            ConditionType.QUALITY_BELOW.value,
            ConditionType.CONFIDENCE_COLLAPSE.value,
            ConditionType.SOURCE_CONFLICT_SPIKE.value,
            ConditionType.BUDGET_THRESHOLD.value,
            ConditionType.REPEATED_JOB_FAILURE.value,
            ConditionType.UNEXPECTED_REMOVALS.value,
        }
        if self.condition_type in needs_threshold and self.threshold is None:
            return False, f"condition {self.condition_type!r} requires a threshold"
        return True, "ok"


# ---------------------------------------------------------------------------
# Event envelope
# ---------------------------------------------------------------------------

@dataclass
class AlertEvent:
    """
    A stream-of-events item the engine evaluates against all rules.
    """
    kind: str                          # matches ConditionType values
    client_id: str = ""
    task_id: str = ""
    job_id: str = ""
    domain: str = ""
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fired alert
# ---------------------------------------------------------------------------

@dataclass
class FiredAlert:
    rule_id: str
    condition_type: str
    severity: str
    scope: str
    scope_id: str
    message: str
    dedup_key: str = ""
    event: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Condition predicates
# ---------------------------------------------------------------------------

def evaluate_condition(rule: AlertRule, event: AlertEvent, state: dict) -> Optional[str]:
    """
    Return a fired-alert message if the rule fires on this event, else None.

    `state` is the engine's per-(rule,scope) memory for windowed / count
    conditions (see engine.py). For stateless conditions, state is unused.
    """
    ct = rule.condition_type
    payload = event.payload or {}

    if ct == ConditionType.JOB_FAILED.value:
        if event.kind != "job_failed":
            return None
        return f"job {event.job_id or '(unknown)'} failed"

    if ct == ConditionType.QUALITY_BELOW.value:
        if event.kind != "job_completed":
            return None
        score = payload.get("quality_score")
        if score is None or rule.threshold is None:
            return None
        if score < rule.threshold:
            return f"quality score {score:.2f} below threshold {rule.threshold:.2f}"
        return None

    if ct == ConditionType.CONFIDENCE_COLLAPSE.value:
        if event.kind != "job_completed":
            return None
        conf = payload.get("confidence_mean")
        if conf is None or rule.threshold is None:
            return None
        if conf < rule.threshold:
            return f"mean confidence {conf:.2f} below threshold {rule.threshold:.2f}"
        return None

    if ct == ConditionType.RECORD_COUNT_DROP.value:
        if event.kind != "job_completed":
            return None
        count = payload.get("records")
        if count is None or rule.threshold is None:
            return None
        if count < rule.threshold:
            return f"record count {count} below threshold {int(rule.threshold)}"
        return None

    if ct == ConditionType.RECORD_COUNT_RISE.value:
        if event.kind != "job_completed":
            return None
        count = payload.get("records")
        if count is None or rule.threshold is None:
            return None
        if count > rule.threshold:
            return f"record count {count} above threshold {int(rule.threshold)}"
        return None

    if ct == ConditionType.CIRCUIT_BREAKER_TRIP.value:
        if event.kind != "circuit_breaker_trip":
            return None
        reason = payload.get("reason", "unspecified")
        return f"circuit breaker tripped: {reason}"

    if ct == ConditionType.SELF_HEALING_EVENT.value:
        if event.kind != "self_healing_event":
            return None
        rung = payload.get("rung", "?")
        return f"self-healing fired at rung {rung} on {event.domain or 'page'}"

    if ct == ConditionType.SCHEMA_CHANGE.value:
        if event.kind != "schema_changed":
            return None
        added = payload.get("added") or []
        removed = payload.get("removed") or []
        return f"schema changed (added={added}, removed={removed})"

    if ct == ConditionType.SOURCE_CONFLICT_SPIKE.value:
        if event.kind != "job_completed":
            return None
        conflicts = payload.get("conflicts", 0)
        if rule.threshold is None:
            return None
        if conflicts >= rule.threshold:
            return f"source conflicts {conflicts} met/exceeded {int(rule.threshold)}"
        return None

    if ct == ConditionType.BUDGET_THRESHOLD.value:
        if event.kind != "budget_event":
            return None
        pct = payload.get("pct")
        if pct is None or rule.threshold is None:
            return None
        if pct >= rule.threshold:
            return f"budget at {pct:.0%} (threshold {rule.threshold:.0%})"
        return None

    if ct == ConditionType.UNEXPECTED_REMOVALS.value:
        if event.kind != "change_detected":
            return None
        removed = payload.get("removed", 0)
        if rule.threshold is None:
            return None
        if removed >= rule.threshold:
            return f"{removed} unexpected removals (threshold {int(rule.threshold)})"
        return None

    if ct == ConditionType.REPEATED_JOB_FAILURE.value:
        if event.kind != "job_failed":
            return None
        # state["recent_failures"] is managed by the engine
        recent = state.get("recent_failures", [])
        threshold = int(rule.threshold or 3)
        if len(recent) >= threshold:
            return f"{len(recent)} failures in window (threshold {threshold})"
        return None

    return None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Validation
    ok, reason = AlertRule(
        condition_type=ConditionType.JOB_FAILED.value,
        severity=Severity.CRITICAL.value,
    ).validates()
    assert ok, reason

    bad = AlertRule(condition_type="not_a_condition")
    ok, reason = bad.validates()
    assert not ok and "condition_type" in reason

    needs_threshold = AlertRule(condition_type=ConditionType.QUALITY_BELOW.value)
    ok, reason = needs_threshold.validates()
    assert not ok and "threshold" in reason

    # Rule firing: JOB_FAILED
    r = AlertRule(condition_type=ConditionType.JOB_FAILED.value,
                  severity=Severity.CRITICAL.value)
    evt = AlertEvent(kind="job_failed", job_id="j-1")
    msg = evaluate_condition(r, evt, {})
    assert msg and "j-1" in msg

    # Non-firing
    assert evaluate_condition(r, AlertEvent(kind="job_completed"), {}) is None

    # QUALITY_BELOW
    r = AlertRule(condition_type=ConditionType.QUALITY_BELOW.value, threshold=0.8)
    evt = AlertEvent(kind="job_completed", payload={"quality_score": 0.5})
    assert evaluate_condition(r, evt, {}) is not None
    evt2 = AlertEvent(kind="job_completed", payload={"quality_score": 0.95})
    assert evaluate_condition(r, evt2, {}) is None

    # RECORD_COUNT_DROP
    r = AlertRule(condition_type=ConditionType.RECORD_COUNT_DROP.value, threshold=100)
    assert evaluate_condition(
        r, AlertEvent(kind="job_completed", payload={"records": 5}), {}) is not None
    assert evaluate_condition(
        r, AlertEvent(kind="job_completed", payload={"records": 500}), {}) is None

    # CIRCUIT_BREAKER_TRIP
    r = AlertRule(condition_type=ConditionType.CIRCUIT_BREAKER_TRIP.value)
    assert evaluate_condition(
        r, AlertEvent(kind="circuit_breaker_trip", payload={"reason": "max_usd"}), {},
    ) is not None

    # Self-healing
    r = AlertRule(condition_type=ConditionType.SELF_HEALING_EVENT.value)
    assert evaluate_condition(
        r, AlertEvent(kind="self_healing_event", payload={"rung": "rung2_text_llm"}), {},
    ) is not None

    # Schema change
    r = AlertRule(condition_type=ConditionType.SCHEMA_CHANGE.value)
    assert evaluate_condition(
        r, AlertEvent(kind="schema_changed",
                      payload={"added": ["new_field"], "removed": []}), {},
    ) is not None

    # Unexpected removals
    r = AlertRule(condition_type=ConditionType.UNEXPECTED_REMOVALS.value, threshold=10)
    assert evaluate_condition(
        r, AlertEvent(kind="change_detected", payload={"removed": 15}), {},
    ) is not None
    assert evaluate_condition(
        r, AlertEvent(kind="change_detected", payload={"removed": 5}), {},
    ) is None

    # Repeated job failures (needs engine-managed state)
    r = AlertRule(condition_type=ConditionType.REPEATED_JOB_FAILURE.value, threshold=3)
    state = {"recent_failures": [1, 2, 3]}
    assert evaluate_condition(r, AlertEvent(kind="job_failed"), state) is not None

    print("Alert rules OK.")