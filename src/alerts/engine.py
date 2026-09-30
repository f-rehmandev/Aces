"""
Alert engine — spec §30A.

Evaluates a stream of `AlertEvent` objects against a set of `AlertRule`s.
Deduplicates and rate-limits alerts so one outage does not generate
thousands of notifications (§30A, last paragraph).

The engine has no I/O. It returns `FiredAlert` objects which callers
deliver via whatever notification channel they want (webhooks, email,
in-app — §47).
"""

from __future__ import annotations
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.alerts.rules import (
    AlertRule, AlertEvent, FiredAlert, ConditionType,
    evaluate_condition,
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AlertEngine:
    """
    Stateless with respect to rules; stateful with respect to:
      - cooldown windows (per rule + scope_id)
      - rolling failure counts for repeated-failure conditions
      - dedup keys for suppression
    """

    def __init__(
        self,
        rules: Optional[list[AlertRule]] = None,
        clock: Callable[[], float] = time.monotonic,
        failure_window_seconds: float = 600.0,
    ):
        self._rules: list[AlertRule] = list(rules or [])
        self._clock = clock
        self._failure_window_seconds = failure_window_seconds

        # state: rule_id -> scope_id -> {"last_fired_at": float}
        self._last_fired: dict[str, dict[str, float]] = defaultdict(dict)
        # recent_failures: task_id -> list[float]
        self._recent_failures: dict[str, list[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------
    def add_rule(self, rule: AlertRule) -> None:
        ok, reason = rule.validates()
        if not ok:
            raise ValueError(f"invalid rule: {reason}")
        self._rules.append(rule)

    def rules(self) -> list[AlertRule]:
        return list(self._rules)

    def rules_for_client(self, client_id: str) -> list[AlertRule]:
        return [r for r in self._rules if r.scope != "client" or r.scope_id in ("", client_id)]

    # ------------------------------------------------------------------
    # Evaluate one event
    # ------------------------------------------------------------------
    def evaluate(self, event: AlertEvent) -> list[FiredAlert]:
        """
        Run every applicable rule against this event, returning fired
        alerts (post-cooldown, post-dedup).
        """
        # Track failures for repeated-failure conditions
        if event.kind == "job_failed" and event.task_id:
            now = self._clock()
            cutoff = now - self._failure_window_seconds
            self._recent_failures[event.task_id] = [
                t for t in self._recent_failures[event.task_id] if t >= cutoff
            ]
            self._recent_failures[event.task_id].append(now)

        fired: list[FiredAlert] = []
        for rule in self._rules:
            if not rule.enabled:
                continue
            if not self._rule_applies(rule, event):
                continue

            state = {
                "recent_failures": self._recent_failures.get(event.task_id, []),
            }
            message = evaluate_condition(rule, event, state)
            if message is None:
                continue

            if not self._cooldown_allows(rule, event):
                continue

            dedup_key = self._dedup_key(rule, event)
            alert = FiredAlert(
                rule_id=rule.rule_id,
                condition_type=rule.condition_type,
                severity=rule.severity,
                scope=rule.scope,
                scope_id=rule.scope_id,
                message=message,
                dedup_key=dedup_key,
                event=event.to_dict(),
            )
            fired.append(alert)
            self._last_fired[rule.rule_id][self._scope_key(event)] = self._clock()

        return fired

    def evaluate_many(self, events: list[AlertEvent]) -> list[FiredAlert]:
        out: list[FiredAlert] = []
        for e in events:
            out.extend(self.evaluate(e))
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def last_fired_at(self, rule_id: str, scope_key: str) -> Optional[float]:
        return self._last_fired.get(rule_id, {}).get(scope_key)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _scope_key(event: AlertEvent) -> str:
        return event.task_id or event.job_id or event.domain or "global"

    @staticmethod
    def _rule_applies(rule: AlertRule, event: AlertEvent) -> bool:
        if rule.scope == "client" and rule.scope_id:
            return rule.scope_id == event.client_id
        if rule.scope == "task":
            return rule.scope_id == event.task_id
        if rule.scope == "job":
            return rule.scope_id == event.job_id
        if rule.scope == "domain":
            return rule.scope_id == event.domain
        return True   # client scope with empty scope_id applies to everything

    def _cooldown_allows(self, rule: AlertRule, event: AlertEvent) -> bool:
        if rule.cooldown_seconds <= 0:
            return True
        key = self._scope_key(event)
        last = self._last_fired.get(rule.rule_id, {}).get(key)
        if last is None:
            return True
        return (self._clock() - last) >= rule.cooldown_seconds

    @staticmethod
    def _dedup_key(rule: AlertRule, event: AlertEvent) -> str:
        """
        Stable key so a consumer (e.g. a webhook dispatcher) can suppress
        repeats within its own window. Not used for engine-side suppression
        directly — that's the cooldown's job.
        """
        raw = f"{rule.rule_id}|{event.kind}|{event.task_id}|{event.job_id}|{event.domain}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_default_engine(clock: Optional[Callable[[], float]] = None) -> AlertEngine:
    """
    An engine pre-loaded with the most common §30A rules at default thresholds.
    """
    rules = [
        AlertRule(condition_type=ConditionType.JOB_FAILED.value,
                  severity="critical", cooldown_seconds=60),
        AlertRule(condition_type=ConditionType.CIRCUIT_BREAKER_TRIP.value,
                  severity="critical", cooldown_seconds=60),
        AlertRule(condition_type=ConditionType.SCHEMA_CHANGE.value,
                  severity="warning", cooldown_seconds=300),
        AlertRule(condition_type=ConditionType.SELF_HEALING_EVENT.value,
                  severity="info", cooldown_seconds=300),
    ]
    return AlertEngine(rules=rules, clock=clock or time.monotonic)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    clock_val = [0.0]
    def clock(): return clock_val[0]

    engine = AlertEngine(clock=clock)

    engine.add_rule(AlertRule(
        condition_type=ConditionType.JOB_FAILED.value,
        severity="critical", cooldown_seconds=60,
    ))
    engine.add_rule(AlertRule(
        condition_type=ConditionType.QUALITY_BELOW.value,
        threshold=0.5, severity="warning", cooldown_seconds=0,
    ))

    # Job failed -> alert
    fired = engine.evaluate(AlertEvent(
        kind="job_failed", client_id="acme", task_id="t-1", job_id="j-1",
    ))
    assert len(fired) == 1
    assert fired[0].condition_type == "job_failed"
    assert fired[0].severity == "critical"

    # Same event again -> suppressed by cooldown
    fired = engine.evaluate(AlertEvent(
        kind="job_failed", client_id="acme", task_id="t-1", job_id="j-1",
    ))
    assert len(fired) == 0

    # 61s later -> allowed again
    clock_val[0] = 61.0
    fired = engine.evaluate(AlertEvent(
        kind="job_failed", client_id="acme", task_id="t-1", job_id="j-1",
    ))
    assert len(fired) == 1

    # Quality below threshold
    fired = engine.evaluate(AlertEvent(
        kind="job_completed", client_id="acme", task_id="t-2",
        payload={"quality_score": 0.3},
    ))
    assert any(f.condition_type == "quality_below" for f in fired)

    # Quality above -> no alert
    fired = engine.evaluate(AlertEvent(
        kind="job_completed", client_id="acme", task_id="t-3",
        payload={"quality_score": 0.9},
    ))
    assert fired == []

    # Repeated failure window
    engine2 = AlertEngine(clock=clock, failure_window_seconds=600)
    engine2.add_rule(AlertRule(
        condition_type=ConditionType.REPEATED_JOB_FAILURE.value,
        threshold=3, severity="critical",
    ))
    # First two failures produce no alert
    assert engine2.evaluate(AlertEvent(kind="job_failed", task_id="x")).__len__() == 0
    assert engine2.evaluate(AlertEvent(kind="job_failed", task_id="x")).__len__() == 0
    # Third produces one
    fired = engine2.evaluate(AlertEvent(kind="job_failed", task_id="x"))
    assert len(fired) == 1
    assert "3 failures" in fired[0].message

    # Rule validation
    try:
        engine2.add_rule(AlertRule(condition_type="not_a_thing"))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # Rule scoping: task-scoped rule only fires for that task
    engine3 = AlertEngine(clock=clock)
    engine3.add_rule(AlertRule(
        condition_type=ConditionType.JOB_FAILED.value,
        scope="task", scope_id="t-A",
    ))
    assert engine3.evaluate(AlertEvent(kind="job_failed", task_id="t-A"))
    assert not engine3.evaluate(AlertEvent(kind="job_failed", task_id="t-B"))

    # Default engine
    de = build_default_engine()
    fired = de.evaluate(AlertEvent(kind="job_failed", task_id="t"))
    assert any(f.condition_type == "job_failed" for f in fired)

    print("Alert engine OK.")