"""
Alert → Incident bridge — spec §30A + §40.6.

Closes the loop between the alert engine (which decides *whether* to
fire) and the incident store (which tracks *what is currently wrong*).

The tracker is not an engine and not a store. It is a thin translator:

    AlertEngine.evaluate(event)             → list[FiredAlert]
              ↓
    IncidentTracker.on_alerts_fired(...)    → touches incident store
              ↓
    IncidentStore.open_or_touch(incident)   → dedupes by dedup_key

And, in the other direction:

    a job succeeds
              ↓
    IncidentTracker.on_job_succeeded(job_id)
              ↓
    IncidentStore.update_status(..., RESOLVED)

Design decisions:
    - Incident dedup key is `rule_id::scope_id::client_id`, so one rule
      scoped to one thing = at most one open incident at a time. A rule
      can't produce a swarm of separate incidents when it keeps firing.
    - Alert severity maps directly to incident severity (same enum names).
    - Alert condition_type maps to a FailureClass via a small lookup
      table. Unknown conditions fall back to UNKNOWN — never guess.
    - Every fired alert touches the incident. Because the AlertEngine
      already applies its cooldown *before* returning a FiredAlert, the
      incident's occurrence_count stays naturally aligned with "how many
      times the alert actually fired".
    - The tracker is stateless. It holds a reference to the store and
      nothing else. Callers can share one instance across a whole run.
"""
from __future__ import annotations

import logging

from src.alerts.rules import FiredAlert
from src.observability.store import IncidentStore
from src.observability.types import (
    DetectionSource,
    FailureClass,
    Incident,
    IncidentSeverity,
    IncidentStatus,
)


logger = logging.getLogger("observability.tracker")


# ---------------------------------------------------------------------------
# Mapping tables — spec §30A condition types → §40.6 failure classes
# ---------------------------------------------------------------------------

_SEVERITY_MAP: dict[str, IncidentSeverity] = {
    "info":     IncidentSeverity.INFO,
    "warning":  IncidentSeverity.WARNING,
    "critical": IncidentSeverity.CRITICAL,
}

# Condition types are strings (defined in src.alerts.rules.ConditionType).
# Anything not in this table falls back to UNKNOWN. We do NOT try to
# guess a failure class from an unrecognized string — that would mask
# real bugs in the rule engine.
_CONDITION_TO_FAILURE_CLASS: dict[str, FailureClass] = {
    # Job-level failures — too generic to classify further
    "job_failed":               FailureClass.UNKNOWN,
    "repeated_job_failure":     FailureClass.UNKNOWN,

    # Data quality signals
    "record_count_drop":        FailureClass.DATA_QUALITY,
    "record_count_rise":        FailureClass.DATA_QUALITY,
    "quality_below":            FailureClass.DATA_QUALITY,
    "confidence_collapse":      FailureClass.DATA_QUALITY,
    "source_conflict_spike":    FailureClass.DATA_QUALITY,
    "unexpected_removals":      FailureClass.DATA_QUALITY,

    # Schema drift
    "schema_change":            FailureClass.SCHEMA,

    # Budget / cost circuit
    "budget_threshold":         FailureClass.BUDGET,
    "circuit_breaker_trip":     FailureClass.BUDGET,

    # Healing is infrastructure, not data
    "self_healing_event":       FailureClass.INFRASTRUCTURE,
}

# Only a couple of conditions have a non-default detection source.
_CONDITION_TO_DETECTION_SOURCE: dict[str, DetectionSource] = {
    "circuit_breaker_trip": DetectionSource.CIRCUIT_BREAKER,
}


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class IncidentTracker:
    """
    Translates FiredAlerts into incident-store operations.

    `store` must satisfy `src.observability.store.IncidentStore`. Both
    the in-memory and the Supabase implementations do.
    """

    def __init__(self, store: IncidentStore):
        self.store = store

    # ------------------------------------------------------------------
    # Fired alerts → open/touch incidents
    # ------------------------------------------------------------------
    async def on_alerts_fired(
        self,
        alerts: list[FiredAlert],
        *,
        client_id: str = "",
    ) -> list[str]:
        """
        Open or touch one incident per alert. Returns the incident ids,
        in the same order as the input alerts. Duplicate alerts (same
        rule + scope + client) collapse to a single incident via the
        store's dedup logic.
        """
        incident_ids: list[str] = []
        for alert in alerts:
            inc = self._incident_from_alert(alert, client_id=client_id)
            iid = await self.store.open_or_touch(inc)
            incident_ids.append(iid)
        return incident_ids

    # ------------------------------------------------------------------
    # Job success → resolve any open incident that mentions this job
    # ------------------------------------------------------------------
    async def on_job_succeeded(self, job_id: str) -> int:
        """
        Resolve every open incident whose `affected_jobs` contains
        `job_id`. Returns the number of incidents resolved.

        This is the natural "the underlying problem went away" signal:
        a job that previously failed (and opened an incident) now runs
        clean, so the incident resolves itself.
        """
        if not job_id:
            return 0

        resolved = 0
        # Snapshot list — safe to iterate while mutating the store.
        for inc in await self.store.list_open():
            if job_id not in inc.affected_jobs:
                continue
            ok = await self.store.update_status(
                inc.incident_id, IncidentStatus.RESOLVED,
            )
            if ok:
                resolved += 1
        return resolved

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _incident_from_alert(
        self,
        alert: FiredAlert,
        *,
        client_id: str = "",
    ) -> Incident:
        event = alert.event or {}
        job_id = str(event.get("job_id") or "")
        domain = str(event.get("domain") or "")

        return Incident(
            dedup_key=self._dedup_key(alert, client_id),
            title=f"Alert: {alert.condition_type}",
            description=alert.message,
            severity=_SEVERITY_MAP.get(alert.severity, IncidentSeverity.WARNING),
            status=IncidentStatus.OPEN,
            failure_class=_CONDITION_TO_FAILURE_CLASS.get(
                alert.condition_type, FailureClass.UNKNOWN,
            ),
            detection_source=_CONDITION_TO_DETECTION_SOURCE.get(
                alert.condition_type, DetectionSource.ALERT_RULE,
            ),
            client_id=client_id or "default",
            affected_jobs=[job_id] if job_id else [],
            affected_domains=[domain] if domain else [],
            linked_alert_ids=[alert.rule_id] if alert.rule_id else [],
        )

    @staticmethod
    def _dedup_key(alert: FiredAlert, client_id: str) -> str:
        """
        One incident per (rule, scope_id, client). `scope_id` may be
        empty (client-wide rules); use "_" so the key stays readable.
        """
        scope_id = alert.scope_id or "_"
        tenant = client_id or "default"
        return f"{alert.rule_id}::{scope_id}::{tenant}"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    from src.observability.store import InMemoryIncidentStore

    async def run():
        # ------------------------------------------------------------------
        # 1. Empty alert list → no incidents, no exception
        # ------------------------------------------------------------------
        store = InMemoryIncidentStore()
        tracker = IncidentTracker(store)
        assert await tracker.on_alerts_fired([]) == []

        # ------------------------------------------------------------------
        # 2. One alert → one incident, fully mapped
        # ------------------------------------------------------------------
        alert = FiredAlert(
            rule_id="rule-1",
            condition_type="job_failed",
            severity="critical",
            scope="task",
            scope_id="t-1",
            message="job j-1 failed",
            dedup_key="ignored-by-incident-store",
            event={"job_id": "j-1", "domain": "shop.example"},
        )
        ids = await tracker.on_alerts_fired([alert], client_id="acme")
        assert len(ids) == 1

        inc = await store.get(ids[0])
        assert inc is not None
        assert inc.severity == IncidentSeverity.CRITICAL
        assert inc.status == IncidentStatus.OPEN
        assert inc.failure_class == FailureClass.UNKNOWN
        assert inc.detection_source == DetectionSource.ALERT_RULE
        assert inc.client_id == "acme"
        assert inc.affected_jobs == ["j-1"]
        assert inc.affected_domains == ["shop.example"]
        assert inc.linked_alert_ids == ["rule-1"]
        assert inc.occurrence_count == 1

        # ------------------------------------------------------------------
        # 3. Same alert again → same incident, count bumped
        # ------------------------------------------------------------------
        ids2 = await tracker.on_alerts_fired([alert], client_id="acme")
        assert ids2 == ids
        inc = await store.get(ids[0])
        assert inc.occurrence_count == 2

        # ------------------------------------------------------------------
        # 4. Severity escalation on touch (uses the rank fix)
        # ------------------------------------------------------------------
        store4 = InMemoryIncidentStore()
        tracker4 = IncidentTracker(store4)

        warn = FiredAlert(
            rule_id="r-q", condition_type="quality_below",
            severity="warning", scope="task", scope_id="t-9",
            message="quality dropped", dedup_key="d",
            event={"job_id": "j-9"},
        )
        warn_ids = await tracker4.on_alerts_fired([warn], client_id="acme")
        inc4 = await store4.get(warn_ids[0])
        assert inc4.severity == IncidentSeverity.WARNING
        assert inc4.failure_class == FailureClass.DATA_QUALITY

        crit = FiredAlert(
            rule_id="r-q", condition_type="quality_below",
            severity="critical", scope="task", scope_id="t-9",
            message="quality collapsed", dedup_key="d",
            event={"job_id": "j-9"},
        )
        crit_ids = await tracker4.on_alerts_fired([crit], client_id="acme")
        assert crit_ids == warn_ids
        inc4 = await store4.get(warn_ids[0])
        assert inc4.severity == IncidentSeverity.CRITICAL
        assert inc4.occurrence_count == 2

        # ------------------------------------------------------------------
        # 5. Different client → different incident
        # ------------------------------------------------------------------
        other_ids = await tracker.on_alerts_fired([alert], client_id="other")
        assert other_ids != ids
        other_inc = await store.get(other_ids[0])
        assert other_inc.client_id == "other"

        # ------------------------------------------------------------------
        # 6. Circuit breaker → own detection source + BUDGET class
        # ------------------------------------------------------------------
        cb_alert = FiredAlert(
            rule_id="r-cb", condition_type="circuit_breaker_trip",
            severity="critical", scope="job", scope_id="j-42",
            message="budget cap reached", dedup_key="cb",
            event={"job_id": "j-42"},
        )
        cb_ids = await tracker.on_alerts_fired([cb_alert], client_id="acme")
        cb_inc = await store.get(cb_ids[0])
        assert cb_inc.detection_source == DetectionSource.CIRCUIT_BREAKER
        assert cb_inc.failure_class == FailureClass.BUDGET
        assert cb_inc.affected_jobs == ["j-42"]

        # ------------------------------------------------------------------
        # 7. Job success resolves the incident
        # ------------------------------------------------------------------
        resolved = await tracker.on_job_succeeded("j-1")
        assert resolved == 1
        inc = await store.get(ids[0])
        assert inc.status == IncidentStatus.RESOLVED
        assert inc.resolved_at  # timestamp was stamped

        # ------------------------------------------------------------------
        # 8. Second call → already resolved, count returns 0
        # ------------------------------------------------------------------
        resolved = await tracker.on_job_succeeded("j-1")
        assert resolved == 0

        # ------------------------------------------------------------------
        # 9. Unknown severity / condition falls back safely
        # ------------------------------------------------------------------
        weird = FiredAlert(
            rule_id="r-weird", condition_type="totally_unknown",
            severity="bogus-severity", scope="task", scope_id="t-x",
            message="?", dedup_key="w",
        )
        weird_ids = await tracker.on_alerts_fired([weird], client_id="acme")
        weird_inc = await store.get(weird_ids[0])
        assert weird_inc.severity == IncidentSeverity.WARNING
        assert weird_inc.failure_class == FailureClass.UNKNOWN
        assert weird_inc.detection_source == DetectionSource.ALERT_RULE

        # ------------------------------------------------------------------
        # 10. on_job_succeeded("") is a clean no-op
        # ------------------------------------------------------------------
        assert await tracker.on_job_succeeded("") == 0

        print("Incident tracker OK.")

    asyncio.run(run())