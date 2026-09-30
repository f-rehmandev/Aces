"""
Unit tests for the alert → incident bridge (spec §30A + §40.6).

Covers:
    - on_alerts_fired: severity mapping, failure-class mapping,
      detection-source mapping, dedup via the store, escalation,
      cross-client isolation, unknown-condition fallback
    - on_job_succeeded: resolves matching open incidents, idempotent,
      no-op on empty job_id, does not resolve unrelated incidents
    - dedup key shape (regression guard — key must be stable)
"""
import asyncio

import pytest

from src.alerts.rules import FiredAlert
from src.observability.store import InMemoryIncidentStore
from src.observability.tracker import IncidentTracker
from src.observability.types import (
    DetectionSource,
    FailureClass,
    IncidentSeverity,
    IncidentStatus,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def env():
    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)
    return tracker, store


def _alert(**kw) -> FiredAlert:
    base = dict(
        rule_id="r-1",
        condition_type="job_failed",
        severity="warning",
        scope="task",
        scope_id="t-1",
        message="something happened",
        dedup_key="d",
        event={"job_id": "j-1", "domain": "shop.example"},
    )
    base.update(kw)
    return FiredAlert(**base)


# ===========================================================================
# on_alerts_fired — happy path
# ===========================================================================

def test_empty_list_opens_nothing(env):
    tracker, store = env
    assert _run(tracker.on_alerts_fired([])) == []
    assert _run(store.list_open()) == []


def test_single_alert_opens_incident(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired([_alert()], client_id="acme"))
    assert len(ids) == 1
    assert _run(store.get(ids[0])) is not None


def test_returns_one_id_per_alert(env):
    tracker, _ = env
    alerts = [
        _alert(rule_id="r-1", scope_id="t-1"),
        _alert(rule_id="r-2", scope_id="t-2"),
        _alert(rule_id="r-3", scope_id="t-3"),
    ]
    ids = _run(tracker.on_alerts_fired(alerts, client_id="acme"))
    assert len(ids) == 3
    assert len(set(ids)) == 3


# ===========================================================================
# Field mapping
# ===========================================================================

def test_severity_critical_maps(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(severity="critical")], client_id="acme",
    ))
    assert _run(store.get(ids[0])).severity == IncidentSeverity.CRITICAL


def test_severity_warning_maps(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(severity="warning")], client_id="acme",
    ))
    assert _run(store.get(ids[0])).severity == IncidentSeverity.WARNING


def test_severity_info_maps(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(severity="info")], client_id="acme",
    ))
    assert _run(store.get(ids[0])).severity == IncidentSeverity.INFO


def test_unknown_severity_falls_back_to_warning(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(severity="nonsense")], client_id="acme",
    ))
    assert _run(store.get(ids[0])).severity == IncidentSeverity.WARNING


@pytest.mark.parametrize("condition,failure_class", [
    ("quality_below",          FailureClass.DATA_QUALITY),
    ("confidence_collapse",    FailureClass.DATA_QUALITY),
    ("record_count_drop",      FailureClass.DATA_QUALITY),
    ("record_count_rise",      FailureClass.DATA_QUALITY),
    ("source_conflict_spike",  FailureClass.DATA_QUALITY),
    ("unexpected_removals",    FailureClass.DATA_QUALITY),
    ("schema_change",          FailureClass.SCHEMA),
    ("budget_threshold",       FailureClass.BUDGET),
    ("circuit_breaker_trip",   FailureClass.BUDGET),
    ("self_healing_event",     FailureClass.INFRASTRUCTURE),
    ("job_failed",             FailureClass.UNKNOWN),
    ("repeated_job_failure",   FailureClass.UNKNOWN),
])
def test_condition_to_failure_class_mapping(env, condition, failure_class):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(condition_type=condition)], client_id="acme",
    ))
    assert _run(store.get(ids[0])).failure_class == failure_class


def test_unknown_condition_falls_back_to_unknown(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(condition_type="mystery_condition")], client_id="acme",
    ))
    assert _run(store.get(ids[0])).failure_class == FailureClass.UNKNOWN


def test_default_detection_source_is_alert_rule(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(condition_type="job_failed")], client_id="acme",
    ))
    assert _run(store.get(ids[0])).detection_source == DetectionSource.ALERT_RULE


def test_circuit_breaker_trip_has_its_own_detection_source(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(condition_type="circuit_breaker_trip")], client_id="acme",
    ))
    assert _run(store.get(ids[0])).detection_source == DetectionSource.CIRCUIT_BREAKER


def test_affected_jobs_and_domains_extracted_from_event(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(event={"job_id": "j-99", "domain": "shop.example"})],
        client_id="acme",
    ))
    inc = _run(store.get(ids[0]))
    assert inc.affected_jobs == ["j-99"]
    assert inc.affected_domains == ["shop.example"]


def test_missing_event_fields_produce_empty_lists(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(event={})], client_id="acme",
    ))
    inc = _run(store.get(ids[0]))
    assert inc.affected_jobs == []
    assert inc.affected_domains == []


def test_no_event_at_all_does_not_crash(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(event=None)], client_id="acme",
    ))
    inc = _run(store.get(ids[0]))
    assert inc.affected_jobs == []


def test_linked_alert_ids_contains_rule_id(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(rule_id="rule-xyz")], client_id="acme",
    ))
    assert _run(store.get(ids[0])).linked_alert_ids == ["rule-xyz"]


def test_empty_client_id_defaults_to_default_tenant(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired([_alert()], client_id=""))
    assert _run(store.get(ids[0])).client_id == "default"


# ===========================================================================
# Dedup + escalation
# ===========================================================================

def test_same_alert_twice_dedupes(env):
    tracker, store = env
    ids1 = _run(tracker.on_alerts_fired([_alert()], client_id="acme"))
    ids2 = _run(tracker.on_alerts_fired([_alert()], client_id="acme"))
    assert ids1 == ids2
    assert _run(store.get(ids1[0])).occurrence_count == 2


def test_different_scope_id_creates_separate_incidents(env):
    tracker, _ = env
    ids_a = _run(tracker.on_alerts_fired(
        [_alert(rule_id="r", scope_id="t-a")], client_id="acme",
    ))
    ids_b = _run(tracker.on_alerts_fired(
        [_alert(rule_id="r", scope_id="t-b")], client_id="acme",
    ))
    assert ids_a != ids_b


def test_different_rule_creates_separate_incidents(env):
    tracker, _ = env
    ids_a = _run(tracker.on_alerts_fired(
        [_alert(rule_id="r-a")], client_id="acme",
    ))
    ids_b = _run(tracker.on_alerts_fired(
        [_alert(rule_id="r-b")], client_id="acme",
    ))
    assert ids_a != ids_b


def test_different_client_creates_separate_incidents(env):
    tracker, _ = env
    ids_a = _run(tracker.on_alerts_fired([_alert()], client_id="acme"))
    ids_b = _run(tracker.on_alerts_fired([_alert()], client_id="other"))
    assert ids_a != ids_b


def test_escalation_on_touch(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(severity="warning")], client_id="acme",
    ))
    assert _run(store.get(ids[0])).severity == IncidentSeverity.WARNING
    _run(tracker.on_alerts_fired(
        [_alert(severity="critical")], client_id="acme",
    ))
    assert _run(store.get(ids[0])).severity == IncidentSeverity.CRITICAL


def test_no_downgrade_on_touch(env):
    tracker, store = env
    _run(tracker.on_alerts_fired(
        [_alert(severity="critical")], client_id="acme",
    ))
    _run(tracker.on_alerts_fired(
        [_alert(severity="warning")], client_id="acme",
    ))
    ids = _run(store.list_open(client_id="acme"))
    assert len(ids) == 1
    assert ids[0].severity == IncidentSeverity.CRITICAL


# ===========================================================================
# on_job_succeeded
# ===========================================================================

def test_job_succeeded_resolves_matching_incident(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(event={"job_id": "j-1"})], client_id="acme",
    ))
    resolved = _run(tracker.on_job_succeeded("j-1"))
    assert resolved == 1
    assert _run(store.get(ids[0])).status == IncidentStatus.RESOLVED


def test_job_succeeded_is_idempotent(env):
    tracker, _ = env
    _run(tracker.on_alerts_fired(
        [_alert(event={"job_id": "j-1"})], client_id="acme",
    ))
    assert _run(tracker.on_job_succeeded("j-1")) == 1
    assert _run(tracker.on_job_succeeded("j-1")) == 0


def test_job_succeeded_leaves_unrelated_incidents_open(env):
    tracker, store = env
    _run(tracker.on_alerts_fired(
        [_alert(rule_id="r-a", event={"job_id": "j-1"})],
        client_id="acme",
    ))
    _run(tracker.on_alerts_fired(
        [_alert(rule_id="r-b", event={"job_id": "j-2"})],
        client_id="acme",
    ))
    resolved = _run(tracker.on_job_succeeded("j-1"))
    assert resolved == 1
    opens = _run(store.list_open(client_id="acme"))
    assert len(opens) == 1
    assert "j-2" in opens[0].affected_jobs


def test_job_succeeded_empty_string_is_noop(env):
    tracker, _ = env
    _run(tracker.on_alerts_fired([_alert()], client_id="acme"))
    assert _run(tracker.on_job_succeeded("")) == 0


def test_job_succeeded_unknown_job_is_noop(env):
    tracker, _ = env
    _run(tracker.on_alerts_fired([_alert()], client_id="acme"))
    assert _run(tracker.on_job_succeeded("never-seen-job")) == 0


def test_job_succeeded_across_clients(env):
    """A single job_id appears only in one client's incidents."""
    tracker, store = env
    _run(tracker.on_alerts_fired(
        [_alert(event={"job_id": "j-1"})], client_id="acme",
    ))
    _run(tracker.on_alerts_fired(
        [_alert(event={"job_id": "j-1"})], client_id="other",
    ))
    # Resolves both, because both have j-1. That's intentional: the job
    # id namespace is global to the tracker today. Documented behavior.
    resolved = _run(tracker.on_job_succeeded("j-1"))
    assert resolved == 2


# ===========================================================================
# Dedup key shape — regression guard
# ===========================================================================

def test_dedup_key_shape(env):
    tracker, _ = env
    alert = _alert(rule_id="r-1", scope_id="t-1")
    key = tracker._dedup_key(alert, "acme")
    assert key == "r-1::t-1::acme"


def test_dedup_key_uses_underscore_for_empty_scope(env):
    tracker, _ = env
    alert = _alert(rule_id="r-1", scope_id="")
    key = tracker._dedup_key(alert, "acme")
    assert key == "r-1::_::acme"


def test_dedup_key_uses_default_for_empty_client(env):
    tracker, _ = env
    alert = _alert(rule_id="r-1", scope_id="t-1")
    key = tracker._dedup_key(alert, "")
    assert key == "r-1::t-1::default"


# ===========================================================================
# Log-only title / message
# ===========================================================================

def test_incident_title_and_description(env):
    tracker, store = env
    ids = _run(tracker.on_alerts_fired(
        [_alert(condition_type="quality_below", message="quality at 0.42")],
        client_id="acme",
    ))
    inc = _run(store.get(ids[0]))
    assert "quality_below" in inc.title
    assert inc.description == "quality at 0.42"