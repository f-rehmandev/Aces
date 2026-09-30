"""
Unit tests for the observability store (spec §40.6).

Covers:
    - InMemoryIncidentStore: dedup, severity escalation (the bug we
      just fixed), client isolation, status transitions, terminal
      handling, affected-list merging
    - InMemorySLOStore: sample idempotency, window filtering,
      evaluation across all statuses and both directions
      (AT_LEAST / AT_MOST)
    - Row translation helpers used by the Supabase path
    - Severity rank ordering (regression guard for the escalation fix)

Style matches the rest of tests/unit/: plain functions, `_run()`
wrapper around asyncio.run, minimal fixtures.
"""
import asyncio

import pytest

from src.observability.store import (
    InMemoryIncidentStore,
    InMemorySLOStore,
    _row_to_incident,
    _row_to_sample,
)
from src.observability.types import (
    _SEVERITY_RANK,
    DetectionSource,
    FailureClass,
    Incident,
    IncidentSeverity,
    IncidentStatus,
    SLO,
    SLODirection,
    SLOSample,
    SLOStatus,
    is_terminal_status,
    severity_rank,
)


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# Incidents
# ===========================================================================

@pytest.fixture
def incidents():
    return InMemoryIncidentStore()


def test_incident_open_returns_id_and_stores(incidents):
    inc = Incident(title="provider down", client_id="acme")
    iid = _run(incidents.open_or_touch(inc))
    assert iid == inc.incident_id
    assert _run(incidents.get(iid)) is inc


def test_incident_dedup_same_key_same_client(incidents):
    a = Incident(title="x", dedup_key="k", client_id="acme")
    b = Incident(title="x", dedup_key="k", client_id="acme")
    iid_a = _run(incidents.open_or_touch(a))
    iid_b = _run(incidents.open_or_touch(b))
    assert iid_a == iid_b
    assert _run(incidents.get(iid_a)).occurrence_count == 2


def test_incident_dedup_escalates_warning_to_critical(incidents):
    """Regression: alphabetical severity comparison used to block this."""
    inc = Incident(title="x", dedup_key="k", client_id="acme",
                   severity=IncidentSeverity.WARNING)
    _run(incidents.open_or_touch(inc))

    inc2 = Incident(title="x", dedup_key="k", client_id="acme",
                    severity=IncidentSeverity.CRITICAL)
    _run(incidents.open_or_touch(inc2))

    got = _run(incidents.get(inc.incident_id))
    assert got.severity == IncidentSeverity.CRITICAL


def test_incident_dedup_never_downgrades_critical(incidents):
    """Critical -> warning must NOT lower the existing incident."""
    inc = Incident(title="x", dedup_key="k", client_id="acme",
                   severity=IncidentSeverity.CRITICAL)
    _run(incidents.open_or_touch(inc))

    inc2 = Incident(title="x", dedup_key="k", client_id="acme",
                    severity=IncidentSeverity.WARNING)
    _run(incidents.open_or_touch(inc2))

    got = _run(incidents.get(inc.incident_id))
    assert got.severity == IncidentSeverity.CRITICAL


def test_incident_different_key_creates_separate(incidents):
    a = Incident(title="x", dedup_key="k1", client_id="acme")
    b = Incident(title="x", dedup_key="k2", client_id="acme")
    assert _run(incidents.open_or_touch(a)) != _run(incidents.open_or_touch(b))


def test_incident_same_key_different_client_isolated(incidents):
    a = Incident(title="x", dedup_key="k", client_id="acme")
    b = Incident(title="x", dedup_key="k", client_id="other")
    assert _run(incidents.open_or_touch(a)) != _run(incidents.open_or_touch(b))


def test_incident_dedup_merges_affected_lists(incidents):
    inc = Incident(title="x", dedup_key="k", client_id="acme",
                   affected_jobs=["j-1"], affected_domains=["a.example"])
    _run(incidents.open_or_touch(inc))

    inc2 = Incident(title="x", dedup_key="k", client_id="acme",
                    affected_jobs=["j-2"], affected_domains=["b.example"])
    _run(incidents.open_or_touch(inc2))

    got = _run(incidents.get(inc.incident_id))
    assert set(got.affected_jobs) == {"j-1", "j-2"}
    assert set(got.affected_domains) == {"a.example", "b.example"}


def test_incident_dedup_does_not_duplicate_list_entries(incidents):
    inc = Incident(title="x", dedup_key="k", client_id="acme",
                   affected_jobs=["j-1"])
    _run(incidents.open_or_touch(inc))
    inc2 = Incident(title="x", dedup_key="k", client_id="acme",
                    affected_jobs=["j-1"])
    _run(incidents.open_or_touch(inc2))
    got = _run(incidents.get(inc.incident_id))
    assert got.affected_jobs == ["j-1"]


def test_incident_list_open_excludes_terminal(incidents):
    a = Incident(title="x", dedup_key="k1", client_id="acme")
    b = Incident(title="y", dedup_key="k2", client_id="acme")
    _run(incidents.open_or_touch(a))
    iid_b = _run(incidents.open_or_touch(b))

    _run(incidents.update_status(iid_b, IncidentStatus.RESOLVED))
    opens = _run(incidents.list_open(client_id="acme"))
    assert len(opens) == 1
    assert opens[0].dedup_key == "k1"


def test_incident_list_open_scoped_by_client(incidents):
    _run(incidents.open_or_touch(
        Incident(title="x", dedup_key="k1", client_id="acme")))
    _run(incidents.open_or_touch(
        Incident(title="y", dedup_key="k2", client_id="other")))

    assert len(_run(incidents.list_open(client_id="acme"))) == 1
    assert len(_run(incidents.list_open(client_id="other"))) == 1
    assert len(_run(incidents.list_open())) == 2


def test_incident_list_recent_includes_terminal(incidents):
    a = Incident(title="x", dedup_key="k1", client_id="acme")
    iid = _run(incidents.open_or_touch(a))
    _run(incidents.update_status(iid, IncidentStatus.CLOSED))
    recent = _run(incidents.list_recent(client_id="acme"))
    assert len(recent) == 1
    assert recent[0].status == IncidentStatus.CLOSED


def test_incident_dedup_key_freed_after_resolution(incidents):
    """Once resolved, a fresh incident with the same dedup_key can open."""
    a = Incident(title="x", dedup_key="k", client_id="acme")
    iid_a = _run(incidents.open_or_touch(a))
    _run(incidents.update_status(iid_a, IncidentStatus.RESOLVED))

    b = Incident(title="x", dedup_key="k", client_id="acme")
    iid_b = _run(incidents.open_or_touch(b))
    assert iid_b != iid_a


def test_update_status_missing_returns_false(incidents):
    assert not _run(incidents.update_status("nope", IncidentStatus.CLOSED))


def test_update_status_acknowledge_records_actor(incidents):
    inc = Incident(title="x")
    iid = _run(incidents.open_or_touch(inc))
    _run(incidents.update_status(
        iid, IncidentStatus.ACKNOWLEDGED, acknowledged_by="op-1",
    ))
    got = _run(incidents.get(iid))
    assert got.status == IncidentStatus.ACKNOWLEDGED
    assert got.acknowledged_by == "op-1"
    assert got.acknowledged_at


def test_update_status_resolve_stamps_resolved_at(incidents):
    iid = _run(incidents.open_or_touch(Incident(title="x")))
    _run(incidents.update_status(iid, IncidentStatus.RESOLVED))
    assert _run(incidents.get(iid)).resolved_at


def test_update_status_close_stamps_closed_at(incidents):
    iid = _run(incidents.open_or_touch(Incident(title="x")))
    _run(incidents.update_status(iid, IncidentStatus.CLOSED))
    assert _run(incidents.get(iid)).closed_at


# ===========================================================================
# SLOs
# ===========================================================================

@pytest.fixture
def slos():
    return InMemorySLOStore()


def _make_slo(**kw) -> SLO:
    base = dict(
        name="job success rate",
        metric="job_success_rate",
        unit="ratio",
        direction=SLODirection.AT_LEAST,
        target=0.95,
        window_days=7,
        min_samples=3,
    )
    base.update(kw)
    return SLO(**base)


def _samples(slo_id, values, client_id="acme"):
    return [
        SLOSample(slo_id=slo_id, value=v, client_id=client_id)
        for v in values
    ]


def test_slo_save_and_get(slos):
    slo = _make_slo()
    _run(slos.save_slo(slo))
    got = _run(slos.get_slo(slo.slo_id))
    assert got is not None
    assert got.name == "job success rate"


def test_slo_list_filters_by_scope(slos):
    _run(slos.save_slo(_make_slo(scope="client", scope_id="a")))
    _run(slos.save_slo(_make_slo(scope="client", scope_id="b")))
    _run(slos.save_slo(_make_slo(scope="system")))
    assert len(_run(slos.list_slos())) == 3
    assert len(_run(slos.list_slos(scope="client", scope_id="a"))) == 1
    assert len(_run(slos.list_slos(scope="client"))) == 2


def test_slo_record_samples_idempotent(slos):
    slo = _make_slo()
    _run(slos.save_slo(slo))
    batch = _samples(slo.slo_id, [0.98, 0.97])
    n1 = _run(slos.record_samples(batch))
    n2 = _run(slos.record_samples(batch))
    assert n1 == 2
    assert n2 == 0


def test_slo_evaluate_insufficient_data(slos):
    slo = _make_slo(min_samples=5)
    _run(slos.save_slo(slo))
    _run(slos.record_samples(_samples(slo.slo_id, [0.98, 0.97])))
    ev = _run(slos.evaluate(slo))
    assert ev.status == SLOStatus.INSUFFICIENT_DATA
    assert ev.sample_count == 2


def test_slo_evaluate_meeting_at_least(slos):
    slo = _make_slo(target=0.95, min_samples=3)
    _run(slos.save_slo(slo))
    _run(slos.record_samples(_samples(slo.slo_id, [0.98, 0.97, 0.99])))
    ev = _run(slos.evaluate(slo))
    assert ev.status == SLOStatus.MEETING
    assert ev.margin is not None and ev.margin > 0


def test_slo_evaluate_at_risk_at_least(slos):
    # target 0.95, mean 0.93 → margin = -0.02, |0.02| <= 0.095 → at_risk
    slo = _make_slo(target=0.95, min_samples=3)
    _run(slos.save_slo(slo))
    _run(slos.record_samples(_samples(slo.slo_id, [0.93, 0.93, 0.93])))
    ev = _run(slos.evaluate(slo))
    assert ev.status == SLOStatus.AT_RISK


def test_slo_evaluate_breached_at_least(slos):
    slo = _make_slo(target=0.95, min_samples=3)
    _run(slos.save_slo(slo))
    _run(slos.record_samples(_samples(slo.slo_id, [0.5, 0.5, 0.5])))
    ev = _run(slos.evaluate(slo))
    assert ev.status == SLOStatus.BREACHED


def test_slo_evaluate_meeting_at_most(slos):
    slo = _make_slo(
        metric="queue_latency", unit="seconds",
        direction=SLODirection.AT_MOST, target=30.0, min_samples=2,
    )
    _run(slos.save_slo(slo))
    _run(slos.record_samples(_samples(slo.slo_id, [25.0, 27.0])))
    ev = _run(slos.evaluate(slo))
    assert ev.status == SLOStatus.MEETING


def test_slo_evaluate_at_risk_at_most(slos):
    # target 30, mean 32 → margin = -2, |2| <= 30 * 0.1 = 3 → at_risk
    slo = _make_slo(
        metric="queue_latency", unit="seconds",
        direction=SLODirection.AT_MOST, target=30.0, min_samples=2,
    )
    _run(slos.save_slo(slo))
    _run(slos.record_samples(_samples(slo.slo_id, [32.0, 32.0])))
    ev = _run(slos.evaluate(slo))
    assert ev.status == SLOStatus.AT_RISK


def test_slo_evaluate_breached_at_most(slos):
    slo = _make_slo(
        metric="error_rate", unit="ratio",
        direction=SLODirection.AT_MOST, target=0.05, min_samples=2,
    )
    _run(slos.save_slo(slo))
    _run(slos.record_samples(_samples(slo.slo_id, [0.20, 0.20])))
    ev = _run(slos.evaluate(slo))
    assert ev.status == SLOStatus.BREACHED


def test_slo_evaluate_disabled(slos):
    slo = _make_slo(enabled=False, min_samples=1)
    _run(slos.save_slo(slo))
    _run(slos.record_samples(_samples(slo.slo_id, [0.0])))
    ev = _run(slos.evaluate(slo))
    assert ev.status == SLOStatus.DISABLED


def test_slo_samples_in_window_excludes_old(slos):
    slo = _make_slo(min_samples=1)
    _run(slos.save_slo(slo))
    old = SLOSample(slo_id=slo.slo_id, value=0.5,
                    sampled_at="2020-01-01T00:00:00.000Z")
    new = SLOSample(slo_id=slo.slo_id, value=0.9,
                    sampled_at="2099-01-01T00:00:00.000Z")
    _run(slos.record_samples([old, new]))
    window = _run(slos.samples_in_window(
        slo.slo_id, since="2025-01-01T00:00:00.000Z",
    ))
    assert len(window) == 1
    assert window[0].value == 0.9


def test_slo_evaluate_reports_metric_and_unit(slos):
    slo = _make_slo(
        metric="queue_latency", unit="seconds",
        direction=SLODirection.AT_MOST, target=30.0, min_samples=2,
    )
    _run(slos.save_slo(slo))
    _run(slos.record_samples(_samples(slo.slo_id, [25.0, 27.0])))
    ev = _run(slos.evaluate(slo))
    assert ev.target == 30.0
    assert ev.direction == "at_most"
    assert ev.current_value is not None


# ===========================================================================
# Row translation (Supabase path)
# ===========================================================================

def test_row_to_incident_basic():
    row = {
        "incident_id": "abc",
        "dedup_key": "k",
        "title": "x",
        "severity": "critical",
        "status": "acknowledged",
        "failure_class": "provider",
        "detection_source": "alert_rule",
        "client_id": "acme",
        "affected_jobs": ["j-1"],
        "affected_domains": ["a.example"],
        "occurrence_count": 3,
    }
    inc = _row_to_incident(row)
    assert inc.severity == IncidentSeverity.CRITICAL
    assert inc.status == IncidentStatus.ACKNOWLEDGED
    assert inc.failure_class == FailureClass.PROVIDER
    assert inc.detection_source == DetectionSource.ALERT_RULE
    assert inc.occurrence_count == 3
    assert inc.affected_jobs == ["j-1"]


def test_row_to_incident_bad_enums_fall_back_safely():
    row = {
        "severity": "bogus", "status": "nope",
        "failure_class": "??", "detection_source": "whatever",
    }
    inc = _row_to_incident(row)
    assert inc.severity == IncidentSeverity.WARNING
    assert inc.status == IncidentStatus.OPEN
    assert inc.failure_class == FailureClass.UNKNOWN
    assert inc.detection_source == DetectionSource.AUTOMATIC


def test_row_to_incident_coerces_malformed_fields():
    row = {
        "affected_jobs": None, "affected_domains": "not a list",
        "linked_alert_ids": None, "metadata": "junk",
    }
    inc = _row_to_incident(row)
    assert inc.affected_jobs == []
    assert inc.affected_domains == []
    assert inc.linked_alert_ids == []
    assert inc.metadata == {}


def test_row_to_sample_basic():
    row = {"slo_id": "s", "value": "0.95", "client_id": "acme"}
    s = _row_to_sample(row)
    assert s.value == 0.95
    assert s.metadata == {}


def test_row_to_sample_bad_value_defaults_to_zero():
    row = {"slo_id": "s", "value": "not a number"}
    assert _row_to_sample(row).value == 0.0


# ===========================================================================
# Severity rank — regression guard for the Y.2.4-fix
# ===========================================================================

def test_severity_rank_is_monotonic():
    assert _SEVERITY_RANK[IncidentSeverity.INFO] < \
           _SEVERITY_RANK[IncidentSeverity.WARNING] < \
           _SEVERITY_RANK[IncidentSeverity.CRITICAL]


def test_severity_rank_property_matches_table():
    assert IncidentSeverity.INFO.rank == 0
    assert IncidentSeverity.WARNING.rank == 1
    assert IncidentSeverity.CRITICAL.rank == 2


def test_severity_rank_free_function():
    assert severity_rank(IncidentSeverity.CRITICAL) == 2


def test_string_enum_values_are_not_alphabetically_ordered():
    """
    Documents WHY rank exists. If anyone ever "simplifies" the store
    back to comparing .value strings, this test explains the trap.
    """
    assert IncidentSeverity.CRITICAL.value < IncidentSeverity.WARNING.value
    assert IncidentSeverity.CRITICAL.rank > IncidentSeverity.WARNING.rank


# ===========================================================================
# Terminal status helper
# ===========================================================================

def test_is_terminal_status_helper():
    assert is_terminal_status(IncidentStatus.RESOLVED)
    assert is_terminal_status(IncidentStatus.CLOSED)
    assert not is_terminal_status(IncidentStatus.OPEN)
    assert not is_terminal_status(IncidentStatus.ACKNOWLEDGED)
    assert not is_terminal_status(IncidentStatus.MITIGATING)