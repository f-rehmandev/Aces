"""Unit tests for the publication gate (spec §22.5, §22.6)."""
import pytest

from src.quality.publication import (
    PublicationGate, PublicationDecision, DatasetState,
)
from src.quality.rules import QualityResult
from src.quality.anomaly import AnomalyFinding, AnomalyReport


def _passed() -> QualityResult:
    return QualityResult(passed=True, score=1.0)


def _failed(rules=None) -> QualityResult:
    return QualityResult(
        passed=False, score=0.3,
        failed_rules=rules or ["min_records"],
        explanation="min_records: 0 < 5",
    )


def _critical_anomaly() -> AnomalyReport:
    return AnomalyReport(findings=[
        AnomalyFinding(kind="sudden_zero", field_name="email",
                       detail="x", severity="critical"),
    ])


def _warning_anomaly() -> AnomalyReport:
    return AnomalyReport(findings=[
        AnomalyFinding(kind="distribution_shift", field_name="cat",
                       detail="x", severity="warning"),
    ])


# --- clean path --------------------------------------------------------

def test_clean_pass_is_publishable():
    d = PublicationGate().evaluate(_passed())
    assert d.allowed
    assert d.state == DatasetState.READY_TO_PUBLISH
    assert d.reason == "all gates passed"


def test_clean_pass_no_anomaly_report():
    d = PublicationGate().evaluate(_passed(), anomaly=None)
    assert d.allowed


# --- quality fail ------------------------------------------------------

def test_quality_fail_blocks_publication():
    d = PublicationGate().evaluate(_failed())
    assert not d.allowed
    assert d.state == DatasetState.QUALITY_FAILED
    assert d.requires_human_approval


def test_quality_fail_reason_mentions_rule():
    d = PublicationGate().evaluate(_failed(["duplicate_rate"]))
    assert "duplicate_rate" in d.reason


# --- anomaly -----------------------------------------------------------

def test_critical_anomaly_quarantines():
    d = PublicationGate().evaluate(_passed(), anomaly=_critical_anomaly())
    assert not d.allowed
    assert d.state == DatasetState.QUARANTINED
    assert d.has_critical_anomaly
    assert d.requires_human_approval


def test_warning_anomaly_allows_publication():
    d = PublicationGate().evaluate(_passed(), anomaly=_warning_anomaly())
    assert d.allowed
    assert d.state == DatasetState.READY_TO_PUBLISH
    assert "non-critical" in d.reason


def test_block_on_critical_anomaly_false_allows():
    d = PublicationGate(block_on_critical_anomaly=False).evaluate(
        _passed(), anomaly=_critical_anomaly(),
    )
    assert d.allowed


# --- quality fail beats anomaly state ---------------------------------

def test_quality_fail_takes_precedence_over_anomaly():
    # If both quality fails AND critical anomaly exist, the state is
    # QUALITY_FAILED (checked first).
    d = PublicationGate().evaluate(_failed(), anomaly=_critical_anomaly())
    assert d.state == DatasetState.QUALITY_FAILED


# --- override ---------------------------------------------------------

def test_override_flips_blocked_decision():
    gate = PublicationGate()
    blocked = gate.evaluate(_failed())
    overridden = gate.override(blocked, "verified manually with operator")
    assert overridden.allowed
    assert overridden.state == DatasetState.READY_TO_PUBLISH
    assert "verified" in overridden.reason


def test_override_requires_justification():
    gate = PublicationGate()
    blocked = gate.evaluate(_failed())
    with pytest.raises(ValueError):
        gate.override(blocked, "")


def test_override_whitespace_justification_rejected():
    gate = PublicationGate()
    blocked = gate.evaluate(_failed())
    with pytest.raises(ValueError):
        gate.override(blocked, "   \n  ")


def test_override_preserves_critical_anomaly_flag():
    gate = PublicationGate()
    blocked = gate.evaluate(_passed(), anomaly=_critical_anomaly())
    overridden = gate.override(blocked, "known migration")
    assert overridden.has_critical_anomaly is True


# --- serialization -----------------------------------------------------

def test_decision_to_dict():
    d = PublicationGate().evaluate(_failed())
    as_dict = d.to_dict()
    assert as_dict["allowed"] is False
    assert as_dict["state"] == "quality_failed"
    assert as_dict["requires_human_approval"] is True