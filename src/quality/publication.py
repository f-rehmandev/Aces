"""
Publication gate + quarantine — spec §22.5, §22.6.

Lifecycle states for a dataset:

    DRAFT ──▶ VALIDATED ──▶ READY_TO_PUBLISH ──▶ PUBLISHED
                   │
                   └──▶ QUALITY_FAILED ──▶ QUARANTINED

The gate refuses to overwrite a previously-published dataset with a
failing one. The operator must explicitly approve override (logged).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.quality.rules import QualityResult
from src.quality.anomaly import AnomalyReport


class DatasetState(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    READY_TO_PUBLISH = "ready_to_publish"
    PUBLISHED = "published"
    QUALITY_FAILED = "quality_failed"
    QUARANTINED = "quarantined"


@dataclass
class PublicationDecision:
    state: DatasetState
    allowed: bool
    reason: str = ""
    requires_human_approval: bool = False
    has_critical_anomaly: bool = False

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "requires_human_approval": self.requires_human_approval,
            "has_critical_anomaly": self.has_critical_anomaly,
        }


class PublicationGate:
    """
    Decide whether a dataset can be published.

    `block_on_critical_anomaly` — if True (default), any critical anomaly
    forces QUARANTINED even if quality rules passed. Set False for tasks
    that legitimately tolerate large distribution shifts.
    """

    def __init__(self, block_on_critical_anomaly: bool = True):
        self.block_on_critical_anomaly = block_on_critical_anomaly

    def evaluate(
        self,
        quality: QualityResult,
        anomaly: Optional[AnomalyReport] = None,
        previous_dataset_exists: bool = False,
    ) -> PublicationDecision:
        # 1. Hard fail — quality rules
        if not quality.passed:
            return PublicationDecision(
                state=DatasetState.QUALITY_FAILED,
                allowed=False,
                reason=(
                    f"quality gate failed: {quality.failed_rules}. "
                    f"{quality.explanation}"
                ),
                requires_human_approval=True,
            )

        # 2. Critical anomaly — quarantine even when quality passes
        if anomaly is not None and anomaly.has_critical and self.block_on_critical_anomaly:
            return PublicationDecision(
                state=DatasetState.QUARANTINED,
                allowed=False,
                reason=f"critical anomaly: {anomaly.summary()}",
                requires_human_approval=True,
                has_critical_anomaly=True,
            )

        # 3. Quality passed; warn if anomalies exist but not critical
        if anomaly is not None and anomaly.is_suspicious:
            return PublicationDecision(
                state=DatasetState.READY_TO_PUBLISH,
                allowed=True,
                reason=f"published with non-critical anomaly: {anomaly.summary()}",
            )

        # 4. Clean
        return PublicationDecision(
            state=DatasetState.READY_TO_PUBLISH,
            allowed=True,
            reason="all gates passed",
        )

    def override(
        self,
        decision: PublicationDecision,
        justification: str,
    ) -> PublicationDecision:
        """
        Human approval path (§22.6). Caller must supply a non-empty
        justification which is meant to be logged as an audit event.
        """
        if not justification or not justification.strip():
            raise ValueError("override requires a non-empty justification")
        return PublicationDecision(
            state=DatasetState.READY_TO_PUBLISH,
            allowed=True,
            reason=f"human override: {justification}",
            requires_human_approval=False,
            has_critical_anomaly=decision.has_critical_anomaly,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.quality.rules import QualityResult

    gate = PublicationGate()

    passed = QualityResult(passed=True, score=1.0)
    failed = QualityResult(
        passed=False, score=0.3,
        failed_rules=["min_records"], explanation="min_records: 0 < 5",
    )

    # 1. Clean pass
    d = gate.evaluate(passed)
    assert d.allowed and d.state == DatasetState.READY_TO_PUBLISH

    # 2. Failed quality -> blocked, needs approval
    d = gate.evaluate(failed)
    assert not d.allowed
    assert d.state == DatasetState.QUALITY_FAILED
    assert d.requires_human_approval

    # 3. Critical anomaly quarantines even when quality passed
    from src.quality.anomaly import AnomalyFinding, AnomalyReport
    critical = AnomalyReport(findings=[
        AnomalyFinding(kind="sudden_zero", field_name="email",
                       detail="field went empty", severity="critical"),
    ])
    d = gate.evaluate(passed, anomaly=critical)
    assert not d.allowed
    assert d.state == DatasetState.QUARANTINED
    assert d.has_critical_anomaly

    # 4. Non-critical anomaly -> allowed with warning
    warn = AnomalyReport(findings=[
        AnomalyFinding(kind="distribution_shift", field_name="cat",
                       detail="shifted", severity="warning"),
    ])
    d = gate.evaluate(passed, anomaly=warn)
    assert d.allowed
    assert "non-critical" in d.reason

    # 5. Human override
    d = gate.evaluate(failed)
    o = gate.override(d, justification="verified manually with operator")
    assert o.allowed
    assert "verified" in o.reason

    # 6. Override without justification rejected
    try:
        gate.override(d, justification="")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # 7. block_on_critical_anomaly=False lets critical through
    lenient = PublicationGate(block_on_critical_anomaly=False)
    d = lenient.evaluate(passed, anomaly=critical)
    assert d.allowed

    print("Publication gate OK.")