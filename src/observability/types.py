"""
Incident & SLO types — spec §40.6.

Incidents
    An explicit record of "something is wrong right now." Unlike a
    single failed job, an incident is a state — it's opened, it may
    persist, and it's closed when the underlying problem resolves.

SLOs
    Service Level Objectives: measurable targets like "95% of jobs
    complete successfully over a 7-day window." The SLO tracker
    records samples and evaluates compliance; it never takes action
    on its own — the alert engine does that.

Both live under `src/observability/`. The store and trackers are in
sibling modules.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Incident severity / status
# ---------------------------------------------------------------------------

class IncidentSeverity(str, Enum):
    """
    Ordered from least to most severe. Same three-tier scheme used by
    the AlertRule engine so alerts and incidents can be cross-linked.

    `.value` is for storage / wire format. `.rank` is the ONLY correct
    way to compare severities — the string values are not alphabetically
    ordered ("critical" < "info" < "warning"), so `a.value > b.value`
    silently gives the wrong answer.
    """
    INFO = "info"          # noted, no user impact
    WARNING = "warning"    # degraded, but work completes
    CRITICAL = "critical"  # work fails or is unsafe to continue

    @property
    def rank(self) -> int:
        """Numeric ordering: INFO=0, WARNING=1, CRITICAL=2."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict["IncidentSeverity", int] = {
    IncidentSeverity.INFO: 0,
    IncidentSeverity.WARNING: 1,
    IncidentSeverity.CRITICAL: 2,
}


def severity_rank(sev: IncidentSeverity) -> int:
    """Free-function form for callers that only have a raw value."""
    return _SEVERITY_RANK[sev]


class IncidentStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"    # someone has seen it, work in progress
    MITIGATING = "mitigating"        # a fix is being applied
    RESOLVED = "resolved"
    CLOSED = "closed"                # resolved + reviewed


TERMINAL_STATUSES = {IncidentStatus.RESOLVED, IncidentStatus.CLOSED}


def is_terminal_status(status: IncidentStatus) -> bool:
    return status in TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# Primary failure classes
# ---------------------------------------------------------------------------

class FailureClass(str, Enum):
    """
    Broad category of what went wrong. Coarser than the FailureKind
    in the healing ladder — this is what you'd group incidents by in a
    dashboard.
    """
    NETWORK = "network"
    AUTH = "auth"
    COMPLIANCE = "compliance"
    BUDGET = "budget"
    PROVIDER = "provider"
    DATA_QUALITY = "data_quality"
    SCHEMA = "schema"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Detection source
# ---------------------------------------------------------------------------

class DetectionSource(str, Enum):
    ALERT_RULE = "alert_rule"
    SLO_BREACH = "slo_breach"
    CIRCUIT_BREAKER = "circuit_breaker"
    OPERATOR = "operator"
    AUTOMATIC = "automatic"


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    """
    A live or historical incident.

    Deduplication: two incidents with the same `dedup_key` and an open
    status are the same incident — the tracker increments
    `occurrence_count` rather than creating a duplicate.
    """
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dedup_key: str = ""

    title: str = ""
    description: str = ""
    severity: IncidentSeverity = IncidentSeverity.WARNING
    status: IncidentStatus = IncidentStatus.OPEN
    failure_class: FailureClass = FailureClass.UNKNOWN
    detection_source: DetectionSource = DetectionSource.AUTOMATIC

    # Scope
    client_id: str = "default"
    affected_jobs: list[str] = field(default_factory=list)
    affected_domains: list[str] = field(default_factory=list)

    # Response
    acknowledged_by: str = ""
    mitigation: str = ""
    root_cause_note: str = ""

    # Linkage
    linked_alert_ids: list[str] = field(default_factory=list)
    linked_healing_ids: list[str] = field(default_factory=list)
    linked_provider_events: list[str] = field(default_factory=list)

    # Timestamps
    started_at: str = field(default_factory=_utc_now_iso)
    acknowledged_at: str = ""
    resolved_at: str = ""
    closed_at: str = ""
    updated_at: str = field(default_factory=_utc_now_iso)

    # Bookkeeping
    occurrence_count: int = 1
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return not is_terminal_status(self.status)

    def duration_seconds(self) -> Optional[float]:
        start = _parse_iso(self.started_at)
        end = _parse_iso(self.resolved_at or self.closed_at or "")
        if not start:
            return None
        if not end:
            end = datetime.now(timezone.utc)
        return round((end - start).total_seconds(), 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["status"] = self.status.value
        d["failure_class"] = self.failure_class.value
        d["detection_source"] = self.detection_source.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Incident":
        d = dict(data)
        for field_name, enum_cls, default in (
            ("severity", IncidentSeverity, IncidentSeverity.WARNING),
            ("status", IncidentStatus, IncidentStatus.OPEN),
            ("failure_class", FailureClass, FailureClass.UNKNOWN),
            ("detection_source", DetectionSource, DetectionSource.AUTOMATIC),
        ):
            try:
                d[field_name] = enum_cls(d.get(field_name, default.value))
            except (ValueError, TypeError):
                d[field_name] = default
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# SLO
# ---------------------------------------------------------------------------

class SLODirection(str, Enum):
    """Whether higher is better or lower is better."""
    AT_LEAST = "at_least"    # "success rate >= 0.95"
    AT_MOST = "at_most"      # "queue start latency <= 30s"


@dataclass
class SLO:
    """
    A service-level objective.

    `target` is a numeric threshold in `unit`'s natural scale:
        - success_rate: 0..1
        - latency_seconds: seconds
        - error_rate: 0..1
        - throughput_per_min: count

    `window_days` is the rolling window used to evaluate compliance.
    `min_samples` protects against a single bad sample making the SLO
    look breached — the tracker returns "insufficient_data" until
    enough samples exist.
    """
    slo_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    metric: str = ""              # "job_success_rate", "queue_start_latency"
    unit: str = "ratio"           # "ratio" | "seconds" | "count"
    direction: SLODirection = SLODirection.AT_LEAST
    target: float = 0.95
    window_days: int = 7
    min_samples: int = 30
    scope: str = "client"         # "client" | "system" | "domain"
    scope_id: str = ""
    enabled: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SLO":
        d = dict(data)
        try:
            d["direction"] = SLODirection(d.get("direction", "at_least"))
        except ValueError:
            d["direction"] = SLODirection.AT_LEAST
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# SLO sample + evaluation
# ---------------------------------------------------------------------------

@dataclass
class SLOSample:
    """
    One observation for an SLO. `value` is measured in the SLO's
    `unit`. `sampled_at` defaults to now.
    """
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    slo_id: str = ""
    client_id: str = "default"
    value: float = 0.0
    sampled_at: str = field(default_factory=_utc_now_iso)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SLOSample":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


class SLOStatus(str, Enum):
    MEETING = "meeting"
    AT_RISK = "at_risk"              # within 10% of the target
    BREACHED = "breached"
    INSUFFICIENT_DATA = "insufficient_data"
    DISABLED = "disabled"


@dataclass
class SLOEvaluation:
    """Result of evaluating one SLO over its window."""
    slo_id: str
    slo_name: str = ""
    status: SLOStatus = SLOStatus.INSUFFICIENT_DATA
    current_value: Optional[float] = None
    target: float = 0.0
    direction: str = SLODirection.AT_LEAST.value
    sample_count: int = 0
    window_start: str = ""
    window_end: str = ""
    margin: Optional[float] = None       # distance from target, positive = good
    reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Incident defaults
    inc = Incident(title="job failures spiking")
    assert inc.status == IncidentStatus.OPEN
    assert inc.severity == IncidentSeverity.WARNING
    assert inc.is_open
    assert inc.occurrence_count == 1

    # Incident round-trip
    inc2 = Incident(
        dedup_key="k1",
        title="provider down",
        severity=IncidentSeverity.CRITICAL,
        status=IncidentStatus.MITIGATING,
        failure_class=FailureClass.PROVIDER,
        detection_source=DetectionSource.ALERT_RULE,
        client_id="acme",
        affected_jobs=["j-1", "j-2"],
        affected_domains=["shop.example"],
    )
    d = inc2.to_dict()
    assert d["severity"] == "critical"
    assert d["status"] == "mitigating"
    assert d["failure_class"] == "provider"
    assert d["detection_source"] == "alert_rule"

    inc3 = Incident.from_dict(d)
    assert inc3.severity == IncidentSeverity.CRITICAL
    assert inc3.affected_jobs == ["j-1", "j-2"]
    assert inc3.client_id == "acme"

    # Bad enum values fall back
    inc4 = Incident.from_dict({
        "severity": "bogus",
        "status": "nope",
        "failure_class": "???",
        "detection_source": "whatever",
    })
    assert inc4.severity == IncidentSeverity.WARNING
    assert inc4.status == IncidentStatus.OPEN
    assert inc4.failure_class == FailureClass.UNKNOWN
    assert inc4.detection_source == DetectionSource.AUTOMATIC

    # Terminal status
    assert is_terminal_status(IncidentStatus.RESOLVED)
    assert is_terminal_status(IncidentStatus.CLOSED)
    assert not is_terminal_status(IncidentStatus.OPEN)

    # Duration
    inc5 = Incident(
        started_at="2026-01-01T00:00:00+00:00",
        resolved_at="2026-01-01T00:05:00+00:00",
    )
    assert inc5.duration_seconds() == 300.0

    # SLO defaults
    slo = SLO(name="success rate", metric="job_success_rate", target=0.95)
    assert slo.direction == SLODirection.AT_LEAST
    assert slo.window_days == 7
    assert slo.min_samples == 30
    assert slo.enabled

    # SLO round-trip
    slo2 = SLO(
        name="queue latency", metric="queue_start_latency",
        unit="seconds", direction=SLODirection.AT_MOST,
        target=30.0, window_days=1,
    )
    d = slo2.to_dict()
    assert d["direction"] == "at_most"
    slo3 = SLO.from_dict(d)
    assert slo3.direction == SLODirection.AT_MOST
    assert slo3.target == 30.0

    # Sample round-trip
    smp = SLOSample(slo_id=slo.slo_id, value=0.92, client_id="acme")
    d = smp.to_dict()
    smp2 = SLOSample.from_dict(d)
    assert smp2.value == 0.92

    # Evaluation shape
    ev = SLOEvaluation(
        slo_id=slo.slo_id, slo_name=slo.name,
        status=SLOStatus.MEETING, current_value=0.97,
        target=0.95, sample_count=100,
        margin=0.02,
    )
    d = ev.to_dict()
    assert d["status"] == "meeting"
    assert d["margin"] == 0.02

    # Status enum
    assert SLOStatus.INSUFFICIENT_DATA.value == "insufficient_data"

    print("Observability types OK.")