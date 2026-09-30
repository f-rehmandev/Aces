"""
Evolution engine types — spec §19.

Self-evolution is *not* self-modifying source code (§19). It is the
mechanism by which ACES proposes, tests, and promotes **configuration-level**
improvements to strategies.

Core data flow:
    Experience         — one run's metrics, fed to the engine
    CandidateStrategy  — a proposed config change
    SandboxResult      — outcome of testing a candidate in isolation
    EvolutionResult    — final decision (promoted / rejected / rolled back)
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Experience — one run's observation
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """
    The engine's input. One per run.

    Keep the numeric fields explicit and comparable — every promotion gate
    reads them (§19.3).
    """
    run_id: str
    task_id: str
    domain: str
    strategy_id: str = ""
    # outcome metrics
    records_extracted: int = 0
    field_completeness: float = 0.0        # 0..1
    schema_validation_pass_rate: float = 1.0
    duplicate_rate: float = 0.0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    failure_class: str = ""                # e.g. "selector", "timeout"
    compliance_flags: list[str] = field(default_factory=list)

    def cost_per_1k_records(self) -> Optional[float]:
        if self.records_extracted <= 0:
            return None
        return (self.cost_usd / self.records_extracted) * 1000

    def latency_per_record(self) -> Optional[float]:
        if self.records_extracted <= 0:
            return None
        return self.latency_seconds / self.records_extracted

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Candidate strategy
# ---------------------------------------------------------------------------

class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    IN_SANDBOX = "in_sandbox"
    REJECTED = "rejected"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


@dataclass
class CandidateStrategy:
    """
    A proposed strategy change. The `configuration` payload is opaque to
    the engine — it just gets passed to the sandbox runner and, on success,
    to the promoter. Real strategies live in src.strategy.strategy_memory.
    """
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain: str = ""
    baseline_strategy_id: str = ""
    configuration: dict = field(default_factory=dict)
    rationale: str = ""
    status: CandidateStatus = CandidateStatus.PROPOSED
    created_at: float = 0.0
    # evidence accumulated during sandbox
    sandbox_results: list["SandboxResult"] = field(default_factory=list)
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["sandbox_results"] = [r.to_dict() for r in self.sandbox_results]
        return d


# ---------------------------------------------------------------------------
# Sandbox result
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """
    One test run of a candidate strategy against a held-out sample.
    """
    records_extracted: int = 0
    field_completeness: float = 0.0
    schema_validation_pass_rate: float = 1.0
    duplicate_rate: float = 0.0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    compliance_flags: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evolution decision
# ---------------------------------------------------------------------------

class EvolutionOutcome(str, Enum):
    PROMOTED = "promoted"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class EvolutionResult:
    candidate_id: str
    outcome: EvolutionOutcome
    reason: str = ""
    baseline_mean: Optional[float] = None       # baseline quality proxy
    candidate_mean: Optional[float] = None
    runs: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


# ---------------------------------------------------------------------------
# Promotion gate thresholds — §19.3
# ---------------------------------------------------------------------------

@dataclass
class PromotionGate:
    min_records: int = 20
    min_field_completeness_delta: float = -0.02     # baseline - 2%
    min_schema_pass_rate: float = 1.0
    max_duplicate_rate_delta: float = 0.01          # +1%
    max_cost_ratio: float = 1.20
    max_latency_ratio: float = 1.30
    require_no_compliance_flags: bool = True
    min_reproducible_runs: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Rollback trigger — §19.4
# ---------------------------------------------------------------------------

@dataclass
class RollbackTrigger:
    success_rate_drop_pct: float = 15.0     # > 15% drop vs 7-day baseline
    quality_score_floor: float = 0.5
    cooldown_days: int = 7

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exp = Experience(
        run_id="r1", task_id="t1", domain="shop.example",
        strategy_id="s1", records_extracted=100, field_completeness=0.9,
        cost_usd=0.05, latency_seconds=30,
    )
    assert exp.cost_per_1k_records() == 0.5
    assert exp.latency_per_record() == 0.3

    # Zero records edge case
    exp0 = Experience(run_id="r0", task_id="t1", domain="x", records_extracted=0)
    assert exp0.cost_per_1k_records() is None
    assert exp0.latency_per_record() is None

    # Candidate serialization
    c = CandidateStrategy(domain="shop.example", configuration={"timeout": 5000})
    d = c.to_dict()
    assert d["status"] == "proposed"
    assert d["configuration"]["timeout"] == 5000

    # Promotion gate defaults
    g = PromotionGate()
    assert g.min_records == 20
    assert g.max_cost_ratio == 1.20

    # Rollback defaults
    rb = RollbackTrigger()
    assert rb.cooldown_days == 7

    # Evolution result enum
    r = EvolutionResult(candidate_id="c1", outcome=EvolutionOutcome.PROMOTED)
    assert r.to_dict()["outcome"] == "promoted"

    print("Evolution types OK.")