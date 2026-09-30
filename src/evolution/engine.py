"""
Evolution engine — spec §19.

Orchestrates the evolution cycle:

    Observe       collect experiences
    Analyze       cluster failures, find patterns
    Generate      propose candidate strategies
    Sandbox       test in isolation against held-out samples
    Compare       vs baseline on the metrics in §19.3
    Promote       OR rollback

Nothing here touches real scrapers or real strategies — the sandbox runner
and the promoter/rollback callables are injected. That keeps the whole
engine testable without a browser, an LLM, or a database.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Optional

from src.evolution.types import (
    CandidateStatus, CandidateStrategy, EvolutionOutcome, EvolutionResult,
    Experience, PromotionGate, RollbackTrigger, SandboxResult,
)


# ---------------------------------------------------------------------------
# Injected collaborators
# ---------------------------------------------------------------------------

# Takes a candidate configuration dict + the domain, returns a SandboxResult.
SandboxRunner = Callable[[CandidateStrategy], SandboxResult]

# Called on promotion — should persist the new strategy, return its id.
Promoter = Callable[[CandidateStrategy], str]

# Called on rollback — restores the baseline for a domain.
Rollbacker = Callable[[str, str], None]   # (domain, reason) -> None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EvolutionEngine:
    def __init__(
        self,
        sandbox_runner: SandboxRunner,
        promoter: Promoter,
        rollbacker: Optional[Rollbacker] = None,
        gate: Optional[PromotionGate] = None,
        rollback_trigger: Optional[RollbackTrigger] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.sandbox_runner = sandbox_runner
        self.promoter = promoter
        self.rollbacker = rollbacker
        self.gate = gate or PromotionGate()
        self.rollback_trigger = rollback_trigger or RollbackTrigger()
        self.clock = clock

        # Memory
        self._experiences: list[Experience] = []
        self._candidates: list[CandidateStrategy] = []
        self._promoted_strategies: dict[str, str] = {}   # domain -> strategy_id
        self._rollback_history: dict[str, float] = {}    # domain -> last rollback ts

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------
    def record_experience(self, exp: Experience) -> None:
        self._experiences.append(exp)

    def experiences(self) -> list[Experience]:
        return list(self._experiences)

    def experiences_for_domain(self, domain: str) -> list[Experience]:
        return [e for e in self._experiences if e.domain == domain]

    # ------------------------------------------------------------------
    # Analyze — cluster by failure class
    # ------------------------------------------------------------------
    def failure_clusters(self, domain: str) -> dict[str, int]:
        """Map failure_class -> count for a domain."""
        counts: dict[str, int] = {}
        for e in self.experiences_for_domain(domain):
            if not e.failure_class:
                continue
            counts[e.failure_class] = counts.get(e.failure_class, 0) + 1
        return counts

    def propose_candidates(
        self,
        domain: str,
        candidate_builders: dict[str, Callable[[], dict]],
    ) -> list[CandidateStrategy]:
        """
        Given a mapping of failure_class -> builder(returns config dict),
        propose one candidate per failure class that actually appeared in
        this domain's experience.

        This is the *Generate* step (§19.2). The actual candidate content
        is decided by the caller — the engine just wires failures to
        proposals.
        """
        clusters = self.failure_clusters(domain)
        proposals: list[CandidateStrategy] = []
        for failure_class, builder in candidate_builders.items():
            if failure_class not in clusters:
                continue
            config = builder()
            c = CandidateStrategy(
                domain=domain,
                baseline_strategy_id=self._baseline_for(domain),
                configuration=config,
                rationale=f"recovering {clusters[failure_class]} failures of class {failure_class!r}",
                created_at=self.clock(),
            )
            self._candidates.append(c)
            proposals.append(c)
        return proposals

    # ------------------------------------------------------------------
    # Sandbox + compare
    # ------------------------------------------------------------------
    def evaluate(
        self,
        candidate: CandidateStrategy,
        baseline_runs: int,
    ) -> EvolutionResult:
        """
        Run the candidate N times in the sandbox, compare against the
        promotion gate (§19.3), and return the decision.
        """
        # Take baseline metrics from the domain's own history
        domain_exps = self.experiences_for_domain(candidate.domain)
        baseline = self._baseline_snapshot(domain_exps)

        # Run the sandbox
        candidate.sandbox_results = []
        candidate.status = CandidateStatus.IN_SANDBOX
        for _ in range(baseline_runs):
            try:
                result = self.sandbox_runner(candidate)
            except Exception as e:
                return EvolutionResult(
                    candidate_id=candidate.candidate_id,
                    outcome=EvolutionOutcome.REJECTED,
                    reason=f"sandbox raised: {type(e).__name__}: {e}",
                    runs=0,
                )
            if result.error:
                return EvolutionResult(
                    candidate_id=candidate.candidate_id,
                    outcome=EvolutionOutcome.REJECTED,
                    reason=f"sandbox error: {result.error}",
                    runs=len(candidate.sandbox_results),
                )
            candidate.sandbox_results.append(result)

        # Aggregate
        cand_summary = self._aggregate(candidate.sandbox_results)

        # Promotion gate checks (§19.3)
        ok, reason = self._passes_gate(baseline, cand_summary, len(candidate.sandbox_results))
        if not ok:
            candidate.status = CandidateStatus.REJECTED
            candidate.rejection_reason = reason
            return EvolutionResult(
                candidate_id=candidate.candidate_id,
                outcome=EvolutionOutcome.REJECTED,
                reason=reason,
                baseline_mean=baseline.get("field_completeness"),
                candidate_mean=cand_summary.get("field_completeness"),
                runs=len(candidate.sandbox_results),
            )

        # Promote
        new_id = self.promoter(candidate)
        candidate.status = CandidateStatus.PROMOTED
        self._promoted_strategies[candidate.domain] = new_id
        return EvolutionResult(
            candidate_id=candidate.candidate_id,
            outcome=EvolutionOutcome.PROMOTED,
            reason="passed promotion gate",
            baseline_mean=baseline.get("field_completeness"),
            candidate_mean=cand_summary.get("field_completeness"),
            runs=len(candidate.sandbox_results),
        )

    # ------------------------------------------------------------------
    # Rollback (§19.4)
    # ------------------------------------------------------------------
    def check_rollback(
        self,
        domain: str,
        current_success_rate: float,
        seven_day_baseline_success_rate: float,
        current_quality_score: float,
    ) -> Optional[str]:
        """
        Return a rollback reason if the promoted strategy has regressed, else None.
        """
        # Cooldown: don't roll back twice within the trigger window.
        # Only applies if a rollback has actually happened for this domain.
        last = self._rollback_history.get(domain)
        if last is not None and \
                (self.clock() - last) < self.rollback_trigger.cooldown_days * 86400:
            return None

        # Success-rate regression
        if seven_day_baseline_success_rate > 0:
            drop_pct = (
                (seven_day_baseline_success_rate - current_success_rate)
                / seven_day_baseline_success_rate
            ) * 100
            if drop_pct > self.rollback_trigger.success_rate_drop_pct:
                reason = (
                    f"success rate dropped {drop_pct:.1f}% "
                    f"(threshold {self.rollback_trigger.success_rate_drop_pct}%)"
                )
                self._do_rollback(domain, reason)
                return reason

        # Quality floor
        if current_quality_score < self.rollback_trigger.quality_score_floor:
            reason = (
                f"quality score {current_quality_score:.2f} below floor "
                f"{self.rollback_trigger.quality_score_floor}"
            )
            self._do_rollback(domain, reason)
            return reason

        return None

    def _do_rollback(self, domain: str, reason: str) -> None:
        if self.rollbacker is not None:
            self.rollbacker(domain, reason)
        self._rollback_history[domain] = self.clock()
        for c in self._candidates:
            if c.domain == domain and c.status == CandidateStatus.PROMOTED:
                c.status = CandidateStatus.ROLLED_BACK

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def promoted_strategy_for(self, domain: str) -> Optional[str]:
        return self._promoted_strategies.get(domain)

    def candidates(self) -> list[CandidateStrategy]:
        return list(self._candidates)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _baseline_for(self, domain: str) -> str:
        # The baseline is whatever was promoted before (or "" if none).
        return self._promoted_strategies.get(domain, "")

    @staticmethod
    def _baseline_snapshot(experiences: list[Experience]) -> dict:
        if not experiences:
            return {}
        return {
            "records": mean(e.records_extracted for e in experiences),
            "field_completeness": mean(e.field_completeness for e in experiences),
            "schema_pass": mean(e.schema_validation_pass_rate for e in experiences),
            "duplicate_rate": mean(e.duplicate_rate for e in experiences),
            "cost_per_1k": mean(
                (e.cost_per_1k_records() or 0.0) for e in experiences
            ),
            "latency_per_record": mean(
                (e.latency_per_record() or 0.0) for e in experiences
            ),
        }

    @staticmethod
    def _aggregate(results: list[SandboxResult]) -> dict:
        if not results:
            return {}
        return {
            "records": sum(r.records_extracted for r in results),
            "field_completeness": mean(r.field_completeness for r in results),
            "schema_pass": mean(r.schema_validation_pass_rate for r in results),
            "duplicate_rate": mean(r.duplicate_rate for r in results),
            "cost_per_1k": mean(
                (r.cost_usd / r.records_extracted * 1000) if r.records_extracted else 0.0
                for r in results
            ),
            "latency_per_record": mean(
                (r.latency_seconds / r.records_extracted) if r.records_extracted else 0.0
                for r in results
            ),
            "compliance_flags": [
                f for r in results for f in r.compliance_flags
            ],
        }

    def _passes_gate(
        self,
        baseline: dict,
        candidate: dict,
        runs: int,
    ) -> tuple[bool, str]:
        g = self.gate

        # Records per run
        avg_records = candidate.get("records", 0) / max(1, runs)
        if avg_records < g.min_records:
            return False, (
                f"avg records {avg_records:.1f} below gate minimum {g.min_records}"
            )

        # Reproducibility
        if runs < g.min_reproducible_runs:
            return False, (
                f"only {runs} run(s); gate requires {g.min_reproducible_runs}"
            )

        # Schema validation must pass 100%
        if candidate.get("schema_pass", 0.0) < g.min_schema_pass_rate:
            return False, (
                f"schema pass rate {candidate['schema_pass']:.2f} below "
                f"{g.min_schema_pass_rate}"
            )

        # No compliance flags
        if g.require_no_compliance_flags and candidate.get("compliance_flags"):
            return False, (
                f"compliance flags present: {candidate['compliance_flags']}"
            )

        # Only compare against baseline when we have one
        if baseline:
            # Field completeness
            delta = candidate.get("field_completeness", 0.0) - baseline.get(
                "field_completeness", 0.0,
            )
            if delta < g.min_field_completeness_delta:
                return False, (
                    f"field completeness dropped {delta:.3f} "
                    f"(floor {g.min_field_completeness_delta})"
                )

            # Duplicate rate
            dup_delta = candidate.get("duplicate_rate", 0.0) - baseline.get(
                "duplicate_rate", 0.0,
            )
            if dup_delta > g.max_duplicate_rate_delta:
                return False, (
                    f"duplicate rate rose {dup_delta:.3f} "
                    f"(ceiling {g.max_duplicate_rate_delta})"
                )

            # Cost ratio
            b_cost = baseline.get("cost_per_1k", 0.0)
            c_cost = candidate.get("cost_per_1k", 0.0)
            if b_cost > 0 and c_cost / b_cost > g.max_cost_ratio:
                return False, (
                    f"cost ratio {c_cost / b_cost:.2f} exceeds "
                    f"{g.max_cost_ratio}"
                )

            # Latency ratio
            b_lat = baseline.get("latency_per_record", 0.0)
            c_lat = candidate.get("latency_per_record", 0.0)
            if b_lat > 0 and c_lat / b_lat > g.max_latency_ratio:
                return False, (
                    f"latency ratio {c_lat / b_lat:.2f} exceeds "
                    f"{g.max_latency_ratio}"
                )

        return True, "passed"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def ok_sandbox(candidate):
        return SandboxResult(
            records_extracted=50, field_completeness=0.95,
            schema_validation_pass_rate=1.0, duplicate_rate=0.0,
            cost_usd=0.02, latency_seconds=10,
        )

    promoted: list[str] = []
    def promoter(c):
        sid = f"new-{c.domain}-{len(promoted)}"
        promoted.append(sid)
        return sid

    engine = EvolutionEngine(
        sandbox_runner=ok_sandbox, promoter=promoter,
    )

    # Seed baseline experiences so a snapshot exists
    for i in range(3):
        engine.record_experience(Experience(
            run_id=f"r{i}", task_id="t", domain="shop.example",
            records_extracted=50, field_completeness=0.90,
            schema_validation_pass_rate=1.0, duplicate_rate=0.0,
            cost_usd=0.02, latency_seconds=10,
            failure_class="selector" if i == 0 else "",
        ))

    # Propose a candidate based on failure class
    candidates = engine.propose_candidates(
        "shop.example",
        {"selector": lambda: {"fallback_selectors": [".price-v2"]}},
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert "selector" in c.rationale

    # Evaluate
    result = engine.evaluate(c, baseline_runs=3)
    assert result.outcome == EvolutionOutcome.PROMOTED, result.reason
    assert engine.promoted_strategy_for("shop.example") == promoted[0]

    # Gate failure: too few records
    def small_sandbox(candidate):
        return SandboxResult(
            records_extracted=5, field_completeness=0.95,
            schema_validation_pass_rate=1.0,
        )
    engine2 = EvolutionEngine(sandbox_runner=small_sandbox, promoter=lambda c: "x")
    c2 = CandidateStrategy(domain="other.example", configuration={})
    result = engine2.evaluate(c2, baseline_runs=3)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "records" in result.reason

    # Gate failure: schema pass rate
    def schema_fail_sandbox(candidate):
        return SandboxResult(
            records_extracted=50, schema_validation_pass_rate=0.8,
        )
    engine3 = EvolutionEngine(sandbox_runner=schema_fail_sandbox, promoter=lambda c: "x")
    c3 = CandidateStrategy(domain="x.example", configuration={})
    result = engine3.evaluate(c3, baseline_runs=3)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "schema" in result.reason

    # Sandbox exception
    def raises(candidate):
        raise RuntimeError("boom")
    engine4 = EvolutionEngine(sandbox_runner=raises, promoter=lambda c: "x")
    c4 = CandidateStrategy(domain="x.example", configuration={})
    result = engine4.evaluate(c4, baseline_runs=1)
    assert result.outcome == EvolutionOutcome.REJECTED
    assert "boom" in result.reason

    # Rollback: success rate regression
    rolled_back: list[tuple[str, str]] = []
    def rollbacker(domain, reason):
        rolled_back.append((domain, reason))
    engine5 = EvolutionEngine(
        sandbox_runner=ok_sandbox,
        promoter=lambda c: "s",
        rollbacker=rollbacker,
    )
    reason = engine5.check_rollback(
        "shop.example",
        current_success_rate=0.5,
        seven_day_baseline_success_rate=0.95,
        current_quality_score=0.9,
    )
    assert reason is not None
    assert "success rate" in reason
    assert rolled_back

    # Rollback: quality floor
    engine6 = EvolutionEngine(
        sandbox_runner=ok_sandbox, promoter=lambda c: "s",
        rollbacker=lambda d, r: None,
    )
    reason = engine6.check_rollback(
        "shop.example",
        current_success_rate=0.95,
        seven_day_baseline_success_rate=0.95,
        current_quality_score=0.3,
    )
    assert reason is not None
    assert "quality" in reason

    # No rollback: everything healthy
    engine7 = EvolutionEngine(
        sandbox_runner=ok_sandbox, promoter=lambda c: "s",
        rollbacker=lambda d, r: None,
    )
    reason = engine7.check_rollback(
        "shop.example",
        current_success_rate=0.95,
        seven_day_baseline_success_rate=0.95,
        current_quality_score=0.9,
    )
    assert reason is None

    print("Evolution engine OK.")