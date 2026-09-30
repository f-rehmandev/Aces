"""
Degradation Recovery Ladder — spec §17.

Walks the four rungs in order, stopping at the first that works:

    Rung 1: Deterministic fallback           (cost: ~0)
    Rung 2: Text-only LLM extraction          (cost: low)
    Rung 3: Multimodal / vision healing       (cost: high)
    Rung 4: Graceful failure                  (cost: ~0)

The rung implementations are injected as callables so tests never touch
a browser or an LLM. Each callable receives a `FailureContext` and returns
a list of records. An empty list = that rung failed.

Rung 3 is gated by the `HealingCircuitBreaker` (§17.2.1) to prevent
hallucination loops. Rung 2 is gated too, because it also spends LLM calls.
Rung 1 is free and always attempted.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Optional

from src.healing.types import (
    FailureContext, FailureKind, HealingResult, LadderRung, RungAttempt,
    EXTRACTION_FAILURES, TERMINAL_FAILURES, RUNG_ORDER,
)
from src.healing.circuit import HealingCircuitBreaker


RungFn = Callable[[FailureContext], list[dict]]


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

class HealingLadder:
    """
    Orchestrates the degradation recovery ladder.

    `deterministic_fn`, `text_llm_fn`, `vision_fn` — callables taking a
    FailureContext, returning recovered records (or []). `None` for a
    rung means "that rung is disabled".
    """

    def __init__(
        self,
        deterministic_fn: Optional[RungFn] = None,
        text_llm_fn: Optional[RungFn] = None,
        vision_fn: Optional[RungFn] = None,
        circuit_breaker: Optional[HealingCircuitBreaker] = None,
    ):
        self.deterministic_fn = deterministic_fn
        self.text_llm_fn = text_llm_fn
        self.vision_fn = vision_fn
        self.circuit = circuit_breaker or HealingCircuitBreaker()

    # ------------------------------------------------------------------
    def attempt(self, ctx: FailureContext) -> HealingResult:
        result = HealingResult(url=ctx.url, domain=ctx.domain)

        # --- pre-flight gates ---
        if ctx.failure_kind in TERMINAL_FAILURES:
            result.gave_up_reason = (
                f"terminal failure: {ctx.failure_kind.value}"
            )
            return result

        if ctx.failure_kind not in EXTRACTION_FAILURES:
            result.gave_up_reason = (
                f"not an extraction failure: {ctx.failure_kind.value}"
            )
            return result

        # --- Rung 1: deterministic (free, always attempted) ---
        self._try_rung(
            LadderRung.DETERMINISTIC_FALLBACK,
            self.deterministic_fn,
            ctx, result,
        )
        if result.recovered:
            return result

        # --- Rung 2: text LLM (gated by circuit) ---
        self._try_gated_rung(
            LadderRung.TEXT_LLM, self.text_llm_fn, ctx, result,
        )
        if result.recovered:
            return result

        # --- Rung 3: vision (gated by circuit) ---
        self._try_gated_rung(
            LadderRung.VISION, self.vision_fn, ctx, result,
        )
        if result.recovered:
            return result

        # --- Rung 4: graceful failure ---
        result.attempts.append(RungAttempt(
            rung=LadderRung.GRACEFUL_FAILURE,
            attempted=True,
            succeeded=False,
            reason="all rungs exhausted",
        ))
        result.gave_up_reason = "all rungs exhausted"

        graceful_warning = (
            "Preserving partial data; escalating to human review. "
            f"Partial records: {len(ctx.partial_records)}."
        )
        # Don't clobber a warning that an earlier stage already set
        # (e.g. "circuit tripped") — append instead.
        if result.warning:
            result.warning = f"{result.warning} | {graceful_warning}"
        else:
            result.warning = graceful_warning

        return result

    # ------------------------------------------------------------------
    # Rung helpers
    # ------------------------------------------------------------------
    def _try_rung(
        self,
        rung: LadderRung,
        fn: Optional[RungFn],
        ctx: FailureContext,
        result: HealingResult,
    ) -> None:
        if fn is None:
            result.attempts.append(RungAttempt(
                rung=rung, attempted=False, reason="rung disabled",
            ))
            return

        start = time.monotonic()
        try:
            records = fn(ctx) or []
        except Exception as e:
            result.attempts.append(RungAttempt(
                rung=rung, attempted=True, succeeded=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                reason=f"raised: {type(e).__name__}: {e}",
            ))
            return

        duration_ms = int((time.monotonic() - start) * 1000)

        if records:
            result.attempts.append(RungAttempt(
                rung=rung, attempted=True, succeeded=True,
                records_recovered=len(records),
                duration_ms=duration_ms,
            ))
            result.recovered = True
            result.rung_used = rung
            result.records = list(records)
        else:
            result.attempts.append(RungAttempt(
                rung=rung, attempted=True, succeeded=False,
                duration_ms=duration_ms, reason="no records produced",
            ))

    def _try_gated_rung(
        self,
        rung: LadderRung,
        fn: Optional[RungFn],
        ctx: FailureContext,
        result: HealingResult,
    ) -> None:
        """
        Circuit-breaker-gated LLM rung. If the breaker says stop, the rung
        is recorded as not-attempted and we move on.
        """
        if fn is None:
            result.attempts.append(RungAttempt(
                rung=rung, attempted=False, reason="rung disabled",
            ))
            return

        allowed, reason = self.circuit.can_attempt(ctx.domain)
        if not allowed:
            result.attempts.append(RungAttempt(
                rung=rung, attempted=False,
                reason=f"circuit tripped: {reason}",
            ))
            if not result.warning:
                result.warning = f"healing circuit tripped on {ctx.domain}"
            return

        self.circuit.record_attempt(ctx.domain)
        self._try_rung(rung, fn, ctx, result)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def make_ladder(circuit_breaker: Optional[HealingCircuitBreaker] = None) -> HealingLadder:
    """
    Build a ladder with only the deterministic rung wired by default.
    Callers plug in their own text_llm_fn / vision_fn as needed.
    """
    return HealingLadder(
        deterministic_fn=_noop_deterministic,
        text_llm_fn=None,
        vision_fn=None,
        circuit_breaker=circuit_breaker,
    )


def _noop_deterministic(ctx: FailureContext) -> list[dict]:
    return []


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.healing.types import FailureKind

    def ctx(**kw):
        base = dict(
            url="https://shop.example/x", domain="shop.example",
            failure_kind=FailureKind.SELECTOR, field_name="price",
        )
        base.update(kw)
        return FailureContext(**base)

    # 1. Rung 1 succeeds -> stops immediately, no LLM cost
    called = {"llm": 0, "vision": 0}
    def r1(c): return [{"price": "$5"}]
    def r2(c): called["llm"] += 1; return [{"price": "$5"}]
    def r3(c): called["vision"] += 1; return [{"price": "$5"}]

    lad = HealingLadder(r1, r2, r3)
    res = lad.attempt(ctx())
    assert res.recovered and res.rung_used == LadderRung.DETERMINISTIC_FALLBACK
    assert called["llm"] == 0 and called["vision"] == 0
    assert res.total_cost_estimate == 0.0

    # 2. Rung 1 fails, Rung 2 succeeds
    called = {"llm": 0, "vision": 0}
    lad = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=r2, vision_fn=r3,
    )
    res = lad.attempt(ctx())
    assert res.recovered and res.rung_used == LadderRung.TEXT_LLM
    assert called["llm"] == 1 and called["vision"] == 0

    # 3. Rungs 1 and 2 fail, Rung 3 (vision) succeeds
    called = {"vision": 0}
    lad = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=lambda c: [],
        vision_fn=r3,
    )
    res = lad.attempt(ctx())
    assert res.recovered and res.rung_used == LadderRung.VISION
    assert called["vision"] == 1

    # 4. All rungs fail -> graceful failure (rung 4)
    lad = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=lambda c: [],
        vision_fn=lambda c: [],
    )
    res = lad.attempt(ctx(partial_records=[{"title": "partial"}]))
    assert not res.recovered
    assert res.gave_up_reason == "all rungs exhausted"
    assert any(a.rung == LadderRung.GRACEFUL_FAILURE for a in res.attempts)
    assert "Partial records: 1" in res.warning

    # 5. Terminal failure -> skips all rungs
    for kind in (FailureKind.ACCESS_BLOCK, FailureKind.COMPLIANCE, FailureKind.BUDGET):
        lad = HealingLadder(
            deterministic_fn=lambda c: [{"x": 1}],
            text_llm_fn=lambda c: [{"x": 1}],
            vision_fn=lambda c: [{"x": 1}],
        )
        res = lad.attempt(ctx(failure_kind=kind))
        assert not res.recovered
        assert "terminal" in res.gave_up_reason

    # 6. Non-extraction failure -> skip
    lad = HealingLadder(deterministic_fn=lambda c: [{"x": 1}])
    res = lad.attempt(ctx(failure_kind=FailureKind.NETWORK))
    assert "not an extraction failure" in res.gave_up_reason

    # 7. Circuit breaker kicks in after 3 attempts
    cb = HealingCircuitBreaker(max_per_domain=3)
    lad = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=lambda c: [],
        vision_fn=lambda c: [],
        circuit_breaker=cb,
    )
    for i in range(3):
        lad.attempt(ctx())
    # 4th attempt: circuit is tripped, LLM rungs skipped
    res = lad.attempt(ctx())
    skipped = [a for a in res.attempts if not a.attempted
               and a.rung in (LadderRung.TEXT_LLM, LadderRung.VISION)]
    assert skipped, res.attempts
    assert "circuit tripped" in res.warning

    # 8. Rung that raises is treated as failed
    def r_raises(c): raise RuntimeError("boom")
    lad = HealingLadder(
        deterministic_fn=r_raises,
        text_llm_fn=lambda c: [{"x": 1}],
    )
    res = lad.attempt(ctx())
    assert res.recovered and res.rung_used == LadderRung.TEXT_LLM
    assert any("raised: RuntimeError" in a.reason for a in res.attempts)

    # 9. Disabled rungs are recorded as not-attempted
    lad = HealingLadder(deterministic_fn=None, text_llm_fn=None, vision_fn=None)
    res = lad.attempt(ctx())
    assert all(not a.attempted for a in res.attempts
               if a.rung != LadderRung.GRACEFUL_FAILURE)

    # 10. to_dict round-trip
    res = HealingLadder(
        deterministic_fn=lambda c: [],
        text_llm_fn=lambda c: [{"a": 1}],
    ).attempt(ctx())
    d = res.to_dict()
    assert d["rung_used"] == "rung2_text_llm"
    assert d["recovered"] is True

    print("Healing ladder OK.")