"""
Budget + circuit breaker — spec §38.

Every job has a budget. When a hard limit is hit, the job pauses gracefully
(never crashes), persists partial data, and surfaces a "human intervention
required" event. The emergency reserve is protected (§15.8).

Also handles the ScraperAPI credit tracking from §15.8.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    max_llm_tokens: int = 100_000
    max_usd: float = 0.50
    max_scraperapi_credits: int = 0
    max_pages: int = 5_000
    max_wall_clock_seconds: int = 3_600
    soft_limit_fraction: float = 0.80

    # live counters
    llm_tokens_used: int = 0
    usd_used: float = 0.0
    scraperapi_credits_used: int = 0
    pages_used: int = 0
    wall_clock_used: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_wire(cls, wire: "object") -> "Budget":
        """
        Build a runtime Budget from a wire-format `TaskSpec.budget`
        (the dataclass in `src.core.task_spec`).

        Both classes carry the same four limit fields. This method
        exists so callers don't have to know that both Budgets exist,
        and so the mapping is defined in exactly one place — currently
        `PipelineRunner._resolve_budget_tracker()`. Future callers
        should route through here instead.

        Wall-clock and soft-limit fields fall back to their defaults
        because the wire format deliberately doesn't carry them.
        """
        return cls(
            max_llm_tokens=int(getattr(wire, "max_llm_tokens", 100_000)),
            max_usd=float(getattr(wire, "max_usd", 0.50)),
            max_scraperapi_credits=int(
                getattr(wire, "max_scraperapi_credits", 0)
            ),
            max_pages=int(getattr(wire, "max_pages", 5_000)),
        )


# ---------------------------------------------------------------------------
# Breaker
# ---------------------------------------------------------------------------

@dataclass
class BreakerState:
    tripped: bool = False
    reason: str = ""
    soft_warning: bool = False
    soft_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class BudgetTracker:
    """
    Tracks usage against a Budget and reports when the soft or hard limit
    has been reached. The tracker itself does not perform I/O.
    """

    def __init__(self, budget: Optional[Budget] = None):
        self.budget = budget or Budget()

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------
    def consume(
        self,
        llm_tokens: int = 0,
        usd: float = 0.0,
        scraperapi_credits: int = 0,
        pages: int = 0,
        wall_clock_seconds: float = 0.0,
    ) -> None:
        if any(x < 0 for x in (llm_tokens, usd, scraperapi_credits, pages, wall_clock_seconds)):
            raise ValueError("budget consumption values must be >= 0")
        self.budget.llm_tokens_used += llm_tokens
        self.budget.usd_used += usd
        self.budget.scraperapi_credits_used += scraperapi_credits
        self.budget.pages_used += pages
        self.budget.wall_clock_used += wall_clock_seconds

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def check(self) -> BreakerState:
        """Return the current breaker state. Called before each unit of work."""
        state = BreakerState()
        b = self.budget

        # Hard limits
        if b.llm_tokens_used >= b.max_llm_tokens:
            state.tripped = True
            state.reason = "max_llm_tokens"
            return state
        if b.usd_used >= b.max_usd:
            state.tripped = True
            state.reason = "max_usd"
            return state
        if b.max_scraperapi_credits > 0 and b.scraperapi_credits_used >= b.max_scraperapi_credits:
            state.tripped = True
            state.reason = "max_scraperapi_credits"
            return state
        if b.pages_used >= b.max_pages:
            state.tripped = True
            state.reason = "max_pages"
            return state
        if b.wall_clock_used >= b.max_wall_clock_seconds:
            state.tripped = True
            state.reason = "max_wall_clock_seconds"
            return state

        # Soft limits — warnings only
        if self._pct(b.llm_tokens_used, b.max_llm_tokens) >= b.soft_limit_fraction:
            state.soft_warning = True
            state.soft_reasons.append("llm_tokens")
        if self._pct(b.usd_used, b.max_usd) >= b.soft_limit_fraction:
            state.soft_warning = True
            state.soft_reasons.append("usd")
        if b.max_scraperapi_credits > 0 and \
                self._pct(b.scraperapi_credits_used, b.max_scraperapi_credits) >= b.soft_limit_fraction:
            state.soft_warning = True
            state.soft_reasons.append("scraperapi_credits")
        if self._pct(b.pages_used, b.max_pages) >= b.soft_limit_fraction:
            state.soft_warning = True
            state.soft_reasons.append("pages")

        return state

    def snapshot(self) -> dict:
        b = self.budget
        return {
            "llm_tokens": {"used": b.llm_tokens_used, "max": b.max_llm_tokens,
                           "pct": self._pct(b.llm_tokens_used, b.max_llm_tokens)},
            "usd": {"used": round(b.usd_used, 4), "max": b.max_usd,
                    "pct": self._pct(b.usd_used, b.max_usd)},
            "scraperapi_credits": {"used": b.scraperapi_credits_used,
                                    "max": b.max_scraperapi_credits,
                                    "pct": self._pct(b.scraperapi_credits_used,
                                                     b.max_scraperapi_credits)},
            "pages": {"used": b.pages_used, "max": b.max_pages,
                      "pct": self._pct(b.pages_used, b.max_pages)},
            "wall_clock_seconds": {"used": round(b.wall_clock_used, 2),
                                    "max": b.max_wall_clock_seconds,
                                    "pct": self._pct(b.wall_clock_used,
                                                     b.max_wall_clock_seconds)},
        }

    @staticmethod
    def _pct(used, limit) -> float:
        if not limit or limit <= 0:
            return 0.0
        return round(min(1.0, used / limit), 4)


# ---------------------------------------------------------------------------
# ScraperAPI credit budget (§15.8)
# ---------------------------------------------------------------------------

@dataclass
class ScraperAPIBudget:
    daily_limit: int = 0
    monthly_limit: int = 0
    per_task_limit: int = 0
    per_domain_limit: int = 0
    emergency_reserve: int = 0
    actual_consumed_today: int = 0
    actual_consumed_month: int = 0

    def remaining(self) -> int:
        if self.monthly_limit <= 0:
            return 0
        used = self.actual_consumed_month
        limit = self.monthly_limit - self.emergency_reserve
        return max(0, limit - used)


class ScraperAPIBudgetTracker:
    def __init__(self, budget: ScraperAPIBudget):
        self.budget = budget

    def can_use(self, estimated_cost: int = 1, allow_reserve: bool = False) -> bool:
        if self.budget.monthly_limit <= 0:
            return False
        protected = 0 if allow_reserve else self.budget.emergency_reserve
        available = self.budget.monthly_limit - protected - self.budget.actual_consumed_month
        return available >= estimated_cost

    def consume(self, actual: int = 1) -> None:
        if actual < 0:
            raise ValueError("actual must be >= 0")
        self.budget.actual_consumed_today += actual
        self.budget.actual_consumed_month += actual

    def snapshot(self) -> dict:
        b = self.budget
        return {
            "daily_limit": b.daily_limit,
            "monthly_limit": b.monthly_limit,
            "per_task_limit": b.per_task_limit,
            "emergency_reserve": b.emergency_reserve,
            "used_today": b.actual_consumed_today,
            "used_month": b.actual_consumed_month,
            "remaining_usable": b.remaining(),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- budget under limits ---
    b = Budget(max_llm_tokens=1000, max_usd=1.0, max_pages=10)
    t = BudgetTracker(b)
    t.consume(llm_tokens=100, usd=0.1, pages=2)
    s = t.check()
    assert not s.tripped
    assert not s.soft_warning

    # --- soft warning ---
    t.consume(llm_tokens=800)   # total 900 of 1000 → 90%
    s = t.check()
    assert s.soft_warning
    assert "llm_tokens" in s.soft_reasons

    # --- hard trip: tokens ---
    t.consume(llm_tokens=200)   # total 1100 → over
    s = t.check()
    assert s.tripped
    assert s.reason == "max_llm_tokens"

    # --- usd trip ---
    t2 = BudgetTracker(Budget(max_usd=1.0))
    t2.consume(usd=1.5)
    assert t2.check().reason == "max_usd"

    # --- pages trip ---
    t3 = BudgetTracker(Budget(max_pages=3))
    t3.consume(pages=3)
    assert t3.check().reason == "max_pages"

    # --- wall-clock trip ---
    t4 = BudgetTracker(Budget(max_wall_clock_seconds=10))
    t4.consume(wall_clock_seconds=15)
    assert t4.check().reason == "max_wall_clock_seconds"

    # --- scraperapi credits trip ---
    t5 = BudgetTracker(Budget(max_scraperapi_credits=10))
    t5.consume(scraperapi_credits=10)
    assert t5.check().reason == "max_scraperapi_credits"

    # --- snapshot ---
    snap = t.snapshot()
    assert snap["llm_tokens"]["used"] == 1100
    assert snap["pages"]["pct"] == 0.2

    # --- negative consumption rejected ---
    try:
        t.consume(llm_tokens=-1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # --- scraperapi budget ---
    sb = ScraperAPIBudget(monthly_limit=5000, emergency_reserve=500)
    strack = ScraperAPIBudgetTracker(sb)
    assert strack.can_use(100)
    strack.consume(4000)
    # available = 5000 - 500 - 4000 = 500
    assert strack.can_use(500)
    assert not strack.can_use(600)
    # allow_reserve ignores reserve
    assert strack.can_use(900, allow_reserve=True)
    assert not strack.can_use(1100, allow_reserve=True)

    print("Budget + circuit breaker OK.")