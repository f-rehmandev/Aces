"""
Self-healing types — spec §17.

The Degradation Recovery Ladder walks an ordered list of recovery rungs,
cheapest first. This module defines the *language* of that walk: what a
failure looks like, which rung recovered, how many rungs were tried.

The ladder orchestrator itself lives in `ladder.py`.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Failure taxonomy (§74)
# ---------------------------------------------------------------------------

class FailureKind(str, Enum):
    NETWORK = "network"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    RATE_LIMIT = "rate_limit"
    ACCESS_BLOCK = "access_block"
    JAVASCRIPT = "javascript"
    SELECTOR = "selector"
    SCHEMA = "schema"
    PARSER = "parser"
    EMPTY = "empty"
    DATA_QUALITY = "data_quality"
    DUPLICATE = "duplicate"
    PROVIDER = "provider"
    BUDGET = "budget"
    AUTHORIZATION = "authorization"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    SYSTEM = "system"
    UNKNOWN = "unknown"


EXTRACTION_FAILURES = {
    FailureKind.SELECTOR,
    FailureKind.SCHEMA,
    FailureKind.PARSER,
    FailureKind.EMPTY,
}

TERMINAL_FAILURES = {
    FailureKind.ACCESS_BLOCK,
    FailureKind.AUTHORIZATION,
    FailureKind.COMPLIANCE,
    FailureKind.SECURITY,
    FailureKind.BUDGET,
}


# ---------------------------------------------------------------------------
# Rungs (§17.1)
# ---------------------------------------------------------------------------

class LadderRung(str, Enum):
    DETERMINISTIC_FALLBACK = "rung1_deterministic"
    TEXT_LLM = "rung2_text_llm"
    VISION = "rung3_vision"
    GRACEFUL_FAILURE = "rung4_graceful"


RUNG_ORDER = [
    LadderRung.DETERMINISTIC_FALLBACK,
    LadderRung.TEXT_LLM,
    LadderRung.VISION,
    LadderRung.GRACEFUL_FAILURE,
]

RUNG_COST = {
    LadderRung.DETERMINISTIC_FALLBACK: 0.0,
    LadderRung.TEXT_LLM: 0.05,
    LadderRung.VISION: 0.30,
    LadderRung.GRACEFUL_FAILURE: 0.0,
}


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------

@dataclass
class FailureContext:
    url: str
    domain: str = ""
    failure_kind: FailureKind = FailureKind.UNKNOWN
    field_name: str = ""
    error_message: str = ""
    html: str = ""
    cleaned_html: str = ""
    previous_html: str = ""
    partial_records: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failure_kind"] = self.failure_kind.value
        for big in ("html", "cleaned_html", "previous_html"):
            if d.get(big):
                d[big] = f"<{len(d[big])} chars>"
        return d


@dataclass
class RungAttempt:
    rung: LadderRung
    attempted: bool = True
    succeeded: bool = False
    records_recovered: int = 0
    duration_ms: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rung"] = self.rung.value
        return d


@dataclass
class HealingResult:
    url: str
    domain: str
    recovered: bool = False
    rung_used: Optional[LadderRung] = None
    records: list[dict] = field(default_factory=list)
    attempts: list[RungAttempt] = field(default_factory=list)
    gave_up_reason: str = ""
    warning: str = ""

    @property
    def total_cost_estimate(self) -> float:
        return round(
            sum(RUNG_COST[a.rung] for a in self.attempts if a.succeeded)
            or sum(RUNG_COST[a.rung] for a in self.attempts),
            4,
        )

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "domain": self.domain,
            "recovered": self.recovered,
            "rung_used": self.rung_used.value if self.rung_used else None,
            "records_count": len(self.records),
            "attempts": [a.to_dict() for a in self.attempts],
            "gave_up_reason": self.gave_up_reason,
            "warning": self.warning,
            "cost_estimate": self.total_cost_estimate,
        }


# ---------------------------------------------------------------------------
# Domain extraction helper
# ---------------------------------------------------------------------------

def domain_of(url: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(url).netloc or "").lower()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ctx = FailureContext(
        url="https://shop.example/category/x",
        domain=domain_of("https://shop.example/category/x"),
        failure_kind=FailureKind.SELECTOR,
        field_name="price",
        error_message="selector .price returned 0 matches",
    )
    assert ctx.domain == "shop.example"

    result = HealingResult(
        url=ctx.url, domain=ctx.domain, recovered=True,
        rung_used=LadderRung.DETERMINISTIC_FALLBACK,
        records=[{"title": "A", "price": "$5"}],
        attempts=[
            RungAttempt(
                rung=LadderRung.DETERMINISTIC_FALLBACK,
                succeeded=True, records_recovered=1, duration_ms=12,
            ),
        ],
    )
    assert result.total_cost_estimate == 0.0
    d = result.to_dict()
    assert d["rung_used"] == "rung1_deterministic"
    assert d["cost_estimate"] == 0.0

    # Default-recovered works
    blank = HealingResult(url="https://x", domain="x")
    assert blank.recovered is False
    assert blank.rung_used is None

    ctx.html = "x" * 5000
    serialized = ctx.to_dict()
    assert "chars" in serialized["html"]

    assert FailureKind.ACCESS_BLOCK in TERMINAL_FAILURES
    assert FailureKind.SELECTOR in EXTRACTION_FAILURES

    print("Healing types OK.")