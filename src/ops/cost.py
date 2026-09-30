"""
Cost intelligence — spec §41.

Tracks LLM tokens (input/output split), per-call model cost, browser time,
ScraperAPI credits, retries, and produces before/after estimates with
cost-per-1k-validated-records as the headline metric.

Model pricing is a table (not hard-coded logic) so it can be updated
without a code change (§92).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Model pricing table (§92 — live-updatable)
# ---------------------------------------------------------------------------

# USD per 1,000 tokens. Values are illustrative; the live table (§92)
# would refresh from provider APIs.
@dataclass
class ModelPricing:
    input_per_1k_usd: float
    output_per_1k_usd: float
    provider: str = ""


DEFAULT_PRICING: dict[str, ModelPricing] = {
    "gemini-1.5-flash":         ModelPricing(0.000_075, 0.000_300, "google"),
    "gemini-1.5-pro":           ModelPricing(0.001_250, 0.005_000, "google"),
    "gpt-4o-mini":              ModelPricing(0.000_150, 0.000_600, "openai"),
    "gpt-4o":                   ModelPricing(0.002_500, 0.010_000, "openai"),
    "openrouter/free":          ModelPricing(0.0, 0.0, "openrouter"),
    "gsk-llama":                ModelPricing(0.0, 0.0, "groq"),
}


# ---------------------------------------------------------------------------
# Cost entry (per call)
# ---------------------------------------------------------------------------

@dataclass
class CostEntry:
    kind: str                      # "llm" | "browser" | "provider" | "retry"
    quantity: float
    unit: str                      # "tokens" | "seconds" | "credits" | "count"
    usd: float = 0.0
    model: str = ""
    provider: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CostTracker:
    def __init__(self, pricing: Optional[dict[str, ModelPricing]] = None):
        self.pricing = dict(DEFAULT_PRICING)
        if pricing:
            self.pricing.update(pricing)
        self.entries: list[CostEntry] = []

    # --- LLM ---
    def record_llm(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEntry:
        pricing = self.pricing.get(model)
        if pricing is None:
            usd = 0.0
            provider = ""
        else:
            usd = (
                input_tokens * pricing.input_per_1k_usd / 1000.0
                + output_tokens * pricing.output_per_1k_usd / 1000.0
            )
            provider = pricing.provider
        entry = CostEntry(
            kind="llm",
            quantity=input_tokens + output_tokens,
            unit="tokens",
            usd=round(usd, 6),
            model=model,
            provider=provider,
        )
        self.entries.append(entry)
        return entry

    # --- Browser ---
    def record_browser(self, seconds: float, usd_per_second: float = 0.0) -> CostEntry:
        entry = CostEntry(
            kind="browser",
            quantity=round(seconds, 3),
            unit="seconds",
            usd=round(seconds * usd_per_second, 6),
        )
        self.entries.append(entry)
        return entry

    # --- External provider (ScraperAPI etc.) ---
    def record_provider(
        self,
        provider: str,
        credits: int,
        usd_per_credit: float = 0.0,
    ) -> CostEntry:
        entry = CostEntry(
            kind="provider",
            quantity=credits,
            unit="credits",
            usd=round(credits * usd_per_credit, 6),
            provider=provider,
        )
        self.entries.append(entry)
        return entry

    # --- Retry ---
    def record_retry(self, count: int = 1, extra_usd: float = 0.0) -> CostEntry:
        entry = CostEntry(
            kind="retry",
            quantity=count,
            unit="count",
            usd=round(extra_usd, 6),
        )
        self.entries.append(entry)
        return entry

    # --- Summary ---
    def total_usd(self) -> float:
        return round(sum(e.usd for e in self.entries), 6)

    def by_kind(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.entries:
            out[e.kind] = round(out.get(e.kind, 0.0) + e.usd, 6)
        return out

    def snapshot(self) -> dict:
        return {
            "total_usd": self.total_usd(),
            "by_kind": self.by_kind(),
            "entries": [asdict(e) for e in self.entries],
        }

    def cost_per_validated_record(self, validated_records: int) -> Optional[float]:
        if validated_records <= 0:
            return None
        return round(self.total_usd() / validated_records, 6)

    def cost_per_1k_validated_records(self, validated_records: int) -> Optional[float]:
        per = self.cost_per_validated_record(validated_records)
        return round(per * 1000, 4) if per is not None else None


# ---------------------------------------------------------------------------
# Pre-run estimate (§41.3 cost guardrails)
# ---------------------------------------------------------------------------

@dataclass
class RunEstimate:
    estimated_llm_usd: float
    estimated_browser_usd: float
    estimated_provider_usd: float

    @property
    def total_usd(self) -> float:
        return round(
            self.estimated_llm_usd
            + self.estimated_browser_usd
            + self.estimated_provider_usd,
            6,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_usd"] = self.total_usd
        return d


def estimate_run(
    pages: int,
    avg_tokens_per_page: int = 2000,
    model: str = "gemini-1.5-flash",
    browser_seconds_per_page: float = 3.0,
    browser_usd_per_second: float = 0.0,
    provider_credits_per_page: int = 0,
    provider_usd_per_credit: float = 0.0,
    pricing: Optional[dict[str, ModelPricing]] = None,
) -> RunEstimate:
    pricing_table = dict(DEFAULT_PRICING)
    if pricing:
        pricing_table.update(pricing)
    p = pricing_table.get(model, ModelPricing(0.0, 0.0))

    # Assume 70/30 input/output split as a rough default.
    in_tokens = int(avg_tokens_per_page * 0.7)
    out_tokens = int(avg_tokens_per_page * 0.3)

    llm_usd = pages * (
        in_tokens * p.input_per_1k_usd / 1000.0
        + out_tokens * p.output_per_1k_usd / 1000.0
    )
    browser_usd = pages * browser_seconds_per_page * browser_usd_per_second
    provider_usd = pages * provider_credits_per_page * provider_usd_per_credit

    return RunEstimate(
        estimated_llm_usd=round(llm_usd, 6),
        estimated_browser_usd=round(browser_usd, 6),
        estimated_provider_usd=round(provider_usd, 6),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t = CostTracker()
    t.record_llm("gemini-1.5-flash", input_tokens=10000, output_tokens=2000)
    t.record_llm("gpt-4o-mini", input_tokens=5000, output_tokens=1000)
    t.record_browser(seconds=30, usd_per_second=0.0)
    t.record_provider("scraperapi", credits=10, usd_per_credit=0.001)
    t.record_retry(count=2)

    total = t.total_usd()
    assert total > 0

    by_kind = t.by_kind()
    assert "llm" in by_kind and "provider" in by_kind

    # Cost per 1k validated
    per = t.cost_per_1k_validated_records(500)
    assert per is not None

    # Unknown model -> 0 cost
    t.record_llm("some-unknown-model", 1000, 1000)
    assert t.entries[-1].usd == 0.0

    # Estimate
    est = estimate_run(pages=100, model="gemini-1.5-flash")
    assert est.total_usd > 0
    assert est.estimated_llm_usd > 0
    assert est.estimated_browser_usd == 0.0

    # snapshot
    snap = t.snapshot()
    assert "total_usd" in snap and "by_kind" in snap

    print(f"Total spent: ${total:.6f}")
    print(f"Cost per 1k validated: ${per}")
    print(f"Estimate for 100 pages: ${est.total_usd}")
    print("Cost tracker OK.")