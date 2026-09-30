"""Unit tests for cost intelligence (spec §41)."""
from src.ops.cost import (
    ModelPricing, CostTracker, CostEntry, RunEstimate, estimate_run,
    DEFAULT_PRICING,
)


# --- pricing ----------------------------------------------------------

def test_llm_recording_uses_default_pricing():
    t = CostTracker()
    entry = t.record_llm("gemini-1.5-flash", input_tokens=10_000, output_tokens=2_000)
    assert entry.usd > 0
    assert entry.model == "gemini-1.5-flash"
    assert entry.kind == "llm"


def test_unknown_model_is_zero_cost():
    t = CostTracker()
    entry = t.record_llm("unknown-model", 1000, 1000)
    assert entry.usd == 0.0


def test_custom_pricing_override():
    custom = {"my-model": ModelPricing(1.0, 2.0, "custom")}
    t = CostTracker(pricing=custom)
    entry = t.record_llm("my-model", input_tokens=1000, output_tokens=1000)
    assert entry.usd == 3.0


# --- totals -----------------------------------------------------------

def test_total_usd_sums_entries():
    t = CostTracker()
    t.record_llm("gemini-1.5-flash", 1000, 500)
    t.record_provider("scraperapi", 10, usd_per_credit=0.001)
    total = t.total_usd()
    assert total > 0


def test_by_kind_groups_correctly():
    t = CostTracker()
    t.record_llm("gemini-1.5-flash", 1000, 500)
    t.record_llm("gemini-1.5-flash", 500, 500)
    t.record_browser(seconds=10, usd_per_second=0.0)
    kinds = t.by_kind()
    assert kinds["llm"] > 0
    assert kinds["browser"] == 0.0


def test_cost_per_1k():
    t = CostTracker()
    t.record_llm("gemini-1.5-flash", 1_000_000, 100_000)
    per_record = t.cost_per_validated_record(1000)
    assert per_record is not None
    per_1k = t.cost_per_1k_validated_records(1000)
    assert per_1k == round(per_record * 1000, 4)


def test_zero_records_returns_none():
    t = CostTracker()
    assert t.cost_per_validated_record(0) is None
    assert t.cost_per_1k_validated_records(0) is None


# --- snapshot ---------------------------------------------------------

def test_snapshot_shape():
    t = CostTracker()
    t.record_llm("gemini-1.5-flash", 1000, 1000)
    s = t.snapshot()
    assert "total_usd" in s
    assert "by_kind" in s
    assert isinstance(s["entries"], list)


# --- estimate ---------------------------------------------------------

def test_estimate_uses_pricing():
    est = estimate_run(pages=10, model="gemini-1.5-flash")
    assert est.estimated_llm_usd > 0
    assert est.total_usd > 0


def test_estimate_browser_cost():
    est = estimate_run(
        pages=10, model="gemini-1.5-flash",
        browser_seconds_per_page=5, browser_usd_per_second=0.001,
    )
    assert est.estimated_browser_usd > 0


def test_estimate_provider_cost():
    est = estimate_run(
        pages=10, model="gemini-1.5-flash",
        provider_credits_per_page=1, provider_usd_per_credit=0.001,
    )
    assert est.estimated_provider_usd > 0


def test_estimate_unknown_model_zero_llm():
    est = estimate_run(pages=10, model="nonexistent")
    assert est.estimated_llm_usd == 0.0


def test_estimate_to_dict():
    est = estimate_run(pages=5)
    d = est.to_dict()
    assert "total_usd" in d
    assert d["total_usd"] >= 0


# --- retries ----------------------------------------------------------

def test_retry_recording():
    t = CostTracker()
    t.record_retry(count=3, extra_usd=0.01)
    kinds = t.by_kind()
    assert kinds["retry"] == 0.01