"""Unit tests for EntitlementEngine (spec §47B)."""
import asyncio

import pytest

from src.usage.entitlements import (
    ClientPlan,
    EntitlementDecision,
    EntitlementEngine,
    OveragePolicy,
    Plan,
    _consumed_for,
    _default_free_plan,
    _first_of_month_iso,
    default_plans,
)
from src.usage.store import InMemoryUsageStore
from src.usage.types import (
    ResourceType,
    UsageEvent,
    UsageSummary,
)


def _run(coro):
    return asyncio.run(coro)


def _mk_engine():
    store = InMemoryUsageStore()
    plans = default_plans()
    return store, plans, EntitlementEngine(store, plans_by_name=plans)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def test_plan_defaults():
    p = Plan(name="x")
    assert p.overage_policy == OveragePolicy.BLOCK
    assert p.limits == {}
    assert p.features == set()


def test_plan_to_dict_serializes_features_set():
    p = Plan(name="x", features={"a", "b"}, overage_policy=OveragePolicy.WARN)
    d = p.to_dict()
    assert d["features"] == ["a", "b"]   # sorted
    assert d["overage_policy"] == "warn"


def test_plan_from_dict_roundtrip():
    p = Plan(
        name="pro",
        limits={"page": 100},
        features={"a", "b"},
        overage_policy=OveragePolicy.ALLOW,
    )
    p2 = Plan.from_dict(p.to_dict())
    assert p2.name == "pro"
    assert p2.limits == {"page": 100}
    assert p2.features == {"a", "b"}
    assert p2.overage_policy == OveragePolicy.ALLOW


def test_plan_from_dict_bad_policy_falls_back_to_block():
    p = Plan.from_dict({"name": "x", "overage_policy": "nonsense"})
    assert p.overage_policy == OveragePolicy.BLOCK


def test_plan_limit_for():
    p = Plan(name="x", limits={"page": 500})
    assert p.limit_for("page") == 500
    assert p.limit_for("token") is None


# ---------------------------------------------------------------------------
# ClientPlan
# ---------------------------------------------------------------------------

def test_client_plan_effective_limits_merge_overrides():
    base = Plan(name="x", limits={"page": 100, "token": 1000})
    cp = ClientPlan(
        client_id="acme",
        plan=base,
        overrides={"limits": {"page": 200}},
    )
    eff = cp.effective_limits()
    assert eff["page"] == 200      # override wins
    assert eff["token"] == 1000    # untouched


def test_client_plan_effective_features_add_and_remove():
    base = Plan(name="x", features={"a", "b", "c"})
    cp = ClientPlan(
        client_id="acme",
        plan=base,
        overrides={
            "features_add": ["d"],
            "features_remove": ["b"],
        },
    )
    assert cp.effective_features() == {"a", "c", "d"}


def test_client_plan_to_dict_shape():
    cp = ClientPlan(client_id="acme", plan=Plan(name="x"))
    d = cp.to_dict()
    assert d["client_id"] == "acme"
    assert d["plan"]["name"] == "x"


# ---------------------------------------------------------------------------
# Engine: plan assignment
# ---------------------------------------------------------------------------

def test_engine_default_plan_for_unassigned_client():
    _, _, engine = _mk_engine()
    assert engine.plan_for("anon").plan.name == "free"


def test_engine_register_and_assign_plan():
    store = InMemoryUsageStore()
    engine = EntitlementEngine(store)
    engine.register_plan(Plan(
        name="custom",
        limits={"page": 42},
        features={"x"},
    ))
    cp = engine.assign_plan("acme", "custom")
    assert cp.plan.name == "custom"
    assert engine.plan_for("acme").limit_for("page") == 42
    assert engine.has_feature("acme", "x")


def test_engine_assign_unknown_plan_raises():
    _, _, engine = _mk_engine()
    with pytest.raises(KeyError):
        engine.assign_plan("acme", "nonexistent")


# ---------------------------------------------------------------------------
# Engine: feature checks
# ---------------------------------------------------------------------------

def test_has_feature_default_plan():
    _, _, engine = _mk_engine()
    assert engine.has_feature("anon", "multi_source_triangulation")
    assert not engine.has_feature("anon", "vision_healing")


def test_has_feature_pro_plan():
    _, _, engine = _mk_engine()
    engine.assign_plan("acme", "pro")
    assert engine.has_feature("acme", "vision_healing")
    assert engine.has_feature("acme", "batch_urls")


def test_feature_override_grants_access():
    _, _, engine = _mk_engine()
    engine.assign_plan("vip", "free", overrides={
        "features_add": ["vision_healing"],
    })
    assert engine.has_feature("vip", "vision_healing")
    # free-tier defaults still hold
    assert engine.has_feature("vip", "multi_source_triangulation")


# ---------------------------------------------------------------------------
# Engine: resource checks
# ---------------------------------------------------------------------------

def test_check_no_limit_always_allowed():
    store = InMemoryUsageStore()
    engine = EntitlementEngine(store)
    engine.register_plan(Plan(name="unlimited", limits={}))
    engine.assign_plan("x", "unlimited")
    d = _run(engine.check("x", "page", requested=1_000_000))
    assert d.allowed is True
    assert "no limit" in d.reason


def test_check_under_limit():
    _, _, engine = _mk_engine()
    d = _run(engine.check("anon", "page", requested=100))
    assert d.allowed is True
    assert d.limit == 500
    assert d.consumed == 0
    assert d.remaining == 400


def test_check_exactly_at_limit():
    _, _, engine = _mk_engine()
    d = _run(engine.check("anon", "page", requested=500))
    assert d.allowed is True
    assert d.remaining == 0


def test_check_over_limit_blocked():
    _, _, engine = _mk_engine()
    d = _run(engine.check("anon", "page", requested=501))
    assert d.allowed is False
    assert "over limit" in d.reason


def test_check_uses_usage_store():
    store = InMemoryUsageStore()
    plans = default_plans()
    engine = EntitlementEngine(store, plans_by_name=plans)
    engine.assign_plan("acme", "pro")

    # Seed 15,000 pages
    _run(store.record_batch([
        UsageEvent(
            client_id="acme",
            resource_type=ResourceType.PAGE,
            quantity=15_000,
            occurred_at=engine.plan_for("acme").effective_from,
        ),
    ]))

    # Under (15_000 + 4_000 <= 20_000)
    d = _run(engine.check("acme", "page", requested=4_000))
    assert d.allowed is True
    assert d.consumed == 15_000
    assert d.remaining == 1_000

    # Over (15_000 + 10_000 > 20_000) — pro is WARN, so still allowed
    d = _run(engine.check("acme", "page", requested=10_000))
    assert d.allowed is True
    assert "WARN" in d.warning


def test_check_usd_resource():
    store = InMemoryUsageStore()
    plans = default_plans()
    engine = EntitlementEngine(store, plans_by_name=plans)
    engine.assign_plan("spendy", "free")

    _run(store.record_batch([
        UsageEvent(
            client_id="spendy",
            resource_type=ResourceType.TOKEN,
            quantity=1000, unit_cost_snapshot=0.001,   # $1
            occurred_at=engine.plan_for("spendy").effective_from,
        ),
    ]))

    # free cap is $1 — already at cap
    d = _run(engine.check("spendy", "usd", requested=0.01))
    assert d.allowed is False
    assert d.consumed == 1.0
    assert d.limit == 1.0


# ---------------------------------------------------------------------------
# Overage policies
# ---------------------------------------------------------------------------

def test_block_policy_rejects():
    store = InMemoryUsageStore()
    engine = EntitlementEngine(store)
    engine.register_plan(Plan(
        name="blocked", limits={"page": 10},
        overage_policy=OveragePolicy.BLOCK,
    ))
    engine.assign_plan("x", "blocked")
    d = _run(engine.check("x", "page", requested=11))
    assert d.allowed is False
    assert d.policy == "block"


def test_warn_policy_allows_with_warning():
    store = InMemoryUsageStore()
    engine = EntitlementEngine(store)
    engine.register_plan(Plan(
        name="warned", limits={"page": 10},
        overage_policy=OveragePolicy.WARN,
    ))
    engine.assign_plan("x", "warned")
    d = _run(engine.check("x", "page", requested=11))
    assert d.allowed is True
    assert d.warning and "WARN" in d.warning


def test_allow_policy_allows_with_warning():
    store = InMemoryUsageStore()
    engine = EntitlementEngine(store)
    engine.register_plan(Plan(
        name="allowed", limits={"page": 10},
        overage_policy=OveragePolicy.ALLOW,
    ))
    engine.assign_plan("x", "allowed")
    d = _run(engine.check("x", "page", requested=11))
    assert d.allowed is True
    assert d.warning and "ALLOW" in d.warning


# ---------------------------------------------------------------------------
# check_many
# ---------------------------------------------------------------------------

def test_check_many():
    _, _, engine = _mk_engine()
    decisions = _run(engine.check_many("anon", {
        "page": 100,
        "token": 10_000,
        "usd": 0.05,
    }))
    assert len(decisions) == 3
    assert all(isinstance(d, EntitlementDecision) for d in decisions)
    assert all(d.allowed for d in decisions)


def test_check_many_with_one_failure():
    _, _, engine = _mk_engine()
    engine.assign_plan("acme", "free")
    decisions = _run(engine.check_many("acme", {
        "page": 100,
        "token": 500_000,   # over the free 200_000 cap
    }))
    by_res = {d.resource: d for d in decisions}
    assert by_res["page"].allowed is True
    assert by_res["token"].allowed is False


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------

def test_explain_shape():
    _, _, engine = _mk_engine()
    engine.assign_plan("acme", "pro")
    info = engine.explain("acme")
    assert info["plan_name"] == "pro"
    assert "page" in info["limits"]
    assert "vision_healing" in info["features"]
    assert info["overage_policy"] == "warn"


def test_explain_default_plan():
    _, _, engine = _mk_engine()
    info = engine.explain("never-seen")
    assert info["plan_name"] == "free"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_consumed_for_usd():
    s = UsageSummary(client_id="x", total_cost_usd=1.5)
    assert _consumed_for(s, "usd") == 1.5


def test_consumed_for_resource():
    s = UsageSummary(
        client_id="x",
        by_resource={"page": {"event_count": 3, "quantity": 42.0, "cost_usd": 0.1}},
    )
    assert _consumed_for(s, "page") == 42.0


def test_consumed_for_unknown_resource():
    s = UsageSummary(client_id="x")
    assert _consumed_for(s, "page") == 0.0
    assert _consumed_for(s, "usd") == 0.0


def test_first_of_month_iso():
    from datetime import datetime, timezone
    s = _first_of_month_iso(datetime(2026, 9, 15, 14, 30, tzinfo=timezone.utc))
    assert s.startswith("2026-09-01T00:00:00")


def test_default_free_plan_has_expected_shape():
    p = _default_free_plan()
    assert p.name == "free"
    assert p.limits["page"] == 500
    assert p.overage_policy == OveragePolicy.BLOCK


def test_default_plans_catalogue():
    plans = default_plans()
    assert set(plans) == {"free", "pro", "enterprise"}
    assert plans["pro"].overage_policy == OveragePolicy.WARN
    assert plans["enterprise"].overage_policy == OveragePolicy.ALLOW


# ---------------------------------------------------------------------------
# Decision serialization
# ---------------------------------------------------------------------------

def test_decision_to_dict():
    d = EntitlementDecision(
        allowed=False,
        client_id="acme",
        resource="page",
        requested=100,
        limit=500,
        consumed=450,
        remaining=50,
        policy="block",
        reason="over limit",
    )
    out = d.to_dict()
    assert out["allowed"] is False
    assert out["remaining"] == 50