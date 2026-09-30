"""
Entitlement engine — spec §47B.

Decides whether a client is ALLOWED to do something, given their plan
and their usage so far. Sits between the usage store (§41.4) and the
callers who want to gate expensive work.

Model:

    Plan
    ├── limits          (per-resource monthly caps)
    ├── features        (named boolean flags)
    ├── quotas          (per-job caps, e.g. max_pages_per_job)
    └── overage_policy  (what to do when a limit is exceeded)

    ClientPlan
    ├── client_id
    ├── plan            (the Plan instance)
    ├── overrides       (per-client adjustments)
    └── effective_from  (billing period start)

Public API:

    engine = EntitlementEngine(plans_by_name, usage_store)

    decision = await engine.check(
        client_id="acme",
        resource="token",
        requested=50_000,
    )
    if not decision.allowed:
        ...

    if engine.has_feature(client_id="acme", feature="vision_healing"):
        ...

No dependency on Stripe or any payment provider. A future
`BillingProviderAdapter` sits above this engine — the engine only
reports allowances, it doesn't charge anyone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.usage.store import UsageStore
from src.usage.types import ResourceType


logger = logging.getLogger("usage.entitlements")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _first_of_month_iso(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0,
    ).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Overage policy
# ---------------------------------------------------------------------------

class OveragePolicy(str, Enum):
    """
    What happens when a client tries to exceed a monthly limit.

        BLOCK       refuse the request (default, safest)
        ALLOW       allow it (charge later via billing adapter)
        WARN        allow but emit a warning in the decision
    """
    BLOCK = "block"
    ALLOW = "allow"
    WARN = "warn"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    """
    A named set of entitlements.

    `limits` keys are ResourceType values ("token", "page", ...) or
    "usd" for a total-cost cap. Values are monthly caps in the
    resource's natural unit. Missing keys mean "no limit".

    `features` is a set of string flags ("vision_healing",
    "batch_urls", "multi_source_triangulation", ...).

    `quotas` is a dict of per-job caps, e.g. {"max_pages_per_job": 100}.
    """
    name: str
    limits: dict[str, float] = field(default_factory=dict)
    features: set[str] = field(default_factory=set)
    quotas: dict[str, float] = field(default_factory=dict)
    overage_policy: OveragePolicy = OveragePolicy.BLOCK
    description: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = sorted(self.features)
        d["overage_policy"] = self.overage_policy.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        d = dict(data)
        try:
            d["overage_policy"] = OveragePolicy(
                d.get("overage_policy", "block")
            )
        except ValueError:
            d["overage_policy"] = OveragePolicy.BLOCK
        if isinstance(d.get("features"), list):
            d["features"] = set(d["features"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def limit_for(self, resource: str) -> Optional[float]:
        return self.limits.get(resource)


# ---------------------------------------------------------------------------
# Client-plan assignment
# ---------------------------------------------------------------------------

@dataclass
class ClientPlan:
    """
    Links a client to a Plan. `overrides` allows per-client tweaks
    on top of the plan:
        overrides = {
            "limits": {"token": 2_000_000},   # raise the token cap
            "features_add": ["vision_healing"],
            "features_remove": ["batch_urls"],
        }
    """
    client_id: str
    plan: Plan
    overrides: dict = field(default_factory=dict)
    effective_from: str = field(default_factory=_first_of_month_iso)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def effective_limits(self) -> dict[str, float]:
        merged = dict(self.plan.limits)
        merged.update(self.overrides.get("limits", {}) or {})
        return merged

    def effective_features(self) -> set[str]:
        base = set(self.plan.features)
        base.update(self.overrides.get("features_add", []) or [])
        base.difference_update(self.overrides.get("features_remove", []) or [])
        return base

    def limit_for(self, resource: str) -> Optional[float]:
        return self.effective_limits().get(resource)

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "plan": self.plan.to_dict(),
            "overrides": dict(self.overrides),
            "effective_from": self.effective_from,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class EntitlementDecision:
    allowed: bool
    client_id: str = ""
    resource: str = ""
    requested: float = 0.0
    limit: Optional[float] = None
    consumed: float = 0.0
    remaining: Optional[float] = None
    policy: str = OveragePolicy.BLOCK.value
    reason: str = ""
    warning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EntitlementEngine:
    """
    Ties Plans to a UsageStore. Ask it whether something is allowed.

    The engine is intentionally read-only with respect to the store:
    it never mutates usage. Callers record usage separately; the
    engine only answers "given what's already been used, is this
    request allowed?".
    """

    def __init__(
        self,
        usage_store: UsageStore,
        plans_by_name: Optional[dict[str, Plan]] = None,
        client_plans: Optional[dict[str, ClientPlan]] = None,
        default_plan: Optional[Plan] = None,
    ):
        self.usage_store = usage_store
        self.plans: dict[str, Plan] = dict(plans_by_name or {})
        self.client_plans: dict[str, ClientPlan] = dict(client_plans or {})
        self.default_plan = default_plan or _default_free_plan()

    # ------------------------------------------------------------------
    # Plan registration
    # ------------------------------------------------------------------
    def register_plan(self, plan: Plan) -> None:
        self.plans[plan.name] = plan

    def assign_plan(
        self,
        client_id: str,
        plan_name: str,
        overrides: Optional[dict] = None,
    ) -> ClientPlan:
        if plan_name not in self.plans:
            raise KeyError(f"unknown plan {plan_name!r}")
        cp = ClientPlan(
            client_id=client_id,
            plan=self.plans[plan_name],
            overrides=dict(overrides or {}),
        )
        self.client_plans[client_id] = cp
        return cp

    def plan_for(self, client_id: str) -> ClientPlan:
        if client_id in self.client_plans:
            return self.client_plans[client_id]
        return ClientPlan(client_id=client_id, plan=self.default_plan)

    # ------------------------------------------------------------------
    # Feature checks
    # ------------------------------------------------------------------
    def has_feature(self, client_id: str, feature: str) -> bool:
        return feature in self.plan_for(client_id).effective_features()

    # ------------------------------------------------------------------
    # Resource checks
    # ------------------------------------------------------------------
    async def check(
        self,
        client_id: str,
        resource: str,
        requested: float = 0.0,
        since: Optional[str] = None,
    ) -> EntitlementDecision:
        """
        Is `client_id` allowed to consume `requested` more of `resource`
        in the current window?
        """
        plan = self.plan_for(client_id)
        limit = plan.limit_for(resource)
        policy = plan.plan.overage_policy

        # No limit — always allowed
        if limit is None:
            return EntitlementDecision(
                allowed=True,
                client_id=client_id,
                resource=resource,
                requested=requested,
                policy=policy.value,
                reason="no limit configured",
            )

        # Fetch current consumption in window
        since = since or plan.effective_from
        summary = await self.usage_store.summarize(
            client_id, since=since,
        )
        consumed = _consumed_for(summary, resource)
        remaining = max(0.0, limit - consumed)

        # Under the cap
        if consumed + requested <= limit:
            return EntitlementDecision(
                allowed=True,
                client_id=client_id,
                resource=resource,
                requested=requested,
                limit=limit,
                consumed=consumed,
                remaining=max(0.0, remaining - requested),
                policy=policy.value,
                reason="within limit",
            )

        # Over the cap — policy decides
        overage = consumed + requested - limit
        if policy == OveragePolicy.ALLOW:
            return EntitlementDecision(
                allowed=True,
                client_id=client_id,
                resource=resource,
                requested=requested,
                limit=limit,
                consumed=consumed,
                remaining=0.0,
                policy=policy.value,
                reason="allowed by overage policy",
                warning=(
                    f"over limit by {overage:.2f} {resource} "
                    f"(overage policy: ALLOW)"
                ),
            )

        if policy == OveragePolicy.WARN:
            return EntitlementDecision(
                allowed=True,
                client_id=client_id,
                resource=resource,
                requested=requested,
                limit=limit,
                consumed=consumed,
                remaining=0.0,
                policy=policy.value,
                reason="allowed with warning",
                warning=(
                    f"over limit by {overage:.2f} {resource} "
                    f"(overage policy: WARN)"
                ),
            )

        # BLOCK
        return EntitlementDecision(
            allowed=False,
            client_id=client_id,
            resource=resource,
            requested=requested,
            limit=limit,
            consumed=consumed,
            remaining=remaining,
            policy=policy.value,
            reason=(
                f"over limit: consumed={consumed:.2f}, "
                f"requested={requested:.2f}, limit={limit:.2f}"
            ),
        )

    async def check_many(
        self,
        client_id: str,
        requests: dict[str, float],
    ) -> list[EntitlementDecision]:
        out: list[EntitlementDecision] = []
        for resource, qty in requests.items():
            out.append(await self.check(client_id, resource, qty))
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def explain(self, client_id: str) -> dict:
        plan = self.plan_for(client_id)
        return {
            "client_id": client_id,
            "plan_name": plan.plan.name,
            "limits": plan.effective_limits(),
            "features": sorted(plan.effective_features()),
            "quotas": dict(plan.plan.quotas),
            "overage_policy": plan.plan.overage_policy.value,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _consumed_for(summary, resource: str) -> float:
    """
    Given a UsageSummary, return the amount consumed for the given
    resource. `usd` maps to summary.total_cost_usd; anything else maps
    to summary.by_resource[resource]["quantity"].
    """
    if resource == "usd":
        return float(summary.total_cost_usd)
    bucket = summary.by_resource.get(resource)
    if not bucket:
        return 0.0
    return float(bucket.get("quantity", 0.0))


def _default_free_plan() -> Plan:
    """
    Applied to clients with no explicit assignment. Conservative but
    usable — enough for real work, not enough to run away.
    """
    return Plan(
        name="free",
        description="Default plan for unassigned clients",
        limits={
            ResourceType.PAGE.value: 500,
            ResourceType.TOKEN.value: 200_000,
            ResourceType.PROVIDER_CREDIT.value: 100,
            "usd": 1.00,
        },
        features={"multi_source_triangulation"},
        quotas={"max_pages_per_job": 100},
        overage_policy=OveragePolicy.BLOCK,
    )


def default_plans() -> dict[str, Plan]:
    """
    The built-in plan catalogue. Register these in the engine at
    startup, or override per deployment.
    """
    free = _default_free_plan()

    pro = Plan(
        name="pro",
        description="Paid tier — heavier usage, more features",
        limits={
            ResourceType.PAGE.value: 20_000,
            ResourceType.TOKEN.value: 10_000_000,
            ResourceType.PROVIDER_CREDIT.value: 5_000,
            "usd": 50.00,
        },
        features={
            "multi_source_triangulation",
            "vision_healing",
            "batch_urls",
            "self_healing",
        },
        quotas={"max_pages_per_job": 5_000},
        overage_policy=OveragePolicy.WARN,
    )

    enterprise = Plan(
        name="enterprise",
        description="Enterprise — no practical caps, ALLOW overage",
        limits={
            ResourceType.PAGE.value: 500_000,
            ResourceType.TOKEN.value: 200_000_000,
            ResourceType.PROVIDER_CREDIT.value: 200_000,
            "usd": 5000.00,
        },
        features={
            "multi_source_triangulation",
            "vision_healing",
            "batch_urls",
            "self_healing",
            "custom_connectors",
            "priority_queue",
        },
        quotas={"max_pages_per_job": 100_000},
        overage_policy=OveragePolicy.ALLOW,
    )

    return {"free": free, "pro": pro, "enterprise": enterprise}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from src.usage.store import InMemoryUsageStore
    from src.usage.types import UsageEvent

    async def run():
        store = InMemoryUsageStore()
        plans = default_plans()
        engine = EntitlementEngine(store, plans_by_name=plans)

        # ---- Default plan assigned automatically ----
        assert engine.plan_for("anon").plan.name == "free"
        assert engine.plan_for("anon").effective_limits()[
            ResourceType.PAGE.value
        ] == 500

        # ---- Feature checks ----
        assert engine.has_feature("anon", "multi_source_triangulation")
        assert not engine.has_feature("anon", "vision_healing")

        # ---- Under limit ----
        d = await engine.check("anon", ResourceType.PAGE.value, requested=100)
        assert d.allowed is True
        assert d.limit == 500
        assert d.consumed == 0
        assert d.remaining == 400

        # ---- Add usage, then re-check ----
        await store.record_batch([
            UsageEvent(
                client_id="anon",
                resource_type=ResourceType.PAGE,
                quantity=450,
                occurred_at=engine.plan_for("anon").effective_from,
            ),
        ])
        d = await engine.check("anon", ResourceType.PAGE.value, requested=40)
        assert d.allowed is True   # 450 + 40 <= 500
        assert d.consumed == 450
        assert d.remaining == 10

        d = await engine.check("anon", ResourceType.PAGE.value, requested=100)
        assert d.allowed is False   # 450 + 100 > 500
        assert "over limit" in d.reason

        # ---- WARN policy ----
        engine.assign_plan("warn-client", "pro")
        engine.plans["pro"].overage_policy = OveragePolicy.WARN
        await store.record_batch([
            UsageEvent(
                client_id="warn-client",
                resource_type=ResourceType.PAGE,
                quantity=19_950,
                occurred_at=engine.plan_for("warn-client").effective_from,
            ),
        ])
        d = await engine.check("warn-client", ResourceType.PAGE.value, 200)
        assert d.allowed is True
        assert "WARN" in d.warning

        # ---- ALLOW policy ----
        engine.assign_plan("ent", "enterprise")
        await store.record_batch([
            UsageEvent(
                client_id="ent",
                resource_type=ResourceType.PAGE,
                quantity=500_000,
                occurred_at=engine.plan_for("ent").effective_from,
            ),
        ])
        d = await engine.check("ent", ResourceType.PAGE.value, 1)
        assert d.allowed is True
        assert "ALLOW" in d.warning

        # ---- USD resource ----
        engine.assign_plan("spendy", "free")
        await store.record_batch([
            UsageEvent(
                client_id="spendy",
                resource_type=ResourceType.TOKEN,
                quantity=1000, unit_cost_snapshot=0.001,  # $1
                occurred_at=engine.plan_for("spendy").effective_from,
            ),
        ])
        d = await engine.check("spendy", "usd", 0.50)
        assert d.allowed is False   # $1.00 cap, already at $1
        assert d.consumed == 1.0

        # ---- Feature override ----
        engine.assign_plan("vip", "free", overrides={
            "features_add": ["vision_healing"],
            "limits": {"page": 10_000},
        })
        assert engine.has_feature("vip", "vision_healing")
        assert engine.plan_for("vip").limit_for("page") == 10_000

        # ---- check_many ----
        decisions = await engine.check_many("anon", {
            "page": 10, "token": 1000,
        })
        assert len(decisions) == 2
        assert all(isinstance(x, EntitlementDecision) for x in decisions)

        # ---- explain ----
        info = engine.explain("ent")
        assert info["plan_name"] == "enterprise"
        assert "priority_queue" in info["features"]

        # ---- Plan serialization ----
        p = plans["pro"]
        d = p.to_dict()
        assert d["overage_policy"] == "warn"
        p2 = Plan.from_dict(d)
        assert p2.name == "pro"
        assert p2.overage_policy == OveragePolicy.WARN

        print("Entitlement engine OK.")

    asyncio.run(run())