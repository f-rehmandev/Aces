"""
Usage metering types — spec §41.4.

Every meaningful resource consumption in ACES is recorded as an
immutable `UsageEvent`. Nothing mutates a usage event after it's
written — corrections happen by writing a new compensating event, so
historical usage can always be reconstructed exactly.

Resource types (§41.4):

    PAGE             one page fetched (any tier)
    BROWSER_SECOND   one second of headless-browser time
    TOKEN            one LLM token (input or output)
    VISION_CALL      one multimodal model invocation
    PROVIDER_CREDIT  one external-provider credit (ScraperAPI, etc.)
    STORAGE_BYTE     one byte written to object storage
    API_CALL         one authenticated API request

Each event records enough to answer three questions later:
    1. WHO paid for this?      (client_id, job_id, task_id)
    2. WHAT was consumed?       (resource_type, quantity, unit)
    3. HOW MUCH did it cost?    (unit_cost_snapshot, total_cost_usd)
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
# Resource types
# ---------------------------------------------------------------------------

class ResourceType(str, Enum):
    PAGE = "page"
    BROWSER_SECOND = "browser_second"
    TOKEN = "token"
    VISION_CALL = "vision_call"
    PROVIDER_CREDIT = "provider_credit"
    STORAGE_BYTE = "storage_byte"
    API_CALL = "api_call"


# The canonical unit for each resource
_UNIT_FOR_RESOURCE = {
    ResourceType.PAGE: "page",
    ResourceType.BROWSER_SECOND: "second",
    ResourceType.TOKEN: "token",
    ResourceType.VISION_CALL: "call",
    ResourceType.PROVIDER_CREDIT: "credit",
    ResourceType.STORAGE_BYTE: "byte",
    ResourceType.API_CALL: "call",
}


def unit_for(resource_type: ResourceType | str) -> str:
    if isinstance(resource_type, str):
        try:
            resource_type = ResourceType(resource_type)
        except ValueError:
            return "unit"
    return _UNIT_FOR_RESOURCE.get(resource_type, "unit")


# ---------------------------------------------------------------------------
# UsageEvent
# ---------------------------------------------------------------------------

@dataclass
class UsageEvent:
    """
    Immutable record of one consumption event.

    `unit_cost_snapshot` is captured at write time so future pricing
    changes never retroactively alter historical cost reports. Compute
    `total_cost_usd = quantity * unit_cost_snapshot` when you need the
    total, or read the precomputed `total_cost_usd` field.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Who
    client_id: str = "default"
    job_id: str = ""
    task_id: str = ""

    # What
    resource_type: ResourceType = ResourceType.PAGE
    quantity: float = 0.0
    unit: str = ""                         # populated from resource_type if blank

    # Cost
    unit_cost_snapshot: float = 0.0        # USD per unit at write time
    total_cost_usd: float = 0.0            # quantity * unit_cost, or explicit

    # Context
    provider: str = ""                     # "scraperapi" | "gemini-2.5" | "s3" | ...
    metadata: dict = field(default_factory=dict)

    occurred_at: str = field(default_factory=_utc_now_iso)

    # ------------------------------------------------------------------
    def __post_init__(self):
        if not self.unit:
            self.unit = unit_for(self.resource_type)
        # Auto-compute total if caller only supplied unit cost
        if self.total_cost_usd == 0.0 and self.unit_cost_snapshot > 0:
            self.total_cost_usd = round(
                self.quantity * self.unit_cost_snapshot, 8,
            )
        if self.quantity < 0:
            raise ValueError("quantity must be >= 0")
        if self.unit_cost_snapshot < 0:
            raise ValueError("unit_cost_snapshot must be >= 0")
        if self.total_cost_usd < 0:
            raise ValueError("total_cost_usd must be >= 0")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resource_type"] = self.resource_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "UsageEvent":
        d = dict(data)
        try:
            d["resource_type"] = ResourceType(
                d.get("resource_type", "page")
            )
        except ValueError:
            d["resource_type"] = ResourceType.PAGE
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Aggregated summary
# ---------------------------------------------------------------------------

@dataclass
class UsageSummary:
    """Aggregated view of usage over a window."""
    client_id: str = ""
    window_start: str = ""
    window_end: str = ""
    by_resource: dict[str, dict] = field(default_factory=dict)
    event_count: int = 0
    total_cost_usd: float = 0.0
    total_quantity: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Auto-populate unit + compute total
    e = UsageEvent(
        client_id="acme",
        resource_type=ResourceType.TOKEN,
        quantity=1000,
        unit_cost_snapshot=0.000075,
        provider="gemini",
    )
    assert e.unit == "token"
    assert abs(e.total_cost_usd - 0.075) < 1e-9, e.total_cost_usd

    # Explicit total wins
    e2 = UsageEvent(
        resource_type=ResourceType.PAGE,
        quantity=5,
        unit_cost_snapshot=0.01,
        total_cost_usd=99.99,
    )
    assert e2.total_cost_usd == 99.99

    # Zero cost allowed
    e3 = UsageEvent(resource_type=ResourceType.PAGE, quantity=1)
    assert e3.total_cost_usd == 0.0
    assert e3.unit == "page"

    # Round-trip
    d = e.to_dict()
    assert d["resource_type"] == "token"
    e4 = UsageEvent.from_dict(d)
    assert e4.quantity == 1000
    assert e4.resource_type == ResourceType.TOKEN

    # Bad resource type → PAGE
    e5 = UsageEvent.from_dict({"resource_type": "nonsense"})
    assert e5.resource_type == ResourceType.PAGE

    # Negative quantity rejected
    try:
        UsageEvent(quantity=-1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # unit_for helper
    assert unit_for(ResourceType.TOKEN) == "token"
    assert unit_for("provider_credit") == "credit"
    assert unit_for("nonsense") == "unit"

    # Summary shape
    s = UsageSummary(
        client_id="acme", event_count=3, total_cost_usd=0.15,
        by_resource={"token": {"quantity": 3000, "cost": 0.15}},
    )
    assert s.to_dict()["event_count"] == 3

    print("Usage types OK.")