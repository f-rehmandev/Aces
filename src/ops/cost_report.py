"""
Per-run cost reports — spec §41.4.

Reads usage_events back from a UsageStore and converts them into a USD
estimate. This is what makes the metering useful to a human: a run's
total cost, its breakdown by resource type, and (given a record count)
a cost-per-1k-validated-records metric.

Design notes:
    - Rates are a dataclass so they can be injected per deployment.
    - Token cost uses the input/output split in event metadata when
      present; falls back to a blended per-token rate otherwise.
    - `by_provider` is populated from the PROVIDER_CREDIT event's
      metadata.breakdown dict. Other resource types do not currently
      carry provider identity at the event level.
    - The reporter never raises on read failures — a broken store
      returns an empty report rather than breaking the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.usage.store import UsageStore
from src.usage.types import ResourceType, UsageEvent


logger = logging.getLogger("ops.cost_report")


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

@dataclass
class CostRates:
    """
    USD rates used to convert usage quantities into cost estimates.

    Defaults are deliberately conservative:
        - Token rates     — Gemini 1.5 Flash free-tier equivalent
        - Browser second  — zero (local Playwright; no cloud bill)
        - Provider credit — ScraperAPI mid-tier, ~$1 per 1,000 credits
        - Vision call     — zero (Gemini free tier for images)
        - Page            — zero (direct fetch, no paid provider)

    Callers deploying against paid infrastructure should override
    these with real account rates.
    """
    token_input_per_1k: float = 0.000_075
    token_output_per_1k: float = 0.000_300
    token_blended_per_1k: float = 0.000_150   # used when split isn't in metadata
    browser_second: float = 0.0
    provider_credit: float = 0.001
    vision_call: float = 0.0
    page: float = 0.0

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class CostReport:
    """
    Result of a cost computation. All monetary values are USD.

    `by_resource` maps resource_type string → {"quantity", "usd"}.
    `by_provider` maps provider name    → {"quantity", "usd"} (provider
    credits only; populated from the PROVIDER_CREDIT event's metadata).
    """
    job_id: str = ""
    client_id: str = ""
    total_usd: float = 0.0
    by_resource: dict[str, dict] = field(default_factory=dict)
    by_provider: dict[str, dict] = field(default_factory=dict)
    event_count: int = 0
    items_listed: int = 0
    cost_per_1k_items: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "client_id": self.client_id,
            "total_usd": round(self.total_usd, 8),
            "by_resource": {
                k: {
                    "quantity": round(v["quantity"], 4),
                    "usd": round(v["usd"], 8),
                }
                for k, v in self.by_resource.items()
            },
            "by_provider": {
                k: {
                    "quantity": round(v["quantity"], 4),
                    "usd": round(v["usd"], 8),
                }
                for k, v in self.by_provider.items()
            },
            "event_count": self.event_count,
            "items_listed": self.items_listed,
            "cost_per_1k_items": (
                round(self.cost_per_1k_items, 6)
                if self.cost_per_1k_items is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class CostReporter:
    """
    Converts a set of UsageEvents into a CostReport.

    Usage:
        reporter = CostReporter(usage_store)
        report = await reporter.report_for_job("job-id")
        print(report.to_dict())

    Or for a client over a window:
        report = await reporter.report_for_client("acme", since="2026-01-01")
    """

    def __init__(
        self,
        usage_store: UsageStore,
        rates: Optional[CostRates] = None,
    ):
        self.usage_store = usage_store
        self.rates = rates or CostRates()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def report_for_job(
        self,
        job_id: str,
        *,
        items_listed: int = 0,
    ) -> CostReport:
        if not job_id:
            return CostReport()
        try:
            events = await self.usage_store.events_for_job(job_id)
        except Exception as e:
            logger.warning(
                f"cost_report: read failed for job {job_id!r}: "
                f"{type(e).__name__}: {e}"
            )
            return CostReport(job_id=job_id)
        return self._build(events, job_id=job_id, items_listed=items_listed)

    async def report_for_client(
        self,
        client_id: str,
        *,
        since: str = "",
        until: str = "",
        items_listed: int = 0,
    ) -> CostReport:
        if not client_id:
            return CostReport()
        try:
            # Reuse the summary path when there's no job filter — it
            # handles the window in one round-trip on the Supabase side.
            if not since and not until:
                summary = await self.usage_store.summarize(client_id)
            else:
                summary = await self.usage_store.summarize(
                    client_id, since=since, until=until,
                )
        except Exception as e:
            logger.warning(
                f"cost_report: read failed for client {client_id!r}: "
                f"{type(e).__name__}: {e}"
            )
            return CostReport(client_id=client_id)

        # The summary aggregates per resource but drops the metadata we
        # need to split tokens and provider credits. Fetch the raw events
        # so we can compute those accurately.
        try:
            events = await self.usage_store.events_for_client(client_id)
        except Exception:
            events = []
        if since:
            events = [e for e in events if e.occurred_at >= since]
        if until:
            events = [e for e in events if e.occurred_at < until]

        return self._build(
            events, client_id=client_id, items_listed=items_listed,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build(
        self,
        events: list[UsageEvent],
        *,
        job_id: str = "",
        client_id: str = "",
        items_listed: int = 0,
    ) -> CostReport:
        report = CostReport(
            job_id=job_id,
            client_id=client_id,
            event_count=len(events),
            items_listed=max(0, int(items_listed)),
        )

        for event in events:
            usd = self._usd_for_event(event)
            rkey = (
                event.resource_type.value
                if isinstance(event.resource_type, ResourceType)
                else str(event.resource_type)
            )

            # Per-resource aggregation
            bucket = report.by_resource.setdefault(
                rkey, {"quantity": 0.0, "usd": 0.0},
            )
            bucket["quantity"] += float(event.quantity)
            bucket["usd"] += usd

            # Per-provider aggregation (PROVIDER_CREDIT only)
            if (
                isinstance(event.resource_type, ResourceType)
                and event.resource_type == ResourceType.PROVIDER_CREDIT
            ):
                breakdown = (event.metadata or {}).get("breakdown") or {}
                if isinstance(breakdown, dict) and breakdown:
                    for provider_name, credits in breakdown.items():
                        try:
                            qty = float(credits)
                        except (TypeError, ValueError):
                            continue
                        pb = report.by_provider.setdefault(
                            str(provider_name), {"quantity": 0.0, "usd": 0.0},
                        )
                        pb["quantity"] += qty
                        pb["usd"] += qty * self.rates.provider_credit
                elif usd > 0:
                    # No breakdown — attribute to a generic provider label.
                    pb = report.by_provider.setdefault(
                        event.provider or "unknown",
                        {"quantity": 0.0, "usd": 0.0},
                    )
                    pb["quantity"] += float(event.quantity)
                    pb["usd"] += usd

            report.total_usd += usd

        if report.items_listed > 0 and report.total_usd > 0:
            report.cost_per_1k_items = (
                report.total_usd / report.items_listed * 1000.0
            )
        elif report.items_listed > 0:
            # Zero cost is a legitimate answer (e.g. all-free-tier run).
            report.cost_per_1k_items = 0.0

        return report

    def _usd_for_event(self, event: UsageEvent) -> float:
        rtype = event.resource_type
        qty = float(event.quantity)

        if rtype == ResourceType.TOKEN:
            return self._token_usd(event, qty)
        if rtype == ResourceType.BROWSER_SECOND:
            return qty * self.rates.browser_second
        if rtype == ResourceType.PROVIDER_CREDIT:
            return qty * self.rates.provider_credit
        if rtype == ResourceType.VISION_CALL:
            return qty * self.rates.vision_call
        if rtype == ResourceType.PAGE:
            return qty * self.rates.page
        return 0.0

    def _token_usd(self, event: UsageEvent, qty: float) -> float:
        """
        Token cost. Uses the input/output split when metadata carries it,
        otherwise falls back to a blended rate.
        """
        meta = event.metadata or {}
        try:
            in_tokens = float(meta.get("input_tokens", 0) or 0)
            out_tokens = float(meta.get("output_tokens", 0) or 0)
        except (TypeError, ValueError):
            in_tokens = out_tokens = 0.0

        if in_tokens or out_tokens:
            return (
                in_tokens / 1000.0 * self.rates.token_input_per_1k
                + out_tokens / 1000.0 * self.rates.token_output_per_1k
            )
        # No split — treat the total as blended.
        return qty / 1000.0 * self.rates.token_blended_per_1k


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    from src.usage.store import InMemoryUsageStore
    from src.usage.types import UsageEvent

    async def main():
        store = InMemoryUsageStore()
        await store.record_batch([
            UsageEvent(
                client_id="acme", job_id="j-1",
                resource_type=ResourceType.PAGE, quantity=10,
            ),
            UsageEvent(
                client_id="acme", job_id="j-1",
                resource_type=ResourceType.TOKEN, quantity=5000,
                metadata={"input_tokens": 4000, "output_tokens": 1000,
                          "llm_calls": 3},
            ),
            UsageEvent(
                client_id="acme", job_id="j-1",
                resource_type=ResourceType.BROWSER_SECOND, quantity=22.5,
            ),
            UsageEvent(
                client_id="acme", job_id="j-1",
                resource_type=ResourceType.PROVIDER_CREDIT, quantity=30,
                metadata={"provider_calls": 3,
                          "breakdown": {"scraperapi": 25, "scrapingant": 5}},
            ),
            UsageEvent(
                client_id="acme", job_id="j-1",
                resource_type=ResourceType.VISION_CALL, quantity=2,
            ),
        ])

        reporter = CostReporter(store)
        report = await reporter.report_for_job("j-1", items_listed=50)
        import json
        print(json.dumps(report.to_dict(), indent=2))

    asyncio.run(main())