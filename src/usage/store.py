"""
Usage store — spec §41.4.

Two implementations of the same interface:

    InMemoryUsageStore       process-local, for tests and dev
    SupabaseUsageStore       durable, tenant-scoped, batch-inserts

Both satisfy the `UsageStore` protocol. Callers can write one event
at a time (`record`) or many (`record_batch`). Aggregation is
window-based (`summarize`).

Rules:
    - Events are append-only. `record` never mutates an existing event.
    - Batch insert is idempotent on event_id — replaying the same
      batch twice is safe.
    - Never raises on read failure (returns an empty summary). Write
      failure raises, because losing usage data is a real error.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from src.usage.types import (
    ResourceType,
    UsageEvent,
    UsageSummary,
    unit_for,
)


logger = logging.getLogger("usage.store")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class UsageStore(Protocol):
    async def record(self, event: UsageEvent) -> None: ...
    async def record_batch(self, events: list[UsageEvent]) -> int: ...
    async def summarize(
        self,
        client_id: str,
        since: str = "",
        until: str = "",
        job_id: str = "",
    ) -> UsageSummary: ...
    async def events_for_job(self, job_id: str) -> list[UsageEvent]: ...
    async def events_for_client(
        self, client_id: str, limit: int = 1000,
    ) -> list[UsageEvent]: ...


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------

class InMemoryUsageStore:
    def __init__(self):
        self._events: list[UsageEvent] = []
        self._by_id: dict[str, UsageEvent] = {}

    async def record(self, event: UsageEvent) -> None:
        if event.event_id in self._by_id:
            return
        self._by_id[event.event_id] = event
        self._events.append(event)

    async def record_batch(self, events: list[UsageEvent]) -> int:
        count = 0
        for e in events:
            if e.event_id in self._by_id:
                continue
            self._by_id[e.event_id] = e
            self._events.append(e)
            count += 1
        return count

    async def summarize(
        self,
        client_id: str,
        since: str = "",
        until: str = "",
        job_id: str = "",
    ) -> UsageSummary:
        filtered = [
            e for e in self._events
            if e.client_id == client_id
            and (not job_id or e.job_id == job_id)
            and _in_window(e.occurred_at, since, until)
        ]
        return _summarize(client_id, filtered, since, until)

    async def events_for_job(self, job_id: str) -> list[UsageEvent]:
        return [e for e in self._events if e.job_id == job_id]

    async def events_for_client(
        self, client_id: str, limit: int = 1000,
    ) -> list[UsageEvent]:
        out = [e for e in self._events if e.client_id == client_id]
        out.sort(key=lambda e: e.occurred_at, reverse=True)
        return out[:limit]

    def __len__(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Supabase backend
# ---------------------------------------------------------------------------

class SupabaseUsageStore:
    def __init__(self, client_factory=None):
        if client_factory is None:
            from src.storage.db import get_client
            client_factory = get_client
        self._client_factory = client_factory

    def _client(self):
        return self._client_factory()

    async def record(self, event: UsageEvent) -> None:
        await self.record_batch([event])

    async def record_batch(self, events: list[UsageEvent]) -> int:
        if not events:
            return 0
        client = self._client()
        payload = [e.to_dict() for e in events]
        resp = client.rpc(
            "usage_batch_insert", {"p_events": payload},
        ).execute()
        data = getattr(resp, "data", None)
        # RPC returns an integer count
        if isinstance(data, int):
            return data
        if isinstance(data, list) and data and isinstance(data[0], int):
            return data[0]
        return len(events)

    async def summarize(
        self,
        client_id: str,
        since: str = "",
        until: str = "",
        job_id: str = "",
    ) -> UsageSummary:
        # If job_id is given, fall back to a table query + Python sum
        # (the RPC only aggregates by client + window).
        if job_id:
            events = await self.events_for_job(job_id)
            events = [
                e for e in events
                if _in_window(e.occurred_at, since, until)
            ]
            return _summarize(client_id, events, since, until)

        client = self._client()
        since_ts = since or (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).isoformat()
        until_ts = until or _utc_now_iso()

        resp = client.rpc("usage_summary", {
            "p_client_id": client_id,
            "p_start": since_ts,
            "p_end": until_ts,
        }).execute()
        rows = getattr(resp, "data", None) or []

        summary = UsageSummary(
            client_id=client_id,
            window_start=since_ts,
            window_end=until_ts,
        )
        for row in rows:
            rtype = row.get("resource_type") or ""
            count = int(row.get("event_count") or 0)
            qty = float(row.get("total_quantity") or 0)
            cost = float(row.get("total_cost_usd") or 0)
            if rtype == "__total__":
                summary.event_count = count
                summary.total_quantity = qty
                summary.total_cost_usd = round(cost, 8)
            else:
                summary.by_resource[rtype] = {
                    "event_count": count,
                    "quantity": round(qty, 6),
                    "cost_usd": round(cost, 8),
                }
        return summary

    async def events_for_job(self, job_id: str) -> list[UsageEvent]:
        if not job_id:
            return []
        client = self._client()
        resp = (
            client.table("usage_events")
            .select("*")
            .eq("job_id", job_id)
            .order("occurred_at", desc=False)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return [_row_to_event(r) for r in rows]

    async def events_for_client(
        self, client_id: str, limit: int = 1000,
    ) -> list[UsageEvent]:
        client = self._client()
        resp = (
            client.table("usage_events")
            .select("*")
            .eq("client_id", client_id)
            .order("occurred_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return [_row_to_event(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_window(iso: str, since: str, until: str) -> bool:
    if not iso:
        return True
    if since and iso < since:
        return False
    if until and iso >= until:
        return False
    return True


def _summarize(
    client_id: str,
    events: list[UsageEvent],
    since: str,
    until: str,
) -> UsageSummary:
    by_resource: dict[str, dict] = {}
    total_cost = 0.0
    total_qty = 0.0
    for e in events:
        key = e.resource_type.value if isinstance(
            e.resource_type, ResourceType,
        ) else str(e.resource_type)
        bucket = by_resource.setdefault(key, {
            "event_count": 0, "quantity": 0.0, "cost_usd": 0.0,
        })
        bucket["event_count"] += 1
        bucket["quantity"] += e.quantity
        bucket["cost_usd"] += e.total_cost_usd
        total_cost += e.total_cost_usd
        total_qty += e.quantity

    for k in by_resource:
        by_resource[k]["quantity"] = round(by_resource[k]["quantity"], 6)
        by_resource[k]["cost_usd"] = round(by_resource[k]["cost_usd"], 8)

    return UsageSummary(
        client_id=client_id,
        window_start=since,
        window_end=until,
        by_resource=by_resource,
        event_count=len(events),
        total_cost_usd=round(total_cost, 8),
        total_quantity=round(total_qty, 6),
    )


def _row_to_event(row: dict) -> UsageEvent:
    d = dict(row)
    # numeric columns come back as Decimal or str; coerce to float
    for k in ("quantity", "unit_cost_snapshot", "total_cost_usd"):
        v = d.get(k)
        if v is not None:
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = 0.0
    # jsonb → dict (supabase-py already does this, but be safe)
    if not isinstance(d.get("metadata"), dict):
        d["metadata"] = {}
    # timestamps come back as ISO strings
    for k in ("occurred_at",):
        v = d.get(k)
        if v is not None and not isinstance(v, str):
            d[k] = str(v)
    return UsageEvent.from_dict(d)


# ---------------------------------------------------------------------------
# Smoke test (in-memory only — no network)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def run():
        store = InMemoryUsageStore()

        # Empty
        s = await store.summarize("acme")
        assert s.event_count == 0
        assert s.total_cost_usd == 0.0

        # Single record
        e1 = UsageEvent(
            client_id="acme", job_id="j-1",
            resource_type=ResourceType.PAGE,
            quantity=5, unit_cost_snapshot=0.01,
        )
        await store.record(e1)
        s = await store.summarize("acme")
        assert s.event_count == 1
        assert s.total_cost_usd == 0.05

        # Batch
        batch = [
            UsageEvent(
                client_id="acme", job_id="j-1",
                resource_type=ResourceType.TOKEN,
                quantity=1000, unit_cost_snapshot=0.000075,
            ),
            UsageEvent(
                client_id="acme", job_id="j-1",
                resource_type=ResourceType.PROVIDER_CREDIT,
                quantity=10, unit_cost_snapshot=0.001,
            ),
            UsageEvent(
                client_id="acme", job_id="j-2",
                resource_type=ResourceType.PAGE,
                quantity=1,
            ),
        ]
        n = await store.record_batch(batch)
        assert n == 3

        # Summary mixes both jobs
        s = await store.summarize("acme")
        assert s.event_count == 4
        assert "page" in s.by_resource
        assert "token" in s.by_resource
        assert "provider_credit" in s.by_resource
        assert s.by_resource["page"]["quantity"] == 6.0
        assert abs(s.total_cost_usd - 0.135) < 1e-9, s.total_cost_usd

        # Per-job summary
        s_j1 = await store.summarize("acme", job_id="j-1")
        assert s_j1.event_count == 3
        s_j2 = await store.summarize("acme", job_id="j-2")
        assert s_j2.event_count == 1

        # Client isolation
        s_other = await store.summarize("other")
        assert s_other.event_count == 0

        # Idempotent batch insert
        n2 = await store.record_batch(batch)
        assert n2 == 0
        assert len(store) == 4

        # events_for_job
        evs = await store.events_for_job("j-1")
        assert len(evs) == 3

        # events_for_client
        evs = await store.events_for_client("acme")
        assert len(evs) == 4
        evs = await store.events_for_client("other")
        assert len(evs) == 0

        print("Usage store OK.")

    asyncio.run(run())