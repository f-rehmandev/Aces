"""
Observability store — spec §40.6.

Two stores side by side:

    IncidentStore     open/ack/resolve/close incidents, dedup by key
    SLOStore          SLO definitions + samples + window evaluation

Both have in-memory and Supabase implementations. The in-memory
versions are used in tests; the Supabase versions satisfy the same
protocols.

Design:
    - Incidents dedupe on (client_id, dedup_key) while open.
    - SLO samples are append-only and idempotent on sample_id.
    - Evaluation is deterministic: given samples in a window and a
      target, decide MEETING / AT_RISK / BREACHED / INSUFFICIENT_DATA.
    - Reads never raise; writes raise on failure (losing incident
      state or SLO samples is a real problem).
"""
from __future__ import annotations

import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from src.observability.types import (
    Incident,
    IncidentSeverity,
    IncidentStatus,
    SLODirection,
    SLOEvaluation,
    SLOSample,
    SLOStatus,
    SLO,
    is_terminal_status,
)


logger = logging.getLogger("observability.store")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# IncidentStore protocol
# ---------------------------------------------------------------------------

class IncidentStore(Protocol):
    async def open_or_touch(self, incident: Incident) -> str:
        # Try to find an existing OPEN incident with the same dedup_key
        if incident.dedup_key:
            for existing in self._by_id.values():
                if (
                    existing.dedup_key == incident.dedup_key
                    and existing.client_id == incident.client_id
                    and not is_terminal_status(existing.status)
                ):
                    # --- dedup hit ---
                    existing.occurrence_count += 1
                    existing.updated_at = _utc_now_iso()

                    # Escalate severity only. Never downgrade.
                    if incident.severity.rank > existing.severity.rank:
                        existing.severity = incident.severity

                    # Merge affected lists, dedup entries.
                    for j in incident.affected_jobs:
                        if j not in existing.affected_jobs:
                            existing.affected_jobs.append(j)
                    for d in incident.affected_domains:
                        if d not in existing.affected_domains:
                            existing.affected_domains.append(d)

                    return existing.incident_id

        # No open incident with this key → create a new one
        self._by_id[incident.incident_id] = incident
        return incident.incident_id


# ---------------------------------------------------------------------------
# In-memory incident store
# ---------------------------------------------------------------------------

class InMemoryIncidentStore:
    def __init__(self):
        self._by_id: dict[str, Incident] = {}

    async def open_or_touch(self, incident: Incident) -> str:
        # Try to find an existing OPEN incident with the same dedup_key
        if incident.dedup_key:
            for existing in self._by_id.values():
                if (
                    existing.dedup_key == incident.dedup_key
                    and existing.client_id == incident.client_id
                    and not is_terminal_status(existing.status)
                ):
                    # --- dedup hit: touch the existing incident ---
                    existing.occurrence_count += 1
                    existing.updated_at = _utc_now_iso()

                    # Escalate severity only. Never downgrade.
                    # rank: INFO=0 < WARNING=1 < CRITICAL=2
                    if incident.severity.rank > existing.severity.rank:
                        existing.severity = incident.severity

                    # Merge affected lists, dedup entries.
                    for j in incident.affected_jobs:
                        if j not in existing.affected_jobs:
                            existing.affected_jobs.append(j)
                    for d in incident.affected_domains:
                        if d not in existing.affected_domains:
                            existing.affected_domains.append(d)

                    return existing.incident_id

        # No open incident with this key → create a new one
        self._by_id[incident.incident_id] = incident
        return incident.incident_id

    async def get(self, incident_id: str) -> Optional[Incident]:
        return self._by_id.get(incident_id)

    async def list_open(self, client_id: str = "") -> list[Incident]:
        out = [
            i for i in self._by_id.values()
            if not is_terminal_status(i.status)
            and (not client_id or i.client_id == client_id)
        ]
        out.sort(key=lambda i: i.started_at, reverse=True)
        return out

    async def list_recent(
        self, client_id: str = "", limit: int = 100,
    ) -> list[Incident]:
        out = [
            i for i in self._by_id.values()
            if not client_id or i.client_id == client_id
        ]
        out.sort(key=lambda i: i.started_at, reverse=True)
        return out[:limit]

    async def update_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        *,
        acknowledged_by: str = "",
        mitigation: str = "",
        root_cause_note: str = "",
    ) -> bool:
        inc = self._by_id.get(incident_id)
        if inc is None:
            return False
        inc.status = status
        if acknowledged_by:
            inc.acknowledged_by = acknowledged_by
            inc.acknowledged_at = _utc_now_iso()
        if mitigation:
            inc.mitigation = mitigation
        if root_cause_note:
            inc.root_cause_note = root_cause_note
        if status == IncidentStatus.RESOLVED and not inc.resolved_at:
            inc.resolved_at = _utc_now_iso()
        if status == IncidentStatus.CLOSED and not inc.closed_at:
            inc.closed_at = _utc_now_iso()
        inc.updated_at = _utc_now_iso()
        return True

# ---------------------------------------------------------------------------
# Supabase incident store
# ---------------------------------------------------------------------------

class SupabaseIncidentStore:
    def __init__(self, client_factory=None):
        if client_factory is None:
            from src.storage.db import get_client
            client_factory = get_client
        self._client_factory = client_factory

    def _client(self):
        return self._client_factory()

    async def open_or_touch(self, incident: Incident) -> str:
        client = self._client()
        resp = client.rpc(
            "incident_open_or_touch",
            {"p_incident": incident.to_dict()},
        ).execute()
        data = getattr(resp, "data", None)
        if isinstance(data, str):
            return data
        if isinstance(data, list) and data:
            return str(data[0])
        return incident.incident_id

    async def get(self, incident_id: str) -> Optional[Incident]:
        client = self._client()
        resp = (
            client.table("incidents")
            .select("*")
            .eq("incident_id", incident_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        if not rows:
            return None
        return _row_to_incident(rows[0])

    async def list_open(self, client_id: str = "") -> list[Incident]:
        client = self._client()
        q = (
            client.table("incidents")
            .select("*")
            .not_.in_("status", ["resolved", "closed"])
            .order("started_at", desc=True)
        )
        if client_id:
            q = q.eq("client_id", client_id)
        resp = q.execute()
        rows = getattr(resp, "data", None) or []
        return [_row_to_incident(r) for r in rows]

    async def list_recent(
        self, client_id: str = "", limit: int = 100,
    ) -> list[Incident]:
        client = self._client()
        q = (
            client.table("incidents")
            .select("*")
            .order("started_at", desc=True)
            .limit(limit)
        )
        if client_id:
            q = q.eq("client_id", client_id)
        resp = q.execute()
        rows = getattr(resp, "data", None) or []
        return [_row_to_incident(r) for r in rows]

    async def update_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        *,
        acknowledged_by: str = "",
        mitigation: str = "",
        root_cause_note: str = "",
    ) -> bool:
        client = self._client()
        patch: dict = {
            "status": status.value,
            "updated_at": _utc_now_iso(),
        }
        if acknowledged_by:
            patch["acknowledged_by"] = acknowledged_by
            patch["acknowledged_at"] = _utc_now_iso()
        if mitigation:
            patch["mitigation"] = mitigation
        if root_cause_note:
            patch["root_cause_note"] = root_cause_note
        if status == IncidentStatus.RESOLVED:
            patch["resolved_at"] = _utc_now_iso()
        if status == IncidentStatus.CLOSED:
            patch["closed_at"] = _utc_now_iso()
        resp = (
            client.table("incidents")
            .update(patch)
            .eq("incident_id", incident_id)
            .execute()
        )
        return bool(getattr(resp, "data", None))


# ---------------------------------------------------------------------------
# SLOStore protocol
# ---------------------------------------------------------------------------

class SLOStore(Protocol):
    async def save_slo(self, slo: SLO) -> bool: ...
    async def get_slo(self, slo_id: str) -> Optional[SLO]: ...
    async def list_slos(
        self, scope: str = "", scope_id: str = "",
    ) -> list[SLO]: ...
    async def record_samples(self, samples: list[SLOSample]) -> int: ...
    async def samples_in_window(
        self, slo_id: str, since: str,
    ) -> list[SLOSample]: ...
    async def evaluate(
        self, slo: SLO, client_id: str = "",
    ) -> SLOEvaluation: ...


# ---------------------------------------------------------------------------
# In-memory SLO store
# ---------------------------------------------------------------------------

class InMemorySLOStore:
    def __init__(self):
        self._slos: dict[str, SLO] = {}
        self._samples: list[SLOSample] = []
        self._sample_ids: set[str] = set()

    async def save_slo(self, slo: SLO) -> bool:
        self._slos[slo.slo_id] = slo
        return True

    async def get_slo(self, slo_id: str) -> Optional[SLO]:
        return self._slos.get(slo_id)

    async def list_slos(
        self, scope: str = "", scope_id: str = "",
    ) -> list[SLO]:
        out = []
        for slo in self._slos.values():
            if scope and slo.scope != scope:
                continue
            if scope_id and slo.scope_id != scope_id:
                continue
            out.append(slo)
        return out

    async def record_samples(self, samples: list[SLOSample]) -> int:
        n = 0
        for s in samples:
            if s.sample_id in self._sample_ids:
                continue
            self._sample_ids.add(s.sample_id)
            self._samples.append(s)
            n += 1
        return n

    async def samples_in_window(
        self, slo_id: str, since: str,
    ) -> list[SLOSample]:
        out = [
            s for s in self._samples
            if s.slo_id == slo_id and (not since or s.sampled_at >= since)
        ]
        out.sort(key=lambda s: s.sampled_at)
        return out

    async def evaluate(
        self, slo: SLO, client_id: str = "",
    ) -> SLOEvaluation:
        since = (
            _utc_now() - timedelta(days=slo.window_days)
        ).isoformat(timespec="milliseconds")
        samples = await self.samples_in_window(slo.slo_id, since)
        return _evaluate_slo(slo, samples, since)


# ---------------------------------------------------------------------------
# Supabase SLO store
# ---------------------------------------------------------------------------

class SupabaseSLOStore:
    def __init__(self, client_factory=None):
        if client_factory is None:
            from src.storage.db import get_client
            client_factory = get_client
        self._client_factory = client_factory

    def _client(self):
        return self._client_factory()

    async def save_slo(self, slo: SLO) -> bool:
        client = self._client()
        client.table("slo_definitions").upsert(
            slo.to_dict(), on_conflict="slo_id",
        ).execute()
        return True

    async def get_slo(self, slo_id: str) -> Optional[SLO]:
        client = self._client()
        resp = (
            client.table("slo_definitions")
            .select("*").eq("slo_id", slo_id).limit(1).execute()
        )
        rows = getattr(resp, "data", None) or []
        return SLO.from_dict(rows[0]) if rows else None

    async def list_slos(
        self, scope: str = "", scope_id: str = "",
    ) -> list[SLO]:
        client = self._client()
        q = client.table("slo_definitions").select("*").eq("enabled", True)
        if scope:
            q = q.eq("scope", scope)
        if scope_id:
            q = q.eq("scope_id", scope_id)
        resp = q.execute()
        rows = getattr(resp, "data", None) or []
        return [SLO.from_dict(r) for r in rows]

    async def record_samples(self, samples: list[SLOSample]) -> int:
        if not samples:
            return 0
        client = self._client()
        payload = [s.to_dict() for s in samples]
        resp = client.rpc(
            "slo_samples_batch", {"p_samples": payload},
        ).execute()
        data = getattr(resp, "data", None)
        if isinstance(data, int):
            return data
        if isinstance(data, list) and data and isinstance(data[0], int):
            return data[0]
        return len(samples)

    async def samples_in_window(
        self, slo_id: str, since: str,
    ) -> list[SLOSample]:
        client = self._client()
        q = (
            client.table("slo_samples")
            .select("*")
            .eq("slo_id", slo_id)
            .order("sampled_at")
        )
        if since:
            q = q.gte("sampled_at", since)
        resp = q.execute()
        rows = getattr(resp, "data", None) or []
        return [_row_to_sample(r) for r in rows]

    async def evaluate(
        self, slo: SLO, client_id: str = "",
    ) -> SLOEvaluation:
        since = (
            _utc_now() - timedelta(days=slo.window_days)
        ).isoformat(timespec="milliseconds")
        samples = await self.samples_in_window(slo.slo_id, since)
        return _evaluate_slo(slo, samples, since)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_AT_RISK_MARGIN = 0.10   # within 10% of target is "at risk"


def _evaluate_slo(
    slo: SLO,
    samples: list[SLOSample],
    window_start: str,
) -> SLOEvaluation:
    window_end = _utc_now_iso()

    if not slo.enabled:
        return SLOEvaluation(
            slo_id=slo.slo_id, slo_name=slo.name,
            status=SLOStatus.DISABLED,
            target=slo.target, direction=slo.direction.value,
            window_start=window_start, window_end=window_end,
            reason="disabled",
        )

    if len(samples) < slo.min_samples:
        return SLOEvaluation(
            slo_id=slo.slo_id, slo_name=slo.name,
            status=SLOStatus.INSUFFICIENT_DATA,
            target=slo.target, direction=slo.direction.value,
            sample_count=len(samples),
            window_start=window_start, window_end=window_end,
            reason=(
                f"{len(samples)} samples < min_samples "
                f"{slo.min_samples}"
            ),
        )

    values = [s.value for s in samples]
    current = statistics.fmean(values)

    # Margin: positive = better than target
    if slo.direction == SLODirection.AT_LEAST:
        margin = current - slo.target
    else:
        margin = slo.target - current

    if margin >= 0:
        status = SLOStatus.MEETING
    elif abs(margin) <= abs(slo.target) * _AT_RISK_MARGIN:
        status = SLOStatus.AT_RISK
    else:
        status = SLOStatus.BREACHED

    reason = ""
    if status == SLOStatus.MEETING:
        reason = f"value {current:.4f} meets target {slo.target}"
    elif status == SLOStatus.AT_RISK:
        reason = (
            f"value {current:.4f} is within "
            f"{_AT_RISK_MARGIN:.0%} of target {slo.target}"
        )
    else:
        reason = (
            f"value {current:.4f} breaches target "
            f"{slo.target} (direction={slo.direction.value})"
        )

    return SLOEvaluation(
        slo_id=slo.slo_id, slo_name=slo.name,
        status=status,
        current_value=round(current, 6),
        target=slo.target,
        direction=slo.direction.value,
        sample_count=len(samples),
        window_start=window_start, window_end=window_end,
        margin=round(margin, 6),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Row translation
# ---------------------------------------------------------------------------

def _row_to_incident(row: dict) -> Incident:
    d = dict(row)
    if not isinstance(d.get("affected_jobs"), list):
        d["affected_jobs"] = []
    if not isinstance(d.get("affected_domains"), list):
        d["affected_domains"] = []
    for k in (
        "linked_alert_ids", "linked_healing_ids", "linked_provider_events",
    ):
        if not isinstance(d.get(k), list):
            d[k] = []
    if not isinstance(d.get("metadata"), dict):
        d["metadata"] = {}
    # Timestamps come back as ISO strings
    for k in (
        "started_at", "acknowledged_at", "resolved_at", "closed_at",
        "updated_at",
    ):
        v = d.get(k)
        if v is not None and not isinstance(v, str):
            d[k] = str(v)
    return Incident.from_dict(d)


def _row_to_sample(row: dict) -> SLOSample:
    d = dict(row)
    try:
        d["value"] = float(d.get("value", 0))
    except (ValueError, TypeError):
        d["value"] = 0.0
    if not isinstance(d.get("metadata"), dict):
        d["metadata"] = {}
    v = d.get("sampled_at")
    if v is not None and not isinstance(v, str):
        d["sampled_at"] = str(v)
    return SLOSample.from_dict(d)


# ---------------------------------------------------------------------------
# Smoke test (in-memory only — no network)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def run():
        # ---- Incidents ----
        store = InMemoryIncidentStore()

        inc = Incident(
            title="provider down",
            dedup_key="provider-outage",
            severity=IncidentSeverity.WARNING,
            client_id="acme",
        )
        iid = await store.open_or_touch(inc)
        assert iid == inc.incident_id

        # Dedup by key
        inc2 = Incident(
            title="provider down",
            dedup_key="provider-outage",
            severity=IncidentSeverity.CRITICAL,
            client_id="acme",
        )
        iid2 = await store.open_or_touch(inc2)
        assert iid2 == iid   # same incident
        got = await store.get(iid)
        assert got.occurrence_count == 2
        # Severity was escalated
        assert got.severity == IncidentSeverity.CRITICAL

        # Different client → different incident
        inc3 = Incident(
            title="provider down", dedup_key="provider-outage",
            client_id="other",
        )
        iid3 = await store.open_or_touch(inc3)
        assert iid3 != iid

        # Different key → different incident
        inc4 = Incident(title="x", dedup_key="other-key", client_id="acme")
        iid4 = await store.open_or_touch(inc4)
        assert iid4 != iid

        # list_open filters terminal
        opens = await store.list_open(client_id="acme")
        assert len(opens) == 2   # inc + inc4

        # Resolve one
        assert await store.update_status(iid, IncidentStatus.RESOLVED)
        opens = await store.list_open(client_id="acme")
        assert len(opens) == 1

        # After resolution, dedup_key frees up — a new incident is created
        inc5 = Incident(title="provider down", dedup_key="provider-outage",
                        client_id="acme")
        iid5 = await store.open_or_touch(inc5)
        assert iid5 != iid

        # list_recent includes everything
        recent = await store.list_recent(client_id="acme")
        assert len(recent) == 3

        # update missing returns False
        assert not await store.update_status(
            "does-not-exist", IncidentStatus.CLOSED,
        )

        # ---- SLOs ----
        slo_store = InMemorySLOStore()

        slo = SLO(
            name="job success rate",
            metric="job_success_rate",
            unit="ratio",
            direction=SLODirection.AT_LEAST,
            target=0.95,
            window_days=7,
            min_samples=3,
        )
        await slo_store.save_slo(slo)
        assert (await slo_store.get_slo(slo.slo_id)) is not None

        # Insufficient data
        ev = await slo_store.evaluate(slo)
        assert ev.status == SLOStatus.INSUFFICIENT_DATA

        # Meeting
        samples = [
            SLOSample(slo_id=slo.slo_id, value=v, client_id="acme")
            for v in (0.98, 0.97, 0.99)
        ]
        n = await slo_store.record_samples(samples)
        assert n == 3
        ev = await slo_store.evaluate(slo)
        assert ev.status == SLOStatus.MEETING, ev.reason
        assert ev.current_value == 0.98
        assert ev.margin > 0

        # At risk: within 10% of target
        slo2 = SLO(
            name="queue latency", metric="queue_start_latency",
            unit="seconds", direction=SLODirection.AT_MOST,
            target=30.0, window_days=1, min_samples=2,
        )
        await slo_store.save_slo(slo2)
        # value 32 → margin = 30 - 32 = -2 → at_risk since |2| <= |30|*0.1=3
        await slo_store.record_samples([
            SLOSample(slo_id=slo2.slo_id, value=32.0),
            SLOSample(slo_id=slo2.slo_id, value=32.0),
        ])
        ev = await slo_store.evaluate(slo2)
        assert ev.status == SLOStatus.AT_RISK, ev.reason

        # Breached: well off target
        slo3 = SLO(
            name="error rate", metric="error_rate",
            unit="ratio", direction=SLODirection.AT_MOST,
            target=0.05, window_days=1, min_samples=2,
        )
        await slo_store.save_slo(slo3)
        await slo_store.record_samples([
            SLOSample(slo_id=slo3.slo_id, value=0.20),
            SLOSample(slo_id=slo3.slo_id, value=0.20),
        ])
        ev = await slo_store.evaluate(slo3)
        assert ev.status == SLOStatus.BREACHED, ev.reason

        # Disabled
        slo4 = SLO(name="disabled", metric="x", enabled=False)
        await slo_store.save_slo(slo4)
        await slo_store.record_samples([
            SLOSample(slo_id=slo4.slo_id, value=0.0),
            SLOSample(slo_id=slo4.slo_id, value=0.0),
        ])
        ev = await slo_store.evaluate(slo4)
        assert ev.status == SLOStatus.DISABLED

        # Idempotent sample insert
        n2 = await slo_store.record_samples(samples)
        assert n2 == 0

        # list_slos
        all_slos = await slo_store.list_slos()
        assert len(all_slos) == 4

        print("Observability store OK.")

    asyncio.run(run())