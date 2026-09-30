"""
Supabase-backed durable queue — spec §37.

Implements the same `QueueBackend` protocol as
`InMemoryQueueBackend`. Uses the two Postgres RPCs
(`queue_enqueue`, `queue_lease_next`) for the operations that must be
atomic, and plain single-row updates for the rest.

Design notes:
    - Uses the service-role Supabase client (backend-only).
    - Any Supabase failure raises — the caller decides whether to
      fall back to the in-memory queue or surface the error.
    - Timestamps are stored as UTC. The Python side normalizes to
      ISO-8601 strings on the way in and out.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.queue.types import (
    JobState,
    Priority,
    QueuedJob,
    is_terminal,
)


logger = logging.getLogger("queue.supabase")


def _row_to_job(row: dict) -> QueuedJob:
    """Translate a job_queue row into a QueuedJob."""
    d = dict(row)
    # priority comes back as smallint
    try:
        d["priority"] = Priority(int(d.get("priority", 2)))
    except (ValueError, TypeError):
        d["priority"] = Priority.P2_SCHEDULED

    # state comes back as text
    try:
        d["state"] = JobState(d.get("state", "queued"))
    except ValueError:
        d["state"] = JobState.QUEUED

    # payload → dict (already jsonb-decoded by supabase-py)
    if not isinstance(d.get("payload"), dict):
        d["payload"] = {}

    # Timestamps come back as ISO strings from PostgREST
    for k in ("leased_at", "lease_expires_at", "created_at", "updated_at"):
        v = d.get(k)
        if v is not None and not isinstance(v, str):
            d[k] = str(v)

    # Required fields
    d.setdefault("idempotency_key", "")
    d.setdefault("client_id", "default")
    d.setdefault("label", "")
    d.setdefault("attempts", 0)
    d.setdefault("max_attempts", 3)
    d.setdefault("last_error", "")
    d.setdefault("leased_by", "")

    known = set(QueuedJob.__dataclass_fields__)
    return QueuedJob(**{k: v for k, v in d.items() if k in known})


class SupabaseQueueBackend:
    """
    Durable queue backed by Postgres via Supabase.

    `client_factory` is injectable so tests can supply a fake. In
    production it defaults to `src.storage.db.get_client`.
    """

    def __init__(self, client_factory=None):
        if client_factory is None:
            from src.storage.db import get_client
            client_factory = get_client
        self._client_factory = client_factory

    def _client(self):
        return self._client_factory()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    async def enqueue(self, job: QueuedJob) -> str:
        client = self._client()
        payload = job.to_dict()
        # The RPC uses int priority and str state — to_dict already
        # serializes them that way, so we can pass payload directly.
        resp = client.rpc("queue_enqueue", {"p_job": payload}).execute()
        data = getattr(resp, "data", None)
        # queue_enqueue returns a text job_id.
        if isinstance(data, str):
            return data
        if isinstance(data, list) and data:
            return str(data[0])
        # Fallback: return our own id (matches in-memory semantics)
        return job.job_id

    async def lease_next(
        self, worker_id: str, lease_seconds: int,
    ) -> Optional[QueuedJob]:
        client = self._client()
        resp = client.rpc("queue_lease_next", {
            "p_worker_id": worker_id,
            "p_lease_seconds": int(lease_seconds),
        }).execute()
        data = getattr(resp, "data", None)
        # The RPC returns a jsonb object or null.
        if data is None:
            return None
        if isinstance(data, list):
            if not data:
                return None
            data = data[0]
        if not isinstance(data, dict):
            return None
        return _row_to_job(data)

    async def heartbeat(
        self, job_id: str, worker_id: str, lease_seconds: int,
    ) -> bool:
        client = self._client()
        resp = (
            client.table("job_queue")
            .update({
                "lease_expires_at": _future_iso(lease_seconds),
                "updated_at": _now_iso(),
            })
            .eq("job_id", job_id)
            .eq("leased_by", worker_id)
            .in_("state", ["leased", "running"])
            .execute()
        )
        return bool(getattr(resp, "data", None))

    async def complete(
        self, job_id: str, result: Optional[dict] = None,
    ) -> bool:
        client = self._client()
        payload_patch = None
        if result is not None:
            # Merge _result into the existing payload — read first.
            current = (
                client.table("job_queue")
                .select("payload, state")
                .eq("job_id", job_id)
                .limit(1)
                .execute()
            )
            rows = getattr(current, "data", None) or []
            if not rows:
                return False
            if rows[0].get("state") == "completed":
                return True   # idempotent
            existing_payload = rows[0].get("payload") or {}
            existing_payload["_result"] = result
            payload_patch = existing_payload

        update = {
            "state": "completed",
            "leased_by": "",
            "lease_expires_at": None,
            "updated_at": _now_iso(),
        }
        if payload_patch is not None:
            update["payload"] = payload_patch

        resp = (
            client.table("job_queue")
            .update(update)
            .eq("job_id", job_id)
            .neq("state", "dead_letter")
            .execute()
        )
        return bool(getattr(resp, "data", None))

    async def fail(
        self, job_id: str, error: str, retryable: bool = True,
    ) -> JobState:
        client = self._client()
        # Read the current row to compute attempts + next state
        current = (
            client.table("job_queue")
            .select("attempts, max_attempts, state")
            .eq("job_id", job_id)
            .limit(1)
            .execute()
        )
        rows = getattr(current, "data", None) or []
        if not rows:
            return JobState.DEAD_LETTER
        row = rows[0]
        if is_terminal(JobState(row.get("state", "queued"))):
            return JobState(row["state"])

        attempts = int(row.get("attempts", 0)) + 1
        max_attempts = int(row.get("max_attempts", 3))

        if retryable and attempts < max_attempts:
            next_state = JobState.QUEUED.value
        else:
            next_state = JobState.DEAD_LETTER.value

        client.table("job_queue").update({
            "attempts": attempts,
            "last_error": error,
            "state": next_state,
            "leased_by": "",
            "leased_at": None,
            "lease_expires_at": None,
            "updated_at": _now_iso(),
        }).eq("job_id", job_id).execute()

        return JobState(next_state)

    async def recover_expired_leases(self) -> int:
        """
        Postgres version: run one lease_next with a tiny timeout and
        check if it recovered anything. The RPC itself does the sweep,
        so we call it and count based on how many dead-letters
        appeared... but that's imprecise.

        Cleaner: call the sweep directly via a table update matching
        the RPC's logic.
        """
        client = self._client()
        # SELECT expired leases, then UPDATE each to the recovered state.
        expired = (
            client.table("job_queue")
            .select("job_id, attempts, max_attempts")
            .in_("state", ["leased", "running"])
            .lt("lease_expires_at", _now_iso())
            .execute()
        )
        rows = getattr(expired, "data", None) or []
        if not rows:
            return 0

        for row in rows:
            attempts = int(row.get("attempts", 0)) + 1
            max_attempts = int(row.get("max_attempts", 3))
            next_state = (
                JobState.DEAD_LETTER.value
                if attempts >= max_attempts
                else JobState.QUEUED.value
            )
            last_error = (
                "lease expired before completion"
                if next_state == JobState.DEAD_LETTER.value
                else "lease expired; re-queued"
            )
            client.table("job_queue").update({
                "attempts": attempts,
                "state": next_state,
                "leased_by": "",
                "leased_at": None,
                "lease_expires_at": None,
                "last_error": last_error,
                "updated_at": _now_iso(),
            }).eq("job_id", row["job_id"]).execute()

        return len(rows)

    async def requeue_dead_letter(self, job_id: str) -> bool:
        client = self._client()
        resp = (
            client.table("job_queue")
            .update({
                "state": "queued",
                "attempts": 0,
                "last_error": "",
                "updated_at": _now_iso(),
            })
            .eq("job_id", job_id)
            .eq("state", "dead_letter")
            .execute()
        )
        return bool(getattr(resp, "data", None))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    async def get(self, job_id: str) -> Optional[QueuedJob]:
        client = self._client()
        resp = (
            client.table("job_queue")
            .select("*")
            .eq("job_id", job_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        if not rows:
            return None
        return _row_to_job(rows[0])

    async def stats(self) -> dict:
        client = self._client()
        resp = client.table("job_queue").select("state, priority").execute()
        rows = getattr(resp, "data", None) or []

        by_state: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        for r in rows:
            s = r.get("state", "queued")
            by_state[s] = by_state.get(s, 0) + 1
            try:
                p = Priority(int(r.get("priority", 2))).name
            except (ValueError, TypeError):
                p = "P2_SCHEDULED"
            by_priority[p] = by_priority.get(p, 0) + 1

        return {
            "total": len(rows),
            "by_state": by_state,
            "by_priority": by_priority,
        }

    async def dead_letters(self, limit: int = 100) -> list[QueuedJob]:
        client = self._client()
        resp = (
            client.table("job_queue")
            .select("*")
            .eq("state", "dead_letter")
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        return [_row_to_job(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _future_iso(seconds: int) -> str:
    from datetime import timedelta
    return (
        datetime.now(timezone.utc) + timedelta(seconds=seconds)
    ).isoformat(timespec="milliseconds")