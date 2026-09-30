"""
Persistent lead memory — spec: discovery vs monitoring mode.

Remembers which Google Maps `place_id`s each client has already seen, so
a "discovery mode" run can return only genuinely-new leads, and a
"monitoring mode" run can mark every lead as NEW or EXISTING.

Backend-agnostic: `fetch_fn` and `write_fn` are injected. Production uses
Supabase; tests use in-memory fakes.

Design notes:
    - Keyed on (client_id, place_id). The same lead seen by two clients
      is remembered independently for each.
    - `query_key` records which search first produced the lead, for
      debugging / analytics only.
    - Never raises on read failure — a broken DB must not break discovery.
      Falls back to "everything is new" so the user still gets data.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger("lead_memory")


# ---------------------------------------------------------------------------
# Diff-status constants written onto annotated records
# ---------------------------------------------------------------------------

STATUS_NEW = "NEW"
STATUS_EXISTING = "EXISTING"
STATUS_EXISTING_UNKNOWN = "EXISTING_UNKNOWN"   # no place_id, can't tell


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class LeadMemoryStore:
    """
    Records and recalls per-client lead identity.

    `fetch_fn(client_id, place_ids) -> set[str]`
        Returns the subset of `place_ids` that this client has seen before.

    `write_fn(client_id, place_ids, query_key) -> None`
        Records (or refreshes) these place_ids for this client.
    """

    def __init__(
        self,
        fetch_fn: Callable[[str, list[str]], set[str]],
        write_fn: Callable[[str, list[str], str], None],
    ):
        self._fetch = fetch_fn
        self._write = write_fn

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def known_place_ids(
        self,
        client_id: str,
        place_ids: list[str],
    ) -> set[str]:
        """Return the subset of `place_ids` already seen by this client."""
        clean = [p for p in (place_ids or []) if p]
        if not clean:
            return set()
        try:
            return set(self._fetch(client_id, clean))
        except Exception as e:
            logger.warning(
                f"lead memory read failed for client {client_id!r}: "
                f"{type(e).__name__}: {e}"
            )
            return set()   # graceful: treat everything as unknown

    def is_known(self, client_id: str, place_id: str) -> bool:
        if not place_id:
            return False
        return place_id in self.known_place_ids(client_id, [place_id])

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def remember_batch(
        self,
        client_id: str,
        place_ids: list[str],
        query_key: str = "",
    ) -> None:
        clean = [p for p in (place_ids or []) if p]
        if not clean:
            return
        try:
            self._write(client_id, clean, query_key or "")
        except Exception as e:
            logger.warning(
                f"lead memory write failed for client {client_id!r}: "
                f"{type(e).__name__}: {e}"
            )

    def remember(
        self,
        client_id: str,
        place_id: str,
        query_key: str = "",
    ) -> None:
        if place_id:
            self.remember_batch(client_id, [place_id], query_key=query_key)

    # ------------------------------------------------------------------
    # Discovery mode — return only genuinely new leads
    # ------------------------------------------------------------------
    def filter_new(
        self,
        client_id: str,
        place_ids: list[str],
    ) -> list[str]:
        """
        Return only the place_ids this client has never seen before.
        Preserves the caller's original order.
        """
        clean = [p for p in (place_ids or []) if p]
        if not clean:
            return []
        known = self.known_place_ids(client_id, clean)
        return [p for p in clean if p not in known]

    # ------------------------------------------------------------------
    # Monitoring mode — annotate each record as NEW or EXISTING
    # ------------------------------------------------------------------
    def annotate_records(
        self,
        client_id: str,
        records: list[dict],
        place_id_field: str = "place_id",
    ) -> tuple[list[dict], int, int]:
        """
        Mark each record's `diff_status` as NEW, EXISTING, or EXISTING_UNKNOWN.

        Returns (annotated_records, new_count, existing_count).

        Records without a place_id are labelled EXISTING_UNKNOWN and counted
        as neither new nor existing — this is honest: we cannot tell whether
        they are duplicates.
        """
        if not records:
            return [], 0, 0

        ids = [str(r.get(place_id_field) or "") for r in records]
        known = self.known_place_ids(client_id, [i for i in ids if i])

        new_count = 0
        existing_count = 0
        annotated: list[dict] = []

        for rec, pid in zip(records, ids):
            rec = dict(rec)   # don't mutate the caller's record
            if not pid:
                rec["diff_status"] = STATUS_EXISTING_UNKNOWN
            elif pid in known:
                rec["diff_status"] = STATUS_EXISTING
                existing_count += 1
            else:
                rec["diff_status"] = STATUS_NEW
                new_count += 1
            annotated.append(rec)

        return annotated, new_count, existing_count


# ---------------------------------------------------------------------------
# Supabase-backed factory
# ---------------------------------------------------------------------------

def build_supabase_store() -> LeadMemoryStore:
    """
    Build a store backed by the `lead_memory` table.
    Any DB failure degrades to "everything is new" — the pipeline keeps
    running and the user still gets data.
    """
    from src.storage.db import get_client

    def fetch(client_id: str, place_ids: list[str]) -> set[str]:
        client = get_client()
        resp = (
            client.table("lead_memory")
            .select("place_id")
            .eq("client_id", client_id)
            .in_("place_id", place_ids)
            .execute()
        )
        data = getattr(resp, "data", None) or []
        return {row["place_id"] for row in data}

    def write(client_id: str, place_ids: list[str], query_key: str) -> None:
        client = get_client()
        client.rpc("upsert_lead_memory", {
            "p_client_id": client_id,
            "p_place_ids": list(place_ids),
            "p_query_key": query_key,
        }).execute()

    return LeadMemoryStore(fetch_fn=fetch, write_fn=write)


# ---------------------------------------------------------------------------
# Smoke test — in-memory only, no network
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows: dict[tuple[str, str], dict] = {}

    def fetch(cid, place_ids):
        return {pid for (c, pid) in rows if c == cid and pid in place_ids}

    def write(cid, place_ids, qk):
        for pid in place_ids:
            if (cid, pid) not in rows:
                rows[(cid, pid)] = {"query_key": qk}

    store = LeadMemoryStore(fetch_fn=fetch, write_fn=write)

    assert store.known_place_ids("acme", ["a", "b", "c"]) == set()
    assert store.filter_new("acme", ["a", "b", "c"]) == ["a", "b", "c"]

    store.remember_batch("acme", ["a", "b"], query_key="pizza in Lahore")
    assert store.known_place_ids("acme", ["a", "b", "c"]) == {"a", "b"}
    assert store.is_known("acme", "a")
    assert not store.is_known("acme", "z")

    assert store.filter_new("acme", ["a", "b", "c", "d"]) == ["c", "d"]

    assert store.known_place_ids("other", ["a"]) == set()

    records = [
        {"place_id": "a", "business_name": "A"},
        {"place_id": "z", "business_name": "Z"},
        {"business_name": "No-ID"},
    ]
    annotated, new_count, existing_count = store.annotate_records("acme", records)
    assert annotated[0]["diff_status"] == "EXISTING"
    assert annotated[1]["diff_status"] == "NEW"
    assert annotated[2]["diff_status"] == "EXISTING_UNKNOWN"
    assert new_count == 1
    assert existing_count == 1

    def boom(cid, pids):
        raise RuntimeError("db down")
    broken = LeadMemoryStore(fetch_fn=boom, write_fn=write)
    assert broken.known_place_ids("acme", ["a"]) == set()
    assert broken.filter_new("acme", ["a"]) == ["a"]

    print("LeadMemoryStore OK.")
