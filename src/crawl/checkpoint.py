"""
Checkpoint persistence — spec §14.3B.

Stores and retrieves CrawlCheckpoint objects so an interrupted crawl can
resume from the last successful page instead of starting over.

Backend-agnostic: `fetch_fn`, `write_fn`, and `clear_fn` are injected.
Production uses Supabase; tests use in-memory fakes.

Design notes:
    - Checkpoints are append-only. Each save writes a new row keyed by
      `checkpoint_id`; the previous ones stay for audit.
    - `load_latest` picks the most recent by `created_at`.
    - Reads and writes degrade gracefully: if the backend is unreachable,
      we log and continue. A crawl losing its checkpoint is far better
      than a crawl crashing on checkpoint I/O.
    - `clear_for_task` is what a successful run calls to clean up its own
      checkpoints — no stale resumption state lingers.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from src.crawl.types import CrawlCheckpoint


logger = logging.getLogger("crawl.checkpoint")


class CheckpointStore:
    """
    Read/write/clear checkpoints for crawl runs.

    Injected collaborators:
        fetch_fn(client_id, task_id) -> dict | None
            Return the JSON-serialized latest checkpoint, or None.
        write_fn(checkpoint_dict) -> None
            Persist one checkpoint snapshot.
        clear_fn(client_id, task_id) -> int
            Delete all checkpoints for a task. Return rows removed.
            Optional — defaults to a no-op returning 0.
    """

    def __init__(
        self,
        fetch_fn: Callable[[str, str], Optional[dict]],
        write_fn: Callable[[dict], None],
        clear_fn: Optional[Callable[[str, str], int]] = None,
    ):
        self._fetch = fetch_fn
        self._write = write_fn
        self._clear = clear_fn or (lambda cid, tid: 0)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def save(self, checkpoint: CrawlCheckpoint) -> bool:
        """
        Persist a checkpoint. Returns True on success, False on failure.
        Never raises — checkpoint I/O must not kill a running crawl.
        """
        try:
            self._write(checkpoint.to_dict())
            return True
        except Exception as e:
            logger.warning(
                f"checkpoint save failed for task {checkpoint.task_id!r}: "
                f"{type(e).__name__}: {e}"
            )
            return False

    def clear_for_task(self, client_id: str, task_id: str) -> int:
        """Delete all checkpoints for a task. Returns rows removed (or 0 on failure)."""
        try:
            return int(self._clear(client_id, task_id) or 0)
        except Exception as e:
            logger.warning(
                f"checkpoint clear failed for task {task_id!r}: "
                f"{type(e).__name__}: {e}"
            )
            return 0

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def load_latest(
        self,
        client_id: str,
        task_id: str,
    ) -> Optional[CrawlCheckpoint]:
        """
        Return the most recent checkpoint for a task, or None if there
        isn't one / the read fails.
        """
        if not task_id:
            return None
        try:
            raw = self._fetch(client_id, task_id)
        except Exception as e:
            logger.warning(
                f"checkpoint load failed for task {task_id!r}: "
                f"{type(e).__name__}: {e}"
            )
            return None

        if not raw:
            return None
        if not isinstance(raw, dict):
            logger.warning(
                f"checkpoint load returned non-dict for task {task_id!r}"
            )
            return None

        try:
            return CrawlCheckpoint.from_dict(raw)
        except Exception as e:
            logger.warning(
                f"checkpoint parse failed for task {task_id!r}: "
                f"{type(e).__name__}: {e}"
            )
            return None

    def has_checkpoint(self, client_id: str, task_id: str) -> bool:
        return self.load_latest(client_id, task_id) is not None


# ---------------------------------------------------------------------------
# Supabase-backed factory
# ---------------------------------------------------------------------------

def build_supabase_store() -> CheckpointStore:
    """
    Build a CheckpointStore backed by the `crawl_checkpoints` table.

    Uses the service-role client because checkpoints are backend-internal,
    not user-facing. All reads/writes are best-effort; the store itself
    swallows and logs exceptions so the crawl loop doesn't have to.
    """
    from src.storage.db import get_client

    def fetch(client_id: str, task_id: str) -> Optional[dict]:
        client = get_client()
        resp = (
            client.table("crawl_checkpoints")
            .select("data")
            .eq("client_id", client_id)
            .eq("task_id", task_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        if not rows:
            return None
        return rows[0].get("data")

    def write(checkpoint_dict: dict) -> None:
        client = get_client()
        client.table("crawl_checkpoints").insert({
            "checkpoint_id": checkpoint_dict["checkpoint_id"],
            "client_id": checkpoint_dict["client_id"],
            "task_id": checkpoint_dict["task_id"],
            "created_at": checkpoint_dict["created_at"],
            "data": checkpoint_dict,
        }).execute()

    def clear(client_id: str, task_id: str) -> int:
        client = get_client()
        # Use the helper RPC so the deletion is a single round-trip and
        # we get the count back atomically.
        resp = client.rpc("clear_crawl_checkpoints", {
            "p_client_id": client_id,
            "p_task_id": task_id,
        }).execute()
        data = getattr(resp, "data", None)
        if isinstance(data, int):
            return data
        return 0

    return CheckpointStore(fetch_fn=fetch, write_fn=write, clear_fn=clear)


# ---------------------------------------------------------------------------
# Smoke test — in-memory only, no network
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple in-memory backend
    rows: dict[str, dict] = {}
    cleared: list[tuple[str, str]] = []

    def fetch(cid, tid):
        key = f"{cid}::{tid}"
        return rows.get(key)

    def write(cp_dict):
        key = f"{cp_dict['client_id']}::{cp_dict['task_id']}"
        rows[key] = cp_dict

    def clear(cid, tid):
        key = f"{cid}::{tid}"
        existed = 1 if key in rows else 0
        rows.pop(key, None)
        cleared.append((cid, tid))
        return existed

    store = CheckpointStore(fetch_fn=fetch, write_fn=write, clear_fn=clear)

    # Empty
    assert store.load_latest("acme", "t-1") is None
    assert not store.has_checkpoint("acme", "t-1")

    # Save & load
    cp = CrawlCheckpoint(client_id="acme", task_id="t-1")
    assert store.save(cp) is True
    loaded = store.load_latest("acme", "t-1")
    assert loaded is not None
    assert loaded.task_id == "t-1"
    assert loaded.checkpoint_id == cp.checkpoint_id

    # Has
    assert store.has_checkpoint("acme", "t-1")

    # Different task isolated
    assert store.load_latest("acme", "t-2") is None
    assert store.load_latest("other", "t-1") is None

    # Clear
    removed = store.clear_for_task("acme", "t-1")
    assert removed == 1
    assert store.load_latest("acme", "t-1") is None

    # Graceful degradation — read failure
    def boom_fetch(cid, tid):
        raise RuntimeError("db down")
    broken = CheckpointStore(fetch_fn=boom_fetch, write_fn=write)
    assert broken.load_latest("acme", "x") is None
    assert not broken.has_checkpoint("acme", "x")

    # Graceful degradation — write failure
    def boom_write(cp_dict):
        raise RuntimeError("db down")
    broken2 = CheckpointStore(fetch_fn=fetch, write_fn=boom_write)
    assert broken2.save(cp) is False

    # Empty task_id short-circuits
    assert store.load_latest("acme", "") is None

    print("CheckpointStore OK.")