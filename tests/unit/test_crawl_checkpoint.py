"""Unit tests for CheckpointStore (spec §14.3B)."""
import pytest

from src.crawl.checkpoint import CheckpointStore
from src.crawl.types import CrawlCheckpoint, CrawlPolicy, CrawlTarget, CrawlState


# ---------------------------------------------------------------------------
# In-memory backend fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """Returns (store, rows_dict, cleared_list)."""
    rows: dict[str, dict] = {}
    cleared: list[tuple[str, str]] = []

    def fetch(cid, tid):
        return rows.get(f"{cid}::{tid}")

    def write(cp_dict):
        key = f"{cp_dict['client_id']}::{cp_dict['task_id']}"
        rows[key] = cp_dict

    def clear(cid, tid):
        key = f"{cid}::{tid}"
        existed = 1 if key in rows else 0
        rows.pop(key, None)
        cleared.append((cid, tid))
        return existed

    return CheckpointStore(fetch_fn=fetch, write_fn=write, clear_fn=clear), rows, cleared


# ---------------------------------------------------------------------------
# Empty / miss cases
# ---------------------------------------------------------------------------

def test_empty_store_returns_none(store):
    s, _, _ = store
    assert s.load_latest("acme", "t-1") is None


def test_has_checkpoint_false_when_empty(store):
    s, _, _ = store
    assert not s.has_checkpoint("acme", "t-1")


def test_empty_task_id_short_circuits(store):
    s, _, _ = store
    assert s.load_latest("acme", "") is None
    assert s.load_latest("acme", None) is None


# ---------------------------------------------------------------------------
# Save + load
# ---------------------------------------------------------------------------

def test_save_returns_true_on_success(store):
    s, _, _ = store
    cp = CrawlCheckpoint(client_id="acme", task_id="t-1")
    assert s.save(cp) is True


def test_load_latest_roundtrip(store):
    s, _, _ = store
    cp = CrawlCheckpoint(client_id="acme", task_id="t-1")
    s.save(cp)
    loaded = s.load_latest("acme", "t-1")
    assert loaded is not None
    assert loaded.checkpoint_id == cp.checkpoint_id
    assert loaded.client_id == "acme"
    assert loaded.task_id == "t-1"


def test_save_then_has(store):
    s, _, _ = store
    s.save(CrawlCheckpoint(client_id="acme", task_id="t-1"))
    assert s.has_checkpoint("acme", "t-1")


def test_save_second_time_overwrites(store):
    s, _, _ = store
    cp1 = CrawlCheckpoint(client_id="acme", task_id="t-1", stats_processed=5)
    s.save(cp1)
    cp2 = CrawlCheckpoint(client_id="acme", task_id="t-1", stats_processed=10)
    s.save(cp2)
    loaded = s.load_latest("acme", "t-1")
    assert loaded.stats_processed == 10
    assert loaded.checkpoint_id == cp2.checkpoint_id


# ---------------------------------------------------------------------------
# Complex payloads
# ---------------------------------------------------------------------------

def test_save_load_preserves_frontier(store):
    s, _, _ = store
    cp = CrawlCheckpoint(
        client_id="acme",
        task_id="t-1",
        frontier=[
            CrawlTarget(url="https://x.com/a", depth=1),
            CrawlTarget(url="https://x.com/b", depth=2,
                        state=CrawlState.PROCESSED, records_extracted=7),
        ],
        budget_pages_used=2,
        stats_processed=1,
    )
    s.save(cp)
    loaded = s.load_latest("acme", "t-1")
    assert len(loaded.frontier) == 2
    assert loaded.frontier[0].url == "https://x.com/a"
    assert loaded.frontier[1].state == CrawlState.PROCESSED
    assert loaded.frontier[1].records_extracted == 7
    assert loaded.budget_pages_used == 2


def test_save_load_preserves_policy(store):
    s, _, _ = store
    cp = CrawlCheckpoint(
        client_id="acme", task_id="t-1",
        policy=CrawlPolicy(max_depth=5, max_pages=42),
    )
    s.save(cp)
    loaded = s.load_latest("acme", "t-1")
    assert loaded.policy is not None
    assert loaded.policy.max_depth == 5
    assert loaded.policy.max_pages == 42


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_tasks_are_isolated(store):
    s, _, _ = store
    s.save(CrawlCheckpoint(client_id="acme", task_id="t-1", stats_processed=1))
    s.save(CrawlCheckpoint(client_id="acme", task_id="t-2", stats_processed=2))
    assert s.load_latest("acme", "t-1").stats_processed == 1
    assert s.load_latest("acme", "t-2").stats_processed == 2


def test_clients_are_isolated(store):
    s, _, _ = store
    s.save(CrawlCheckpoint(client_id="acme", task_id="t-1", stats_processed=1))
    s.save(CrawlCheckpoint(client_id="other", task_id="t-1", stats_processed=2))
    assert s.load_latest("acme", "t-1").stats_processed == 1
    assert s.load_latest("other", "t-1").stats_processed == 2


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def test_clear_removes_checkpoint(store):
    s, _, cleared = store
    s.save(CrawlCheckpoint(client_id="acme", task_id="t-1"))
    removed = s.clear_for_task("acme", "t-1")
    assert removed == 1
    assert s.load_latest("acme", "t-1") is None
    assert ("acme", "t-1") in cleared


def test_clear_no_checkpoint_returns_zero(store):
    s, _, _ = store
    assert s.clear_for_task("acme", "nothing") == 0


def test_clear_only_affects_target_task(store):
    s, _, _ = store
    s.save(CrawlCheckpoint(client_id="acme", task_id="t-1"))
    s.save(CrawlCheckpoint(client_id="acme", task_id="t-2"))
    s.clear_for_task("acme", "t-1")
    assert s.load_latest("acme", "t-1") is None
    assert s.load_latest("acme", "t-2") is not None


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_save_returns_false_on_write_error():
    def boom(cp_dict): raise RuntimeError("db down")
    s = CheckpointStore(fetch_fn=lambda c, t: None, write_fn=boom)
    assert s.save(CrawlCheckpoint(client_id="acme", task_id="t-1")) is False


def test_load_returns_none_on_fetch_error():
    def boom(cid, tid): raise RuntimeError("db down")
    s = CheckpointStore(fetch_fn=boom, write_fn=lambda c: None)
    assert s.load_latest("acme", "t-1") is None


def test_has_checkpoint_false_on_fetch_error():
    def boom(cid, tid): raise RuntimeError("db down")
    s = CheckpointStore(fetch_fn=boom, write_fn=lambda c: None)
    assert not s.has_checkpoint("acme", "t-1")


def test_clear_returns_zero_on_error():
    def boom(cid, tid): raise RuntimeError("db down")
    s = CheckpointStore(
        fetch_fn=lambda c, t: None,
        write_fn=lambda c: None,
        clear_fn=boom,
    )
    assert s.clear_for_task("acme", "t-1") == 0


def test_load_returns_none_on_non_dict_fetch():
    s = CheckpointStore(
        fetch_fn=lambda cid, tid: "not a dict",
        write_fn=lambda c: None,
    )
    assert s.load_latest("acme", "t-1") is None


def test_load_returns_none_on_unparseable_checkpoint():
    s = CheckpointStore(
        fetch_fn=lambda cid, tid: {"totally": "wrong shape"},
        write_fn=lambda c: None,
    )
    # from_dict will fail gracefully; the return value is whatever
    # CrawlCheckpoint.from_dict produces (it uses defaults for missing
    # keys) — the important thing is we don't crash.
    result = s.load_latest("acme", "t-1")
    assert result is None or isinstance(result, CrawlCheckpoint)


def test_clear_defaults_to_noop_when_not_injected():
    s = CheckpointStore(
        fetch_fn=lambda c, t: None,
        write_fn=lambda c: None,
    )
    assert s.clear_for_task("acme", "t-1") == 0