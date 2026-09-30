"""Unit tests for LeadMemoryStore (discovery vs monitoring mode)."""
import pytest

from src.discovery.lead_memory import (
    LeadMemoryStore,
    STATUS_NEW,
    STATUS_EXISTING,
)


@pytest.fixture
def store():
    rows: dict[tuple[str, str], dict] = {}

    def fetch(cid, place_ids):
        return {pid for (c, pid) in rows if c == cid and pid in place_ids}

    def write(cid, place_ids, qk):
        for pid in place_ids:
            if (cid, pid) not in rows:
                rows[(cid, pid)] = {"query_key": qk}

    return LeadMemoryStore(fetch_fn=fetch, write_fn=write), rows


# --- reads ---------------------------------------------------------------

def test_empty_memory_returns_nothing(store):
    s, _ = store
    assert s.known_place_ids("acme", ["a", "b"]) == set()


def test_is_known_true_for_remembered(store):
    s, _ = store
    s.remember_batch("acme", ["a"])
    assert s.is_known("acme", "a")


def test_is_known_false_for_unknown(store):
    s, _ = store
    assert not s.is_known("acme", "z")


def test_is_known_empty_place_id_false(store):
    s, _ = store
    assert not s.is_known("acme", "")


# --- writes --------------------------------------------------------------

def test_remember_batch_records_all(store):
    s, rows = store
    s.remember_batch("acme", ["a", "b", "c"], query_key="q1")
    assert ("acme", "a") in rows
    assert ("acme", "b") in rows
    assert ("acme", "c") in rows


def test_remember_batch_records_query_key_once(store):
    s, rows = store
    s.remember_batch("acme", ["a"], query_key="pizza")
    assert rows[("acme", "a")]["query_key"] == "pizza"


def test_remember_does_not_overwrite_first_query_key(store):
    s, rows = store
    s.remember_batch("acme", ["a"], query_key="first")
    s.remember_batch("acme", ["a"], query_key="second")
    # Original query_key preserved (first-seen semantics)
    assert rows[("acme", "a")]["query_key"] == "first"


def test_remember_batch_skips_empty_place_ids(store):
    s, rows = store
    s.remember_batch("acme", ["a", "", None, "b"])
    assert ("acme", "a") in rows
    assert ("acme", "b") in rows
    assert ("acme", "") not in rows


def test_remember_batch_empty_is_noop(store):
    s, rows = store
    s.remember_batch("acme", [])
    assert rows == {}


# --- filter_new (discovery mode) ----------------------------------------

def test_filter_new_returns_everything_when_empty_memory(store):
    s, _ = store
    assert s.filter_new("acme", ["a", "b", "c"]) == ["a", "b", "c"]


def test_filter_new_excludes_known(store):
    s, _ = store
    s.remember_batch("acme", ["a"])
    assert s.filter_new("acme", ["a", "b", "c"]) == ["b", "c"]


def test_filter_new_preserves_order(store):
    s, _ = store
    s.remember_batch("acme", ["b"])
    assert s.filter_new("acme", ["a", "b", "c"]) == ["a", "c"]


def test_filter_new_empty_input(store):
    s, _ = store
    assert s.filter_new("acme", []) == []


# --- client isolation ---------------------------------------------------

def test_memory_is_per_client(store):
    s, _ = store
    s.remember_batch("acme", ["a"])
    assert s.is_known("acme", "a")
    assert not s.is_known("other", "a")


def test_same_place_id_two_clients_isolated(store):
    s, rows = store
    s.remember_batch("acme", ["x"], query_key="acme-query")
    s.remember_batch("other", ["x"], query_key="other-query")
    assert ("acme", "x") in rows
    assert ("other", "x") in rows
    assert rows[("acme", "x")]["query_key"] == "acme-query"
    assert rows[("other", "x")]["query_key"] == "other-query"


# --- annotate_records (monitoring mode) --------------------------------

def test_annotate_marks_existing(store):
    s, _ = store
    s.remember_batch("acme", ["a"])
    records = [{"place_id": "a", "business_name": "A"}]
    annotated, new_count, existing_count = s.annotate_records("acme", records)
    assert annotated[0]["diff_status"] == STATUS_EXISTING
    assert new_count == 0
    assert existing_count == 1


def test_annotate_marks_new(store):
    s, _ = store
    records = [{"place_id": "z", "business_name": "Z"}]
    annotated, new_count, existing_count = s.annotate_records("acme", records)
    assert annotated[0]["diff_status"] == STATUS_NEW
    assert new_count == 1
    assert existing_count == 0


def test_annotate_mixed(store):
    s, _ = store
    s.remember_batch("acme", ["a", "b"])
    records = [
        {"place_id": "a"},
        {"place_id": "c"},
        {"place_id": "b"},
        {"place_id": "d"},
    ]
    annotated, new_count, existing_count = s.annotate_records("acme", records)
    statuses = [r["diff_status"] for r in annotated]
    assert statuses == ["EXISTING", "NEW", "EXISTING", "NEW"]
    assert new_count == 2
    assert existing_count == 2


def test_annotate_record_without_place_id(store):
    s, _ = store
    records = [{"business_name": "No ID"}]
    annotated, new_count, existing_count = s.annotate_records("acme", records)
    assert annotated[0]["diff_status"] == "EXISTING_UNKNOWN"
    assert new_count == 0
    assert existing_count == 0


def test_annotate_does_not_mutate_input(store):
    s, _ = store
    original = {"place_id": "z", "business_name": "Z"}
    s.annotate_records("acme", [original])
    # The dict the caller passed in is untouched
    assert "diff_status" not in original


def test_annotate_empty(store):
    s, _ = store
    annotated, n, e = s.annotate_records("acme", [])
    assert annotated == [] and n == 0 and e == 0


# --- graceful degradation ----------------------------------------------

def test_broken_fetch_treated_as_empty():
    def boom(cid, pids):
        raise RuntimeError("db down")
    s = LeadMemoryStore(fetch_fn=boom, write_fn=lambda a, b, c: None)
    assert s.known_place_ids("acme", ["a"]) == set()
    assert s.filter_new("acme", ["a"]) == ["a"]


def test_broken_write_does_not_raise():
    def boom(cid, pids, qk):
        raise RuntimeError("db down")
    s = LeadMemoryStore(fetch_fn=lambda a, b: set(), write_fn=boom)
    s.remember_batch("acme", ["a"])   # must not raise


def test_annotate_on_broken_store_labels_everything_new():
    def boom(cid, pids):
        raise RuntimeError("db down")
    s = LeadMemoryStore(fetch_fn=boom, write_fn=lambda a, b, c: None)
    annotated, n, e = s.annotate_records("acme", [{"place_id": "a"}])
    assert annotated[0]["diff_status"] == STATUS_NEW
    assert n == 1 and e == 0
