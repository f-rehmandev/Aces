"""Unit tests for discovery/monitoring mode in run_grid_leads_query."""
import json

import pytest

from src.discovery.grid_search import run_grid_leads_query
from src.discovery.gmaps_client import GmapsJobResult
from src.discovery.lead_memory import LeadMemoryStore


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
    def call(self, prompt):
        return {"text": json.dumps(self.payload), "provider": "fake"}


class FakeGmapsClient:
    """Returns the same configured records regardless of keyword."""
    def __init__(self, records_per_query=None):
        self.records_per_query = records_per_query or {}
        self.calls: list[str] = []

    def scrape(self, keywords, **kwargs):
        q = keywords[0]
        self.calls.append(q)
        return GmapsJobResult(
            job_id=f"job-{len(self.calls)}",
            status="ok",
            records=list(self.records_per_query.get(q, [])),
        )


def _in_memory_store() -> tuple[LeadMemoryStore, dict]:
    rows: dict[tuple[str, str], dict] = {}

    def fetch(cid, place_ids):
        return {pid for (c, pid) in rows if c == cid and pid in place_ids}

    def write(cid, place_ids, qk):
        for pid in place_ids:
            if (cid, pid) not in rows:
                rows[(cid, pid)] = {"query_key": qk}

    return LeadMemoryStore(fetch_fn=fetch, write_fn=write), rows


def _llm_two_areas() -> FakeLLM:
    return FakeLLM({
        "entity": "pizza shops",
        "city": "Lahore",
        "areas": ["DHA", "Gulberg"],
    })


def _rec(place_id: str, title: str) -> dict:
    return {
        "place_id": place_id,
        "title": title,
        "address": f"{title} st",
        "phone": "0300",
        "website": "",
    }


# ---------------------------------------------------------------------------
# No memory → identical to old behavior
# ---------------------------------------------------------------------------

def test_no_memory_produces_no_diff_status():
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A")],
        "pizza shops in Gulberg Lahore": [_rec("2", "B")],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        lead_memory=None,
    )
    assert result.total_returned == 2
    for r in result.records:
        assert "diff_status" not in r
    assert result.new_count == 0
    assert result.existing_count == 0


# ---------------------------------------------------------------------------
# Monitoring mode (discovery_mode=False) — annotate, keep everything
# ---------------------------------------------------------------------------

def test_monitoring_marks_new_when_memory_empty():
    store, _ = _in_memory_store()
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A")],
        "pizza shops in Gulberg Lahore": [_rec("2", "B")],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=False,
    )
    statuses = sorted(r["diff_status"] for r in result.records)
    assert statuses == ["NEW", "NEW"]
    assert result.new_count == 2
    assert result.existing_count == 0
    assert result.total_returned == 2


def test_monitoring_marks_existing_for_known_ids():
    store, _ = _in_memory_store()
    store.remember_batch("acme", ["1"])   # seed memory
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A")],
        "pizza shops in Gulberg Lahore": [_rec("2", "B")],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=False,
    )
    by_id = {r["place_id"]: r["diff_status"] for r in result.records}
    assert by_id["1"] == "EXISTING"
    assert by_id["2"] == "NEW"
    assert result.new_count == 1
    assert result.existing_count == 1
    assert result.total_returned == 2   # monitoring keeps everything


def test_monitoring_isolates_clients():
    store, _ = _in_memory_store()
    store.remember_batch("acme", ["1"])
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A")],
        "pizza shops in Gulberg Lahore": [],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="other",       # different client
        lead_memory=store,
        discovery_mode=False,
    )
    assert result.records[0]["diff_status"] == "NEW"


# ---------------------------------------------------------------------------
# Discovery mode (discovery_mode=True) — filter, keep only NEW
# ---------------------------------------------------------------------------

def test_discovery_filters_existing():
    store, _ = _in_memory_store()
    store.remember_batch("acme", ["1"])   # seed memory
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A")],
        "pizza shops in Gulberg Lahore": [_rec("2", "B")],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=True,
    )
    # Only the NEW record is returned
    assert result.total_returned == 1
    assert result.records[0]["place_id"] == "2"
    assert result.records[0]["diff_status"] == "NEW"
    # Counters reflect the FULL scrape (1 existing filtered out + 1 new kept)
    assert result.new_count == 1
    assert result.existing_count == 1
    assert result.discovery_mode is True


def test_discovery_returns_everything_when_memory_empty():
    store, _ = _in_memory_store()
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A")],
        "pizza shops in Gulberg Lahore": [_rec("2", "B")],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=True,
    )
    assert result.total_returned == 2


def test_discovery_can_return_zero():
    store, _ = _in_memory_store()
    store.remember_batch("acme", ["1", "2"])
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A")],
        "pizza shops in Gulberg Lahore": [_rec("2", "B")],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=True,
    )
    assert result.total_returned == 0
    assert result.succeeded is True    # run itself succeeded
    assert result.new_count == 0


# ---------------------------------------------------------------------------
# Memory update semantics
# ---------------------------------------------------------------------------

def test_second_run_returns_no_new_leads_in_discovery():
    store, _ = _in_memory_store()
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A"), _rec("2", "B")],
        "pizza shops in Gulberg Lahore": [],
    })

    # Run 1 — first time seeing these leads
    r1 = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=True,
    )
    assert r1.total_returned == 2

    # Run 2 — same data, memory now knows them
    r2 = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=True,
    )
    assert r2.total_returned == 0


def test_monitoring_sees_them_as_existing_on_second_run():
    store, _ = _in_memory_store()
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A"), _rec("2", "B")],
        "pizza shops in Gulberg Lahore": [],
    })

    run_grid_leads_query(
        "pizza shops in Lahore", llm_router=_llm_two_areas(), client=client,
        client_id="acme", lead_memory=store, discovery_mode=False,
    )
    r2 = run_grid_leads_query(
        "pizza shops in Lahore", llm_router=_llm_two_areas(), client=client,
        client_id="acme", lead_memory=store, discovery_mode=False,
    )
    assert r2.new_count == 0
    assert r2.existing_count == 2


# ---------------------------------------------------------------------------
# Unverified records (no place_id)
# ---------------------------------------------------------------------------

def test_discovery_excludes_unverified_and_warns():
    store, _ = _in_memory_store()
    # Row has no place_id → _dedupe_key falls back to title|address,
    # but the annotated record has no place_id for memory lookup.
    no_id_row = {"title": "Ghost", "address": "Nowhere", "phone": "0300",
                 "website": ""}
    with_id = _rec("99", "Real")
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [with_id, no_id_row],
        "pizza shops in Gulberg Lahore": [],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=True,
    )
    assert result.total_returned == 1
    assert result.unverified_count == 1
    assert "without a place_id" in result.discovery_warning


def test_monitoring_keeps_unverified_records():
    store, _ = _in_memory_store()
    no_id_row = {"title": "Ghost", "address": "Nowhere", "phone": "0300",
                 "website": ""}
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [no_id_row],
        "pizza shops in Gulberg Lahore": [],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=False,
    )
    assert result.total_returned == 1
    assert result.records[0]["diff_status"] == "NEW_UNVERIFIED"
    assert result.unverified_count == 1


# ---------------------------------------------------------------------------
# Ordering / interaction with no-website filter
# ---------------------------------------------------------------------------

def test_no_website_filter_still_applies():
    store, _ = _in_memory_store()
    has_site = _rec("1", "Has"); has_site["website"] = "https://a.example"
    no_site = _rec("2", "None")
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [has_site, no_site],
        "pizza shops in Gulberg Lahore": [],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=False,
        no_website_only=True,
    )
    assert result.total_returned == 1
    assert result.records[0]["place_id"] == "2"
    assert result.filtered_no_website == 1


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_broken_memory_still_returns_leads():
    def boom(cid, pids):
        raise RuntimeError("db down")
    store = LeadMemoryStore(fetch_fn=boom, write_fn=lambda a, b, c: None)
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A")],
        "pizza shops in Gulberg Lahore": [],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore",
        llm_router=_llm_two_areas(),
        client=client,
        client_id="acme",
        lead_memory=store,
        discovery_mode=False,
    )
    # Graceful fallback: every lead treated as NEW
    assert result.total_returned == 1
    assert result.records[0]["diff_status"] == "NEW"