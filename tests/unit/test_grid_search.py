"""Unit tests for Google Maps grid search."""

import json

import pytest

from src.discovery.grid_search import (
    GridExpansion, dedupe_by_identity, expand_grid_search,
    run_grid_leads_query, _dedupe_key,
)
from src.discovery.gmaps_client import GmapsJobResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
    def call(self, prompt):
        return {"text": json.dumps(self.payload), "provider": "fake"}


class BrokenLLM:
    def call(self, prompt):
        raise RuntimeError("LLM offline")


class BadJSONLLM:
    def call(self, prompt):
        return {"text": "not json at all", "provider": "fake"}


class FakeGmapsClient:
    """Returns different records per keyword so we can test dedup."""
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def scrape(self, keywords, **kwargs):
        q = keywords[0]
        self.calls.append(q)
        return GmapsJobResult(
            job_id=f"job-{len(self.calls)}",
            status="ok",
            records=list(self.mapping.get(q, [])),
        )


# ---------------------------------------------------------------------------
# expand_grid_search
# ---------------------------------------------------------------------------

def test_expand_uses_llm_to_build_area_queries():
    llm = FakeLLM({
        "entity": "pizza shops",
        "city": "Lahore",
        "areas": ["DHA", "Gulberg", "Johar Town"],
    })
    exp = expand_grid_search("pizza shops in Lahore", llm_router=llm)
    assert exp.source == "llm"
    assert exp.entity == "pizza shops"
    assert exp.city == "Lahore"
    assert exp.areas == ["DHA", "Gulberg", "Johar Town"]
    assert exp.expanded_queries == [
        "pizza shops in DHA Lahore",
        "pizza shops in Gulberg Lahore",
        "pizza shops in Johar Town Lahore",
    ]


def test_expand_does_not_duplicate_city_if_area_includes_it():
    llm = FakeLLM({
        "entity": "pizza shops", "city": "Lahore",
        "areas": ["DHA Lahore"],
    })
    exp = expand_grid_search("pizza shops in Lahore", llm_router=llm)
    assert exp.expanded_queries == ["pizza shops in DHA Lahore"]


def test_expand_with_explicit_areas_skips_llm():
    exp = expand_grid_search(
        "pizza shops in Lahore",
        areas=["DHA", "Gulberg"],
    )
    assert exp.source == "explicit"
    assert exp.expanded_queries == [
        "pizza shops in Lahore DHA",
        "pizza shops in Lahore Gulberg",
    ]


def test_expand_falls_back_when_llm_returns_no_areas():
    llm = FakeLLM({"entity": "pizza shops", "city": "Lahore", "areas": []})
    exp = expand_grid_search("pizza shops in Lahore", llm_router=llm)
    assert exp.source == "passthrough"
    assert exp.expanded_queries == ["pizza shops in Lahore"]


def test_expand_falls_back_on_llm_exception():
    exp = expand_grid_search("pizza shops in Lahore", llm_router=BrokenLLM())
    assert exp.source == "passthrough"
    assert exp.expanded_queries == ["pizza shops in Lahore"]
    assert "failed" in exp.error.lower() or "LLM" in exp.error


def test_expand_falls_back_on_bad_json():
    exp = expand_grid_search("pizza shops in Lahore", llm_router=BadJSONLLM())
    assert exp.source == "passthrough"
    assert "non-JSON" in exp.error or "JSON" in exp.error


def test_expand_caps_max_areas():
    llm = FakeLLM({
        "entity": "x", "city": "Lahore",
        "areas": [f"A{i}" for i in range(30)],
    })
    exp = expand_grid_search("x in Lahore", llm_router=llm, max_areas=5)
    assert len(exp.areas) == 5
    assert len(exp.expanded_queries) == 5


def test_expand_dedupes_repeated_areas():
    llm = FakeLLM({
        "entity": "pizza", "city": "Lahore",
        "areas": ["DHA", "DHA", "Gulberg"],
    })
    exp = expand_grid_search("pizza in Lahore", llm_router=llm)
    assert exp.areas == ["DHA", "Gulberg"]


def test_expand_strips_code_fences():
    class FencedLLM:
        def call(self, prompt):
            return {"text": "```json\n" + json.dumps({
                "entity": "x", "city": "Y", "areas": ["Z"],
            }) + "\n```", "provider": "fake"}
    exp = expand_grid_search("x in Y", llm_router=FencedLLM())
    assert exp.source == "llm"
    assert exp.expanded_queries == ["x in Z Y"]


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def test_dedupe_prefers_place_id():
    rows = [
        {"place_id": "abc", "title": "A"},
        {"place_id": "abc", "title": "A (dupe)"},
    ]
    uniq, dropped = dedupe_by_identity(rows)
    assert dropped == 1
    assert len(uniq) == 1


def test_dedupe_falls_back_to_title_and_address():
    rows = [
        {"title": "X", "address": "Y"},
        {"title": "X", "address": "Y"},
    ]
    uniq, dropped = dedupe_by_identity(rows)
    assert dropped == 1
    assert len(uniq) == 1


def test_dedupe_distinct_rows_kept():
    rows = [
        {"place_id": "abc"},
        {"place_id": "def"},
    ]
    uniq, dropped = dedupe_by_identity(rows)
    assert dropped == 0
    assert len(uniq) == 2


def test_dedupe_empty():
    uniq, dropped = dedupe_by_identity([])
    assert uniq == [] and dropped == 0


def test_dedupe_key_variants():
    assert _dedupe_key({"place_id": "abc"}).startswith("id:")
    assert _dedupe_key({"title": "T", "address": "A"}).startswith("ta:")
    assert _dedupe_key({}).startswith("h:")


# ---------------------------------------------------------------------------
# run_grid_leads_query
# ---------------------------------------------------------------------------

def _rec(place_id, title, website=""):
    return {
        "place_id": place_id,
        "title": title,
        "address": f"{title} st",
        "phone": "0300",
        "website": website,
    }


def test_grid_merges_and_dedupes():
    llm = FakeLLM({
        "entity": "pizza shops", "city": "Lahore",
        "areas": ["DHA", "Gulberg"],
    })
    client = FakeGmapsClient({
        "pizza shops in DHA Lahore": [_rec("1", "A"), _rec("2", "B")],
        "pizza shops in Gulberg Lahore": [_rec("2", "B"), _rec("3", "C")],
    })
    result = run_grid_leads_query(
        "pizza shops in Lahore", llm_router=llm, client=client,
    )
    assert result.areas_searched == 2
    assert result.total_scraped == 4
    assert result.duplicates_removed == 1
    assert result.total_returned == 3


def test_grid_applies_no_website_filter():
    llm = FakeLLM({"entity": "x", "city": "Y", "areas": ["A", "B"]})
    client = FakeGmapsClient({
        "x in A Y": [_rec("1", "Has", "https://a.example")],
        "x in B Y": [_rec("2", "None", "")],
    })
    result = run_grid_leads_query(
        "x in Y", llm_router=llm, client=client, no_website_only=True,
    )
    assert result.filtered_no_website == 1
    assert result.total_returned == 1


def test_grid_single_query_passthrough():
    llm = FakeLLM({"entity": "x", "city": "Y", "areas": []})
    client = FakeGmapsClient({"x in Y": [_rec("1", "A")]})
    result = run_grid_leads_query(
        "x in Y", llm_router=llm, client=client,
    )
    assert result.areas_searched == 1
    assert result.total_returned == 1


def test_grid_collects_per_area_errors():
    llm = FakeLLM({"entity": "x", "city": "Y", "areas": ["A", "B"]})

    class PartialClient:
        def scrape(self, keywords, **kwargs):
            q = keywords[0]
            if "A" in q:
                raise RuntimeError("area A blocked")
            return GmapsJobResult(
                job_id="j", status="ok",
                records=[_rec("1", "OK")],
            )

    result = run_grid_leads_query(
        "x in Y", llm_router=llm, client=PartialClient(),
    )
    assert result.areas_searched == 2
    assert len(result.per_area_errors) == 1
    assert result.total_returned == 1


def test_grid_no_results_reports_error():
    llm = FakeLLM({"entity": "x", "city": "Y", "areas": ["A"]})
    client = FakeGmapsClient({"x in A Y": []})
    result = run_grid_leads_query("x in Y", llm_router=llm, client=client)
    assert not result.succeeded
    assert "no results" in result.error.lower()


def test_grid_progress_callback_fires_per_area():
    llm = FakeLLM({"entity": "x", "city": "Y", "areas": ["A", "B", "C"]})
    client = FakeGmapsClient({f"x in {a} Y": [] for a in "ABC"})
    calls = []
    run_grid_leads_query(
        "x in Y", llm_router=llm, client=client,
        progress_callback=lambda i, n, q: calls.append((i, n, q)),
    )
    assert [c[0] for c in calls] == [1, 2, 3]
    assert all(c[1] == 3 for c in calls)