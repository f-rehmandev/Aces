"""Unit tests for multi-source triangulation (spec §23)."""
import pytest

from src.trust.triangulation import (
    Triangulator, Observation, ConsensusResult, Conflict,
    normalize_for_comparison, triangulate, observations_from_records,
)


# --- normalization ------------------------------------------------------

def test_currency_normalization():
    assert normalize_for_comparison("$24.99", "currency") == "24.9900"
    assert normalize_for_comparison("24.99 USD", "currency") == "24.9900"
    assert normalize_for_comparison("$24.99", "currency") == normalize_for_comparison("24.99", "currency")


def test_text_normalization_lowercases_and_trims():
    assert normalize_for_comparison("  Hello World  ", "text") == "hello world"


def test_url_normalization():
    a = normalize_for_comparison("https://X.com/a?utm_source=foo", "url")
    b = normalize_for_comparison("https://x.com/a", "url")
    assert a == b


def test_email_normalization():
    assert normalize_for_comparison("A@B.com", "email") == "a@b.com"


def test_date_normalization():
    assert normalize_for_comparison("2026-09-22T10:00:00Z", "date") == "2026-09-22"


def test_none_normalizes_to_empty():
    assert normalize_for_comparison(None, "text") == ""


# --- unanimity ---------------------------------------------------------

def test_three_independent_sources_agree():
    obs = [
        Observation("r1", "price", "a.example", "$24.99", trust_score=0.9),
        Observation("r1", "price", "b.example", "24.99 USD", trust_score=0.85),
        Observation("r1", "price", "c.example", "$24.99", trust_score=0.9),
    ]
    results = triangulate(obs)
    assert len(results) == 1
    r = results[0]
    assert r.consensus_value in ("$24.99", "24.99 USD")
    assert r.confidence >= 0.6
    assert r.conflicts == []


def test_single_source_confidence_is_capped():
    obs = [Observation("r1", "x", "solo.example", "only", trust_score=0.9)]
    r = triangulate(obs)[0]
    assert r.confidence <= 0.5
    assert "only one" in r.note


def test_unanimous_four_sources_full_confidence():
    obs = [Observation("r1", "x", f"s{i}.example", "V", trust_score=0.9) for i in range(4)]
    r = triangulate(obs)[0]
    assert r.confidence >= 0.95


# --- conflict ----------------------------------------------------------

def test_two_agree_one_dissents():
    obs = [
        Observation("r1", "price", "a.example", "$24.99", trust_score=0.9),
        Observation("r1", "price", "b.example", "$24.99", trust_score=0.9),
        Observation("r1", "price", "c.example", "$29.99", trust_score=0.9),
    ]
    r = triangulate(obs)[0]
    assert "24.99" in str(r.consensus_value)
    assert len(r.conflicts) == 1
    assert r.conflicts[0].value == "$29.99"
    assert r.conflicts[0].sources == ["c.example"]
    assert r.has_conflict


# --- independence clustering ------------------------------------------

def test_same_domain_sources_are_one_cluster():
    obs = [
        Observation("r1", "x", "a.example", "V", trust_score=0.9),
        Observation("r1", "x", "a.example", "V", trust_score=0.9),
        Observation("r1", "x", "b.example", "W", trust_score=0.9),
    ]
    r = triangulate(obs)[0]
    assert r.total_clusters == 2
    assert r.consensus_value == "V"   # cluster A has weight 1.8 vs cluster B 0.9


def test_explicit_independence_groups_override_domain():
    obs = [
        Observation("r1", "x", "a.example", "V", trust_score=0.9, independence_group="feed1"),
        Observation("r1", "x", "b.example", "V", trust_score=0.9, independence_group="feed1"),
        Observation("r1", "x", "c.example", "W", trust_score=0.9, independence_group="feed2"),
    ]
    r = triangulate(obs)[0]
    assert r.total_clusters == 2


# --- trust weighting --------------------------------------------------

def test_high_trust_source_beats_two_low_trust():
    obs = [
        Observation("r1", "x", "trusted.example", "A", trust_score=0.99),
        Observation("r1", "x", "cheap1.example", "B", trust_score=0.1),
        Observation("r1", "x", "cheap2.example", "B", trust_score=0.1),
    ]
    r = triangulate(obs)[0]
    assert r.consensus_value == "A"


def test_all_zero_trust_falls_back_to_majority():
    obs = [
        Observation("r1", "x", "a.example", "A", trust_score=0.0),
        Observation("r1", "x", "b.example", "A", trust_score=0.0),
        Observation("r1", "x", "c.example", "B", trust_score=0.0),
    ]
    r = triangulate(obs)[0]
    assert r.consensus_value == "A"


# --- edge cases -------------------------------------------------------

def test_missing_values_are_skipped():
    obs = [
        Observation("r1", "x", "a.example", None),
        Observation("r1", "x", "b.example", "A"),
    ]
    r = triangulate(obs)[0]
    assert r.consensus_value == "A"


def test_empty_observations():
    assert triangulate([]) == []


def test_multiple_records_dont_mix():
    obs = [
        Observation("r1", "price", "a.example", "$10"),
        Observation("r2", "price", "a.example", "$20"),
    ]
    results = triangulate(obs)
    assert len(results) == 2
    by_rec = {r.record_id: r for r in results}
    assert by_rec["r1"].consensus_value == "$10"
    assert by_rec["r2"].consensus_value == "$20"


def test_multiple_fields_dont_mix():
    obs = [
        Observation("r1", "price", "a.example", "$10"),
        Observation("r1", "title", "a.example", "Widget"),
    ]
    results = triangulate(obs)
    assert len(results) == 2
    fields = {r.field_name for r in results}
    assert fields == {"price", "title"}


# --- field type inference --------------------------------------------

def test_field_type_inferred_from_name():
    obs = [
        Observation("r1", "price", "a.example", "$10.00"),
        Observation("r1", "price", "b.example", "10.00"),
    ]
    r = triangulate(obs)[0]
    # Both normalize to "10.0000"
    assert not r.conflicts


def test_explicit_field_types_override():
    obs = [
        Observation("r1", "custom", "a.example", "Hello"),
        Observation("r1", "custom", "b.example", "hello"),
    ]
    r = triangulate(obs, field_types={"custom": "text"})[0]
    assert not r.conflicts


# --- observations_from_records helper ---------------------------------

def test_observations_from_records():
    records = [
        {"title": "A", "price": "$1"},
        {"title": "B", "price": "$2"},
    ]
    obs = observations_from_records(records, source_domain="x.example", trust_score=0.8)
    assert len(obs) == 4
    assert all(o.source_domain == "x.example" for o in obs)
    assert all(o.trust_score == 0.8 for o in obs)


def test_observations_from_records_skips_bookkeeping():
    records = [{"title": "A", "source_url": "https://x.com/a", "diff_status": "New"}]
    obs = observations_from_records(records, source_domain="x.example")
    fields = {o.field_name for o in obs}
    assert fields == {"title"}


# --- serialization ----------------------------------------------------

def test_consensus_result_to_dict():
    obs = [Observation("r1", "price", "a.example", "$10")]
    r = triangulate(obs)[0]
    d = r.to_dict()
    assert d["record_id"] == "r1"
    assert d["field_name"] == "price"
    assert "conflicts" in d


def test_conflict_to_dict():
    c = Conflict(value="$5", normalized="5.0000", sources=["a.example"], weight=0.9)
    d = c.to_dict()
    assert d["value"] == "$5"
    assert d["sources"] == ["a.example"]