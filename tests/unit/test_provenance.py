"""Unit tests for provenance (spec §26)."""
from src.trust.provenance import (
    ProvenanceStore, ProvenanceRecord, make_record_id, domain_of,
)


# --- record identity ---------------------------------------------------

def test_record_id_stable_for_same_title():
    a = make_record_id({"title": "Wireless Mouse"})
    b = make_record_id({"title": "Wireless Mouse"})
    assert a == b


def test_record_id_differs_for_different_titles():
    a = make_record_id({"title": "A"})
    b = make_record_id({"title": "B"})
    assert a != b


def test_record_id_prefers_title_over_name():
    r = {"name": "N", "title": "T"}
    rid = make_record_id(r)
    assert rid == make_record_id({"title": "T"})


def test_record_id_falls_back_to_url():
    rid = make_record_id({"url": "https://x.com/a"})
    assert rid.startswith("rec_")


def test_record_id_uses_random_when_nothing_available():
    rid = make_record_id({"unrelated": 1})
    assert rid.startswith("rec_")


def test_domain_of_strips_scheme_and_path():
    assert domain_of("https://Example.com/x?y=1") == "example.com"
    assert domain_of("http://sub.example.co.uk/a/b") == "sub.example.co.uk"


# --- store: writes -----------------------------------------------------

def test_record_single_field():
    store = ProvenanceStore(run_id="r", job_id="j")
    p = store.record(
        record_id="rec1",
        field_name="price",
        source_url="https://shop.example/a",
        extraction_method="css",
        raw_value="$24.99",
    )
    assert p.record_id == "rec1"
    assert p.field_name == "price"
    assert p.source_domain == "shop.example"
    assert p.extraction_method == "css"
    assert p.run_id == "r"
    assert p.job_id == "j"
    assert p.observed_at
    assert p.fetched_at


def test_record_preserves_raw_and_normalized_separately():
    store = ProvenanceStore()
    p = store.record(
        record_id="r", field_name="price",
        source_url="https://x.com/a", extraction_method="css",
        raw_value="£51.77", normalized_value="51.77",
    )
    assert p.raw_value == "£51.77"
    assert p.normalized_value == "51.77"


def test_record_stringifies_non_string_values():
    store = ProvenanceStore()
    p = store.record(
        record_id="r", field_name="count",
        source_url="https://x.com/a", extraction_method="css",
        raw_value=42,
    )
    assert p.raw_value == "42"


def test_record_none_value_preserved():
    store = ProvenanceStore()
    p = store.record(
        record_id="r", field_name="email",
        source_url="https://x.com/a", extraction_method="css",
        raw_value=None,
    )
    assert p.raw_value is None


# --- store: batch ------------------------------------------------------

def test_batch_records_every_field():
    store = ProvenanceStore()
    records = [{"title": "A", "price": "$1"}]
    out = store.record_batch(
        records, source_url="https://x.com/a", extraction_method="css",
    )
    field_names = {p.field_name for p in out}
    assert field_names == {"title", "price"}


def test_batch_ignores_bookkeeping_fields():
    store = ProvenanceStore()
    records = [{"title": "A", "diff_status": "New", "_numeric_price": 1.0}]
    out = store.record_batch(
        records, source_url="https://x.com/a", extraction_method="css",
    )
    names = {p.field_name for p in out}
    assert names == {"title"}


def test_batch_records_multiple_records():
    store = ProvenanceStore()
    records = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
    out = store.record_batch(
        records, source_url="https://x.com/a", extraction_method="css",
    )
    record_ids = {p.record_id for p in out}
    assert len(record_ids) == 3


# --- store: reads ------------------------------------------------------

def test_for_record_returns_only_matching():
    store = ProvenanceStore()
    store.record("r1", "title", "https://x.com/a", "css", "A")
    store.record("r1", "price", "https://x.com/a", "css", "$1")
    store.record("r2", "title", "https://x.com/b", "css", "B")
    assert len(store.for_record("r1")) == 2
    assert len(store.for_record("r2")) == 1
    assert len(store.for_record("nonexistent")) == 0


def test_for_field_returns_only_matching():
    store = ProvenanceStore()
    store.record("r1", "title", "https://x.com/a", "css", "A")
    store.record("r1", "price", "https://x.com/a", "css", "$1")
    fields = store.for_field("r1", "price")
    assert len(fields) == 1
    assert fields[0].field_name == "price"


def test_len_matches_number_of_records():
    store = ProvenanceStore()
    for i in range(5):
        store.record(f"r{i}", "title", "https://x.com/a", "css", "x")
    assert len(store) == 5


def test_to_list_serializes():
    store = ProvenanceStore()
    store.record("r1", "title", "https://x.com/a", "css", "A")
    data = store.to_list()
    assert isinstance(data, list)
    assert data[0]["record_id"] == "r1"
    assert data[0]["field_name"] == "title"
    assert data[0]["source_domain"] == "x.com"