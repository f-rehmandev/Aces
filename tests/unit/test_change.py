"""Unit tests for change classification (spec §29)."""
from src.history.change import (
    ChangeClassifier, classify, identity_key, ChangeSet,
    RecordChange, FieldChange,
)


# --- identity ---------------------------------------------------------

def test_identity_priority_order():
    assert identity_key({"title": "T", "name": "N"}) == "T"
    assert identity_key({"business_name": "B", "name": "N"}) == "B"
    assert identity_key({"name": "N"}) == "N"


def test_identity_none_when_no_candidates():
    assert identity_key({"x": 1}) is None


# --- basic classification --------------------------------------------

def test_new_record():
    cs = classify([], [{"title": "A"}])
    assert len(cs.new) == 1
    assert cs.new[0].change_type == "NEW"


def test_removed_record():
    cs = classify([{"title": "A"}], [])
    assert len(cs.removed) == 1
    assert cs.removed[0].change_type == "REMOVED"


def test_modified_record():
    cs = classify(
        [{"title": "A", "price": "$10"}],
        [{"title": "A", "price": "$15"}],
    )
    assert len(cs.modified) == 1
    assert cs.modified[0].change_type == "MODIFIED"
    assert cs.modified[0].field_changes[0].field_name == "price"


def test_unchanged_record():
    cs = classify(
        [{"title": "A", "price": "$10"}],
        [{"title": "A", "price": "$10"}],
    )
    assert len(cs.unchanged) == 1


def test_full_change_set():
    prev = [{"title": "A"}, {"title": "B"}]
    curr = [{"title": "B"}, {"title": "C"}]
    cs = classify(prev, curr)
    assert len(cs.new) == 1
    assert len(cs.removed) == 1
    assert len(cs.unchanged) == 1


# --- noise suppression ------------------------------------------------

def test_case_only_change_is_ignored():
    cs = classify(
        [{"title": "A", "note": "in stock"}],
        [{"title": "A", "note": "IN STOCK"}],
    )
    assert not cs.has_changes


def test_whitespace_only_change_is_ignored():
    cs = classify(
        [{"title": "A", "note": "in  stock"}],
        [{"title": "A", "note": "in stock"}],
    )
    assert not cs.has_changes


def test_currency_formatting_equivalence():
    cs = classify(
        [{"title": "A", "price": "$1,299.00"}],
        [{"title": "A", "price": "1299.00 USD"}],
    )
    assert not cs.has_changes


def test_price_change_detected_numerically():
    cs = classify(
        [{"title": "A", "price": "$10.00"}],
        [{"title": "A", "price": "$10.50"}],
    )
    assert cs.has_changes
    assert cs.modified[0].field_changes[0].field_name == "price"


def test_timestamp_change_suppressed_by_default():
    cs = classify(
        [{"title": "A", "updated_at": "2026-01-01"}],
        [{"title": "A", "updated_at": "2026-09-24"}],
    )
    assert not cs.has_changes


def test_timestamp_change_detected_when_not_suppressed():
    classifier = ChangeClassifier(suppress_timestamps=False)
    cs = classifier.classify(
        [{"title": "A", "updated_at": "2026-01-01"}],
        [{"title": "A", "updated_at": "2026-09-24"}],
    )
    assert cs.has_changes


def test_source_url_field_ignored():
    cs = classify(
        [{"title": "A", "source_url": "https://x.com/a"}],
        [{"title": "A", "source_url": "https://x.com/b"}],
    )
    assert not cs.has_changes


# --- field-level diff -------------------------------------------------

def test_multiple_field_changes():
    cs = classify(
        [{"title": "A", "price": "$10", "note": "old"}],
        [{"title": "A", "price": "$15", "note": "new"}],
    )
    fc_names = {fc.field_name for fc in cs.modified[0].field_changes}
    assert fc_names == {"price", "note"}


def test_field_added_appears_as_change():
    cs = classify(
        [{"title": "A"}],
        [{"title": "A", "note": "new"}],
    )
    assert cs.has_changes
    fc = cs.modified[0].field_changes[0]
    assert fc.field_name == "note"
    assert fc.old_value is None


def test_field_removed_appears_as_change():
    cs = classify(
        [{"title": "A", "note": "old"}],
        [{"title": "A"}],
    )
    assert cs.has_changes


# --- result objects ---------------------------------------------------

def test_change_set_total():
    cs = classify(
        [{"title": "A"}, {"title": "B"}],
        [{"title": "B"}, {"title": "C"}],
    )
    assert cs.total == 3


def test_change_set_to_dict():
    cs = classify([{"title": "A"}], [{"title": "B"}])
    d = cs.to_dict()
    assert "summary" in d
    assert d["summary"]["new"] == 1
    assert d["summary"]["removed"] == 1


def test_has_changes_false_when_all_unchanged():
    cs = classify([{"title": "A"}], [{"title": "A"}])
    assert not cs.has_changes


def test_has_changes_true_when_any_change():
    cs = classify([{"title": "A"}], [{"title": "B"}])
    assert cs.has_changes