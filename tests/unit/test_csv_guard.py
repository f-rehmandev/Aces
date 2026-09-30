"""Unit tests for CSV/formula injection guard (spec §51)."""
from src.security.csv_guard import (
    guard_cell, guard_record, guard_dataset, is_dangerous,
)


# --- dangerous prefixes -----------------------------------------------

def test_equals_prefix_is_escaped():
    r = guard_cell("=cmd|'/c calc'!A0")
    assert r.was_sanitized
    assert r.sanitized == "'=cmd|'/c calc'!A0"


def test_plus_prefix_escaped_when_not_number():
    r = guard_cell("+1+1")
    assert r.was_sanitized


def test_at_prefix_escaped():
    r = guard_cell("@SUM(A1)")
    assert r.was_sanitized


def test_tab_prefix_escaped():
    r = guard_cell("\tfoo")
    assert r.was_sanitized


def test_cr_prefix_escaped():
    r = guard_cell("\rfoo")
    assert r.was_sanitized


def test_newline_prefix_escaped():
    r = guard_cell("\nfoo")
    assert r.was_sanitized


# --- numeric exemptions (§51.1) --------------------------------------

def test_negative_number_exempt():
    r = guard_cell("-15.99")
    assert not r.was_sanitized
    assert r.sanitized == "-15.99"


def test_positive_number_exempt():
    r = guard_cell("+42")
    assert not r.was_sanitized


def test_pure_int_exempt():
    r = guard_cell(42)
    assert not r.was_sanitized


def test_float_exempt():
    r = guard_cell(-3.14)
    assert not r.was_sanitized


def test_currency_string_exempt():
    r = guard_cell("1299.00 USD")
    assert not r.was_sanitized


# --- non-strings pass through ----------------------------------------

def test_none_passes_through():
    r = guard_cell(None)
    assert r.sanitized is None
    assert not r.was_sanitized


def test_bool_passes_through():
    r = guard_cell(True)
    assert r.sanitized is True


# --- safe strings untouched ------------------------------------------

def test_plain_string_untouched():
    r = guard_cell("hello")
    assert not r.was_sanitized


def test_empty_string_untouched():
    r = guard_cell("")
    assert r.sanitized == ""


# --- records ---------------------------------------------------------

def test_guard_record_does_not_mutate_input():
    rec = {"a": "=x", "b": "safe"}
    guard_record(rec)
    assert rec["a"] == "=x"


def test_guard_record_sanitizes_dangerous_only():
    rec = {"a": "=x", "b": "safe", "c": -1.5}
    cleaned = guard_record(rec)
    assert cleaned["a"] == "'=x"
    assert cleaned["b"] == "safe"
    assert cleaned["c"] == -1.5


def test_guard_dataset_counts_sanitized_cells():
    dataset = [{"a": "=x", "b": "ok"}, {"a": "ok", "b": "@y"}]
    cleaned, count = guard_dataset(dataset)
    assert count == 2
    assert cleaned[0]["a"] == "'=x"
    assert cleaned[1]["b"] == "'@y"


def test_guard_dataset_returns_new_list():
    dataset = [{"a": "=x"}]
    cleaned, _ = guard_dataset(dataset)
    assert dataset[0]["a"] == "=x"      # original untouched
    assert cleaned[0]["a"] == "'=x"


# --- is_dangerous -----------------------------------------------------

def test_is_dangerous_true_for_formula():
    assert is_dangerous("=1+1")


def test_is_dangerous_false_for_number():
    assert not is_dangerous("-5")


def test_is_dangerous_false_for_plain_text():
    assert not is_dangerous("hello")