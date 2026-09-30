"""Unit tests for the PII egress guard (spec §27.1)."""
from src.trust.pii_guard import PIIGuard, scan


def test_credit_card_redacted_when_not_declared():
    r = scan([{"note": "card on file 4111-1111-1111-1111"}])
    assert r.has_redactions
    assert r.records[0]["note"] == "[REDACTED_PII]"
    assert r.redactions[0].kind == "credit_card"


def test_ssn_redacted():
    r = scan([{"note": "SSN 123-45-6789"}])
    assert r.has_redactions
    assert r.redactions[0].kind == "ssn"


def test_password_redacted():
    r = scan([{"note": "see password=hunter2hunter2"}])
    assert r.has_redactions
    assert r.redactions[0].kind == "credential"


def test_clean_record_untouched():
    r = scan([{"note": "nothing sensitive here"}])
    assert not r.has_redactions
    assert r.records[0]["note"] == "nothing sensitive here"


def test_declared_field_is_preserved():
    r = scan([{"email": "user@example.com", "note": "card 4111-1111-1111-1111"}],
             declared_fields=["email"])
    assert r.records[0]["email"] == "user@example.com"
    assert r.records[0]["note"] == "[REDACTED_PII]"


def test_declared_field_matching_pii_kind_preserved():
    # Even if the user explicitly asked for credit_card-shaped data,
    # we honour the declaration.
    r = scan([{"credit_card": "4111-1111-1111-1111"}], declared_fields=["credit_card"])
    assert r.records[0]["credit_card"] == "4111-1111-1111-1111"
    assert not r.has_redactions


def test_multiple_records_scanned():
    r = scan([
        {"n": "clean"},
        {"n": "ssn 123-45-6789"},
        {"n": "clean too"},
    ])
    assert len(r.redactions) == 1
    assert r.records[0]["n"] == "clean"
    assert r.records[1]["n"] == "[REDACTED_PII]"
    assert r.records[2]["n"] == "clean too"


def test_non_string_values_ignored():
    r = scan([{"count": 42, "nested": {"a": 1}}])
    assert not r.has_redactions


def test_empty_value_ignored():
    r = scan([{"n": ""}, {"n": None}])
    assert not r.has_redactions


def test_summary_reports_kinds():
    r = scan([
        {"a": "4111-1111-1111-1111"},
        {"b": "123-45-6789"},
    ])
    s = r.summary()
    assert "credit_card" in s
    assert "ssn" in s


def test_classify_helper():
    assert PIIGuard._classify("4111-1111-1111-1111") == "credit_card"
    assert PIIGuard._classify("123-45-6789") == "ssn"
    assert PIIGuard._classify("hello world") is None
    assert PIIGuard._classify("") is None