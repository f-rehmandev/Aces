"""Unit tests for cleaning + normalization (spec §27)."""
from src.trust.cleaning import (
    MISSING, is_missing,
    normalize_text, parse_currency, normalize_date,
    normalize_phone, normalize_email,
    clean_record, clean_records,
)


# --- text ----------------------------------------------------------------

def test_text_trims_whitespace():
    assert normalize_text("  hello  ") == "hello"


def test_text_collapses_internal_whitespace():
    assert normalize_text("a    b   c") == "a b c"


def test_text_replaces_smart_quotes():
    assert normalize_text("\u201chi\u201d") == '"hi"'
    assert normalize_text("it\u2019s") == "it's"


def test_text_strips_zero_width():
    assert normalize_text("a\u200bb") == "ab"


def test_text_handles_nbsp():
    assert normalize_text("a\u00a0b") == "a b"


def test_text_non_string_passthrough():
    assert normalize_text(42) == 42
    assert normalize_text(None) is None


# --- currency ------------------------------------------------------------

def test_currency_usd_symbol():
    c = parse_currency("$1,299.00")
    assert c.currency == "USD"
    assert c.normalized_price == 1299.0
    assert c.raw_price == "$1,299.00"


def test_currency_gbp_symbol():
    c = parse_currency("£51.77")
    assert c.currency == "GBP"
    assert c.normalized_price == 51.77


def test_currency_iso_code():
    c = parse_currency("1299.00 USD")
    assert c.currency == "USD"
    assert c.normalized_price == 1299.0


def test_currency_default_when_none_detected():
    c = parse_currency("1299.00", default_currency="EUR")
    assert c.currency == "EUR"


def test_currency_from_number():
    c = parse_currency(42)
    assert c.normalized_price == 42.0


def test_currency_none():
    c = parse_currency(None)
    assert c.normalized_price is None


def test_currency_unparseable():
    c = parse_currency("Free")
    assert c.normalized_price is None
    assert c.raw_price == "Free"


# --- date ----------------------------------------------------------------

def test_date_iso_strips_time():
    assert normalize_date("2026-09-22T10:00:00Z") == "2026-09-22"


def test_date_slash_format():
    assert normalize_date("22/09/2026") == "2026-09-22"


def test_date_dash_format():
    assert normalize_date("22-09-2026") == "2026-09-22"


def test_date_two_digit_year():
    assert normalize_date("22/09/26") == "2026-09-22"


def test_date_unparseable():
    assert normalize_date("yesterday") is None


def test_date_none():
    assert normalize_date(None) is None
    assert normalize_date("") is None


# --- phone ---------------------------------------------------------------

def test_phone_e164_passthrough():
    assert normalize_phone("+92 300 1234567") == "+923001234567"


def test_phone_with_country_hint():
    assert normalize_phone("0300-1234567", default_country="PK") == "+923001234567"


def test_phone_without_country_hint_returns_none():
    assert normalize_phone("0300-1234567") is None


def test_phone_empty():
    assert normalize_phone(None) is None
    assert normalize_phone("") is None


def test_phone_unknown_country():
    assert normalize_phone("0300-1234567", default_country="XX") is None


# --- email ---------------------------------------------------------------

def test_email_lowercases_domain_only():
    assert normalize_email("Alice@Example.COM") == "Alice@example.com"


def test_email_invalid_returns_none():
    assert normalize_email("not-an-email") is None


def test_email_empty():
    assert normalize_email(None) is None
    assert normalize_email("") is None


# --- missing sentinel ---------------------------------------------------

def test_missing_sentinel_is_falsy():
    assert not MISSING


def test_missing_sentinel_singleton():
    from src.trust.cleaning import Missing
    assert Missing() is MISSING


def test_is_missing_checks():
    assert is_missing(MISSING)
    assert is_missing(None)
    assert not is_missing("")
    assert not is_missing(0)


# --- record cleaning ---------------------------------------------------

def test_clean_record_produces_currency_triple():
    r = clean_record({"price": "$1,299.00"})
    assert r.cleaned["price"] == "$1,299.00"
    assert r.cleaned["price_currency"] == "USD"
    assert r.cleaned["price_normalized"] == 1299.0


def test_clean_record_handles_date():
    r = clean_record({"published": "22/09/2026"})
    assert r.cleaned["published_normalized"] == "2026-09-22"


def test_clean_record_normalizes_email():
    r = clean_record({"contact_email": "Info@Shop.COM"})
    assert r.cleaned["contact_email"] == "Info@shop.com"


def test_clean_record_missing_becomes_sentinel():
    r = clean_record({"email": None})
    assert is_missing(r.cleaned["email"])


def test_clean_record_empty_string_becomes_sentinel():
    r = clean_record({"title": ""})
    assert is_missing(r.cleaned["title"])


def test_clean_record_passes_through_plain_fields():
    r = clean_record({"title": "  Widget  "})
    assert r.cleaned["title"] == "Widget"


def test_clean_records_batch():
    records = [{"price": "$1"}, {"price": "$2"}]
    cleaned, warnings = clean_records(records)
    assert len(cleaned) == 2
    assert cleaned[0]["price_normalized"] == 1.0
    assert cleaned[1]["price_normalized"] == 2.0
    assert warnings == []


def test_clean_records_reports_warnings():
    records = [{"price": "Free"}, {"published": "yesterday"}]
    _, warnings = clean_records(records)
    assert len(warnings) >= 1