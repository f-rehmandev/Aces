"""
Cleaning + normalization — spec §27.

Extraction is not enough. Data must be sanitized, formatted, and checked
for sensitive leakage before entering deliverables.

This module implements:
    §27.2  text normalization
    §27.3  currency → raw + currency + normalized_price
    §27.4  dates → ISO 8601 in normalized_date, raw preserved
    §27.5  phone → E.164 when country known
    §27.6  email → validate, preserve local-part case
    §27.7  missing-value sentinel distinguishable from empty string
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Missing-value sentinel (§27.7)
# ---------------------------------------------------------------------------

class Missing:
    """Distinguishable missing-value sentinel."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self) -> str:
        return "<MISSING>"
    def __bool__(self) -> bool:
        return False


MISSING = Missing()


def is_missing(v: Any) -> bool:
    return v is MISSING or v is None


# ---------------------------------------------------------------------------
# Text normalization (§27.2)
# ---------------------------------------------------------------------------

_SMART_QUOTES = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u00ab": '"', "\u00bb": '"',
}
_NBSP = "\u00a0"
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTISPACE = re.compile(r"\s+")


def normalize_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    s = s.replace(_NBSP, " ")
    s = _ZERO_WIDTH.sub("", s)
    s = s.strip()
    s = _MULTISPACE.sub(" ", s)
    return s


# ---------------------------------------------------------------------------
# Currency (§27.3)
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOLS = {
    "$": "USD", "US$": "USD",
    "£": "GBP",
    "€": "EUR",
    "¥": "JPY", "￥": "JPY",
    "₹": "INR",
    "₨": "PKR", "Rs": "PKR", "PKR": "PKR",
    "AED": "AED", "SAR": "SAR",
    "CNY": "CNY", "RMB": "CNY",
    "CAD": "CAD", "AUD": "AUD",
}

_CURRENCY_CODE_RE = re.compile(
    r"\b(USD|EUR|GBP|JPY|INR|PKR|AED|SAR|CNY|CAD|AUD)\b",
    re.IGNORECASE,
)

# Single regex that handles both comma-grouped and plain integers.
# The `[\d,]*` middle allows "1,299", "12,345,678", or "1299", and the
# optional decimal tail handles cents. This avoids the alternation-order
# bug that made "1299.00" match as "129".
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class CurrencyResult:
    raw_price: Optional[str] = None
    currency: Optional[str] = None
    normalized_price: Optional[float] = None


def parse_currency(value: Any, default_currency: Optional[str] = None) -> CurrencyResult:
    """Parse a price string into (raw, currency, normalized)."""
    if value in (None, ""):
        return CurrencyResult(raw_price=None, currency=default_currency, normalized_price=None)

    if isinstance(value, (int, float)):
        return CurrencyResult(
            raw_price=str(value),
            currency=default_currency,
            normalized_price=float(value),
        )

    s = normalize_text(str(value))

    # Detect currency code or symbol.
    currency: Optional[str] = None
    m = _CURRENCY_CODE_RE.search(s)
    if m:
        currency = m.group(1).upper()
    else:
        for sym, code in _CURRENCY_SYMBOLS.items():
            if sym in s:
                currency = code
                break
    if currency is None:
        currency = default_currency

    # Extract numeric value.
    num_match = _NUMBER_RE.search(s)
    if not num_match:
        return CurrencyResult(raw_price=s, currency=currency, normalized_price=None)

    try:
        num = float(num_match.group(0).replace(",", ""))
    except ValueError:
        return CurrencyResult(raw_price=s, currency=currency, normalized_price=None)

    return CurrencyResult(raw_price=s, currency=currency, normalized_price=num)


# ---------------------------------------------------------------------------
# Dates (§27.4)
# ---------------------------------------------------------------------------

_ISO_PREFIX_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_DMY_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")


def normalize_date(value: Any) -> Optional[str]:
    """
    Return an ISO 8601 date string (YYYY-MM-DD) if parseable, else None.
    """
    if value in (None, ""):
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        m = _ISO_PREFIX_RE.match(s)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        m = _DMY_RE.match(s)
        if m:
            a, b, year = m.group(1), m.group(2), m.group(3)
            if len(year) == 2:
                year = "20" + year
            day, month = a, b
            try:
                return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Phone (§27.5)
# ---------------------------------------------------------------------------

def normalize_phone(value: Any, default_country: Optional[str] = None) -> Optional[str]:
    """Return E.164 (``+CCNNNNNNN``) when country is known, else None."""
    if value in (None, ""):
        return None
    s = re.sub(r"[^\d+]", "", str(value))
    if not s:
        return None
    if s.startswith("+"):
        digits = s[1:]
        if 8 <= len(digits) <= 15:
            return "+" + digits
        return None
    if not default_country:
        return None
    cc = {"PK": "92", "IN": "91", "US": "1", "GB": "44", "AE": "971"}.get(
        default_country.upper())
    if not cc:
        return None
    digits = s.lstrip("0")
    if not digits:
        return None
    return f"+{cc}{digits}"


# ---------------------------------------------------------------------------
# Email (§27.6)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def normalize_email(value: Any) -> Optional[str]:
    """Validate and normalize: lowercase the domain, preserve local-part."""
    if value in (None, ""):
        return None
    s = str(value).strip()
    if not _EMAIL_RE.match(s):
        return None
    local, _, domain = s.partition("@")
    return f"{local}@{domain.lower()}"


# ---------------------------------------------------------------------------
# Record-level cleaning
# ---------------------------------------------------------------------------

# Field-name → inferred kind. We check substring membership, so a field
# named "published_date" or "release_date" or just "published" all match.
_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "currency": ("price", "cost", "amount", "fee"),
    "date":     ("date", "published", "created", "updated", "modified",
                 "release", "posted", "timestamp", "time"),
    "email":    ("email", "mail"),
    "phone":    ("phone", "tel", "mobile", "contact_number"),
}


def _infer(field_name: str) -> Optional[str]:
    n = field_name.lower()
    for kind, hints in _TYPE_HINTS.items():
        if any(h in n for h in hints):
            return kind
    return None


@dataclass
class CleanedRecord:
    original: dict
    cleaned: dict
    warnings: list[str] = field(default_factory=list)


def clean_record(
    record: dict,
    default_currency: Optional[str] = None,
    default_country: Optional[str] = None,
    preserve_missing_as_sentinel: bool = True,
) -> CleanedRecord:
    """Clean a single record."""
    cleaned: dict = {}
    warnings: list[str] = []

    for fname, value in record.items():
        kind = _infer(fname)

        if value is None or value == "":
            cleaned[fname] = MISSING if preserve_missing_as_sentinel else None
            continue

        if kind == "currency":
            c = parse_currency(value, default_currency=default_currency)
            cleaned[f"{fname}_raw"] = c.raw_price
            cleaned[f"{fname}_currency"] = c.currency
            cleaned[f"{fname}_normalized"] = c.normalized_price
            cleaned[fname] = c.raw_price
            if c.normalized_price is None and c.raw_price:
                warnings.append(f"{fname}: could not normalize {c.raw_price!r}")

        elif kind == "date":
            iso = normalize_date(value)
            cleaned[f"{fname}_raw"] = value
            cleaned[f"{fname}_normalized"] = iso
            cleaned[fname] = normalize_text(value)
            if iso is None:
                warnings.append(f"{fname}: could not parse date {value!r}")

        elif kind == "email":
            norm = normalize_email(value)
            cleaned[fname] = norm if norm else normalize_text(value)
            if norm is None:
                warnings.append(f"{fname}: invalid email {value!r}")

        elif kind == "phone":
            norm = normalize_phone(value, default_country=default_country)
            cleaned[fname] = norm if norm else normalize_text(value)
            if norm is None:
                warnings.append(f"{fname}: could not normalize phone {value!r}")

        else:
            cleaned[fname] = normalize_text(value)

    return CleanedRecord(original=record, cleaned=cleaned, warnings=warnings)


def clean_records(
    records: list[dict],
    default_currency: Optional[str] = None,
    default_country: Optional[str] = None,
) -> tuple[list[dict], list[str]]:
    """Batch clean. Returns (cleaned_records, all_warnings)."""
    out: list[dict] = []
    warnings: list[str] = []
    for i, r in enumerate(records):
        cr = clean_record(r, default_currency=default_currency,
                          default_country=default_country)
        out.append(cr.cleaned)
        for w in cr.warnings:
            warnings.append(f"record[{i}] {w}")
    return out, warnings


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Text
    assert normalize_text("  Hello\u00a0World  ") == "Hello World"
    assert normalize_text("\u201cquoted\u201d") == '"quoted"'
    assert normalize_text("zero\u200bwidth") == "zerowidth"

    # Currency — both comma-grouped and plain
    assert parse_currency("$1,299.00").normalized_price == 1299.0
    assert parse_currency("$1,299.00").currency == "USD"
    assert parse_currency("£51.77").currency == "GBP"
    assert parse_currency("1299.00 USD").currency == "USD"
    assert parse_currency("1299.00 USD").normalized_price == 1299.0
    assert parse_currency("1,299.00 USD").normalized_price == 1299.0
    assert parse_currency("12,345,678.90").normalized_price == 12345678.90
    assert parse_currency(42).normalized_price == 42.0
    assert parse_currency(None).normalized_price is None
    assert parse_currency("Free").normalized_price is None

    # Date
    assert normalize_date("2026-09-22T10:00:00Z") == "2026-09-22"
    assert normalize_date("22/09/2026") == "2026-09-22"
    assert normalize_date("garbage") is None

    # Phone
    assert normalize_phone("+92 300 1234567") == "+923001234567"
    assert normalize_phone("0300-1234567", default_country="PK") == "+923001234567"
    assert normalize_phone("0300-1234567") is None
    assert normalize_phone("") is None

    # Email
    assert normalize_email("Alice@Example.COM") == "Alice@example.com"
    assert normalize_email("not-email") is None

    # Record cleaning — including a `published` field that should be
    # treated as a date via the broadened hint list.
    rec = {
        "title": "  Widget  ",
        "price": "$1,299.00",
        "published": "22/09/2026",
        "email": "Info@Shop.COM",
        "phone": "+92 300 1234567",
        "empty_field": "",
    }
    cr = clean_record(rec)
    assert cr.cleaned["title"] == "Widget"
    assert cr.cleaned["price"] == "$1,299.00"
    assert cr.cleaned["price_normalized"] == 1299.0
    assert cr.cleaned["price_currency"] == "USD"
    assert cr.cleaned["published_normalized"] == "2026-09-22"
    assert cr.cleaned["email"] == "Info@shop.com"
    assert cr.cleaned["phone"] == "+923001234567"
    assert cr.cleaned["empty_field"] is MISSING

    # Missing sentinel
    assert is_missing(MISSING)
    assert not MISSING
    assert is_missing(None)

    print("Cleaning OK.")