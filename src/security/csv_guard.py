"""
CSV / formula injection guard — spec §51.

A scraped value beginning with ``=``, ``+``, ``-``, ``@``, tab, CR, or LF
can be interpreted as a formula by Excel and other spreadsheets. If the
value came from a malicious page, opening the deliverable can execute
attacker-controlled commands.

Defense: prefix dangerous text values with a single quote (') unless the
value is a legitimate number (§51.1).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any


# Danger prefixes per §51.
_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")

# A plain number, optionally with sign and decimal part. These are exempt
# from prefixing so numeric columns still work (§51.1).
_PURE_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$"
)

# ISO-style currency strings like "1299.00 USD" (with one optional currency
# code or symbol) are also exempt — they're number-like, not formulas.
_CURRENCY_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+\.?\d*|\.\d+)\s*(?:USD|EUR|GBP|PKR|INR|AED|SAR|JPY|CNY|CAD|AUD)?$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class CsvGuardResult:
    value: Any
    sanitized: Any
    was_sanitized: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _is_safe_number(s: str) -> bool:
    stripped = s.strip()
    if not stripped:
        return False
    if _PURE_NUMBER_RE.match(stripped):
        return True
    if _CURRENCY_NUMBER_RE.match(stripped):
        return True
    return False


def guard_cell(value: Any) -> CsvGuardResult:
    """
    Sanitize a single cell value against formula injection.

    Non-string values (int, float, None) pass through unchanged.
    Numeric-looking strings are exempt (§51.1).
    """
    if value is None or isinstance(value, (int, float, bool)):
        return CsvGuardResult(value=value, sanitized=value)

    if not isinstance(value, str):
        # e.g. lists, dicts — stringify and re-check? No: caller should
        # have already stringified. Treat as safe to avoid loops.
        return CsvGuardResult(value=value, sanitized=value)

    if not value:
        return CsvGuardResult(value=value, sanitized=value)

    first = value[0]

    # Numeric exemption: if the whole string is a safe number, leave alone.
    if first in "+-" and _is_safe_number(value):
        return CsvGuardResult(value=value, sanitized=value)

    if first in _DANGEROUS_PREFIXES:
        return CsvGuardResult(
            value=value,
            sanitized="'" + value,
            was_sanitized=True,
            reason=f"prefixed with ' (started with {first!r})",
        )

    return CsvGuardResult(value=value, sanitized=value)


def guard_record(record: dict) -> dict:
    """
    Sanitize every value in a record dict. Returns a new dict; the input
    is not mutated.
    """
    out = {}
    for k, v in record.items():
        out[k] = guard_cell(v).sanitized
    return out


def guard_dataset(records: list[dict]) -> tuple[list[dict], int]:
    """
    Sanitize a list of records.

    Returns (sanitized_records, total_cells_sanitized).
    """
    total = 0
    out = []
    for rec in records:
        cleaned = {}
        for k, v in rec.items():
            res = guard_cell(v)
            if res.was_sanitized:
                total += 1
            cleaned[k] = res.sanitized
        out.append(cleaned)
    return out, total


def is_dangerous(value: Any) -> bool:
    """Check whether a value would be sanitized, without sanitizing it."""
    if not isinstance(value, str) or not value:
        return False
    if value[0] in "+-" and _is_safe_number(value):
        return False
    return value[0] in _DANGEROUS_PREFIXES


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Dangerous strings get prefixed
    assert guard_cell("=cmd|'/c calc'!A0").sanitized == "'=cmd|'/c calc'!A0"
    assert guard_cell("+1+1").was_sanitized
    assert guard_cell("@SUM(A1)").was_sanitized
    assert guard_cell("\tfoo").was_sanitized
    assert guard_cell("-5+5").was_sanitized      # not a number

    # Safe numbers pass through
    assert guard_cell(-15.99).sanitized == -15.99
    assert guard_cell("1299.00").sanitized == "1299.00"
    assert guard_cell("+42").sanitized == "+42"        # valid signed number
    assert guard_cell("-3.14").sanitized == "-3.14"
    assert guard_cell("1299.00 USD").sanitized == "1299.00 USD"

    # Non-strings pass through
    assert guard_cell(None).sanitized is None
    assert guard_cell(True).sanitized is True

    # Empty strings
    assert guard_cell("").sanitized == ""

    # Records
    rec = {"name": "=danger", "price": "-15.99", "note": "safe"}
    cleaned = guard_record(rec)
    assert cleaned["name"] == "'=danger"
    assert cleaned["price"] == "-15.99"
    assert cleaned["note"] == "safe"

    # Dataset-level: count how many cells were sanitized
    dataset = [
        {"a": "=x", "b": "1"},
        {"a": "safe", "b": "@y"},
    ]
    cleaned, count = guard_dataset(dataset)
    assert count == 2
    assert cleaned[0]["a"] == "'=x"
    assert cleaned[1]["b"] == "'@y"

    # is_dangerous
    assert is_dangerous("=x")
    assert not is_dangerous("-5")          # valid number
    assert not is_dangerous("plain")

    print("CSV guard OK.")