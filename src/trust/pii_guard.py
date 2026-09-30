"""
PII egress guard — spec §27.1.

Before writing records to storage or generating workbooks, this scans all
extracted text fields for inadvertent PII exposure (credit cards,
SSNs/national IDs, credentials).

Rule (§27.1):
    - If a matched field was NOT explicitly requested in the task's schema,
      redact it.
    - Explicitly requested PII (e.g. contact emails for lead-gen tasks) is
      preserved and only validated, never redacted.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ -]*?){13,19}\b"
)

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Generic national ID patterns common outside the US.
_CNIC_RE = re.compile(r"\b\d{5}-\d{7}-\d\b")     # Pakistan
_AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")  # India
_NINO_RE = re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")  # UK

# Credentials / private keys / API-key shaped strings.
_CREDENTIAL_RE = re.compile(
    r"(?:password|passwd|pwd)\s*[:=]\s*\S+"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\bsk-(?:proj-|ant-|or-v1-)?[A-Za-z0-9_\-]{20,}\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PIIRedaction:
    field_name: str
    kind: str              # "credit_card" | "ssn" | "credential" | "national_id"
    sample: str            # redacted preview for logs

    def to_dict(self) -> dict:
        return {"field_name": self.field_name, "kind": self.kind, "sample": self.sample}


@dataclass
class PIIGuardResult:
    record_index: int
    field_name: str
    original_value: str
    redacted_value: str
    kind: str

    def to_dict(self) -> dict:
        return {
            "record_index": self.record_index,
            "field_name": self.field_name,
            "kind": self.kind,
            "original": self.original_value,
            "redacted": self.redacted_value,
        }


@dataclass
class PIIReport:
    records: list[dict] = field(default_factory=list)
    redactions: list[PIIGuardResult] = field(default_factory=list)

    @property
    def has_redactions(self) -> bool:
        return bool(self.redactions)

    def summary(self) -> str:
        if not self.redactions:
            return "no PII detected"
        kinds: dict[str, int] = {}
        for r in self.redactions:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        parts = [f"{n} × {k}" for k, n in sorted(kinds.items())]
        return "PII redacted: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_REDACTED_PLACEHOLDER = "[REDACTED_PII]"


class PIIGuard:
    """
    Scans records and redacts undeclared PII.

    `declared_fields` — fields the user explicitly asked for. PII in those
    fields is preserved (still validated by the caller if desired). If
    empty, ALL PII is redacted.
    """

    def __init__(self, declared_fields: Optional[Iterable[str]] = None):
        self.declared = {f.lower() for f in (declared_fields or [])}

    def _is_declared(self, field_name: str) -> bool:
        return field_name.lower() in self.declared

    @staticmethod
    def _classify(value: str) -> Optional[str]:
        if not isinstance(value, str) or not value:
            return None
        if _CREDIT_CARD_RE.search(value):
            # Only classify as credit card if the digit count matches Luhn-ish length
            digits = re.sub(r"\D", "", value)
            if 13 <= len(digits) <= 19:
                return "credit_card"
        if _SSN_RE.search(value):
            return "ssn"
        if _CNIC_RE.search(value):
            return "national_id"
        if _AADHAAR_RE.search(value):
            return "national_id"
        if _NINO_RE.search(value):
            return "national_id"
        if _CREDENTIAL_RE.search(value):
            return "credential"
        return None

    def scan_record(self, record: dict, record_index: int = -1) -> tuple[dict, list[PIIGuardResult]]:
        """Redact undeclared PII from one record. Returns (cleaned, findings)."""
        cleaned = dict(record)
        findings: list[PIIGuardResult] = []
        for fname, value in record.items():
            if not isinstance(value, str) or not value:
                continue
            if self._is_declared(fname):
                continue
            kind = self._classify(value)
            if kind is None:
                continue
            cleaned[fname] = _REDACTED_PLACEHOLDER
            findings.append(PIIGuardResult(
                record_index=record_index,
                field_name=fname,
                original_value=value,
                redacted_value=_REDACTED_PLACEHOLDER,
                kind=kind,
            ))
        return cleaned, findings

    def scan(self, records: list[dict]) -> PIIReport:
        cleaned_records: list[dict] = []
        redactions: list[PIIGuardResult] = []
        for i, rec in enumerate(records):
            cleaned, findings = self.scan_record(rec, record_index=i)
            cleaned_records.append(cleaned)
            redactions.extend(findings)
        return PIIReport(records=cleaned_records, redactions=redactions)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def scan(records: list[dict], declared_fields: Optional[Iterable[str]] = None) -> PIIReport:
    return PIIGuard(declared_fields=declared_fields).scan(records)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    records = [
        {"title": "Widget", "note": "Card 4111-1111-1111-1111 on file"},
        {"title": "Gadget", "note": "SSN 123-45-6789"},
        {"title": "Thing", "note": "See password=hunter2hunter2"},
        {"title": "Clean", "note": "no pii here"},
    ]

    # Without declared fields -> redact everything
    r = scan(records)
    assert r.has_redactions
    assert r.records[0]["note"] == "[REDACTED_PII]"
    assert r.records[3]["note"] == "no pii here"
    kinds = {redact.kind for redact in r.redactions}
    assert "credit_card" in kinds and "ssn" in kinds and "credential" in kinds
    print(r.summary())

    # Declared field preserved
    r = scan([{"email": "user@example.com", "note": "card 4111-1111-1111-1111"}],
             declared_fields=["email"])
    assert r.records[0]["email"] == "user@example.com"
    assert r.records[0]["note"] == "[REDACTED_PII]"

    # Explicitly declared credit_card field preserved (edge case)
    r = scan([{"credit_card": "4111-1111-1111-1111"}], declared_fields=["credit_card"])
    assert r.records[0]["credit_card"] == "4111-1111-1111-1111"

    # Clean records untouched
    r = scan([{"title": "A"}])
    assert r.records == [{"title": "A"}]
    assert not r.has_redactions

    # _classify unit checks
    assert PIIGuard._classify("4111-1111-1111-1111") == "credit_card"
    assert PIIGuard._classify("123-45-6789") == "ssn"
    assert PIIGuard._classify("hello") is None
    assert PIIGuard._classify("") is None

    print("PII guard OK.")