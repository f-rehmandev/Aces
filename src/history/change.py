"""
Change classification — spec §29.

Given a previous dataset and a current dataset, classify every record as
NEW, REMOVED, MODIFIED, or UNCHANGED, and for modified records produce a
field-level diff.

Noise suppression (§29.3) is on by default:
    - reordering without value change → not a change
    - whitespace / case-only differences → normalized away
    - currency formatting with same numeric value → not a change
    - timestamps-only changes → suppressed unless requested
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import re


# ---------------------------------------------------------------------------
# Identity (matches diff_engine / quality / provenance priority order)
# ---------------------------------------------------------------------------

_IDENTITY_PRIORITY = ("title", "business_name", "job_title", "name", "url", "source_url")


def identity_key(record: dict) -> Optional[str]:
    for key in _IDENTITY_PRIORITY:
        v = record.get(key)
        if v:
            return str(v)
    return None


# ---------------------------------------------------------------------------
# Normalization for comparison (§29.3)
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_SUPPRESSED_FIELDS_DEFAULT = {
    "source_url",
    "diff_status",
    "_numeric_price",
    # Derived bookkeeping — an assessment of the data, not the data itself.
    # A change in confidence alone should not read as a modified record.
    "confidence",
}


def _normalize_for_compare(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    s = _WHITESPACE_RE.sub(" ", str(value).strip())
    s = s.lower()
    return s


def _normalize_price(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _NUMBER_RE.search(str(value))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _is_price_field(name: str) -> bool:
    n = name.lower()
    return "price" in n or "cost" in n or "amount" in n


def _is_timestamp_field(name: str) -> bool:
    n = name.lower()
    return "timestamp" in n or n.endswith("_at") or "updated" in n


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class FieldChange:
    field_name: str
    old_value: Any
    new_value: Any

    def to_dict(self) -> dict:
        return {"field": self.field_name, "old": self.old_value, "new": self.new_value}


@dataclass
class RecordChange:
    change_type: str           # "NEW" | "REMOVED" | "MODIFIED" | "UNCHANGED"
    identity: str
    record: dict               # the current (or removed) record
    field_changes: list[FieldChange] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "change_type": self.change_type,
            "identity": self.identity,
            "record": self.record,
            "field_changes": [fc.to_dict() for fc in self.field_changes],
        }


@dataclass
class ChangeSet:
    new: list[RecordChange] = field(default_factory=list)
    removed: list[RecordChange] = field(default_factory=list)
    modified: list[RecordChange] = field(default_factory=list)
    unchanged: list[RecordChange] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.new) + len(self.removed) + len(self.modified) + len(self.unchanged)

    @property
    def has_changes(self) -> bool:
        return bool(self.new or self.removed or self.modified)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "new": len(self.new),
                "removed": len(self.removed),
                "modified": len(self.modified),
                "unchanged": len(self.unchanged),
                "total": self.total,
            },
            "new": [r.to_dict() for r in self.new],
            "removed": [r.to_dict() for r in self.removed],
            "modified": [r.to_dict() for r in self.modified],
            "unchanged": [r.to_dict() for r in self.unchanged],
        }


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

class ChangeClassifier:
    """
    Compare two datasets.

    `suppress_timestamps` — if True (default), fields ending in `_at` or
    containing `timestamp` are ignored for change detection (§29.3).
    """

    def __init__(self, suppress_timestamps: bool = True):
        self.suppress_timestamps = suppress_timestamps

    def classify(self, previous: list[dict], current: list[dict]) -> ChangeSet:
        cs = ChangeSet()

        prev_by_id: dict[str, dict] = {}
        for r in previous:
            k = identity_key(r)
            if k:
                prev_by_id[k] = r

        curr_by_id: dict[str, dict] = {}
        for r in current:
            k = identity_key(r)
            if k:
                curr_by_id[k] = r

        # --- NEW / MODIFIED / UNCHANGED ---
        for rid, rec in curr_by_id.items():
            prev = prev_by_id.get(rid)
            if prev is None:
                cs.new.append(RecordChange("NEW", rid, rec))
                continue

            diffs = self._field_changes(prev, rec)
            if diffs:
                cs.modified.append(RecordChange("MODIFIED", rid, rec, diffs))
            else:
                cs.unchanged.append(RecordChange("UNCHANGED", rid, rec))

        # --- REMOVED ---
        for rid, rec in prev_by_id.items():
            if rid not in curr_by_id:
                cs.removed.append(RecordChange("REMOVED", rid, rec))

        return cs

    # ------------------------------------------------------------------
    # Field-level diff with noise suppression
    # ------------------------------------------------------------------
    def _field_changes(self, prev: dict, curr: dict) -> list[FieldChange]:
        changes: list[FieldChange] = []

        # Union of fields on both sides, minus suppressed bookkeeping.
        fields = set(prev.keys()) | set(curr.keys())
        fields -= _SUPPRESSED_FIELDS_DEFAULT
        if self.suppress_timestamps:
            fields = {f for f in fields if not _is_timestamp_field(f)}

        for f in sorted(fields):
            old = prev.get(f)
            new = curr.get(f)

            # Price fields compare numerically if both parse as numbers.
            if _is_price_field(f):
                old_n = _normalize_price(old)
                new_n = _normalize_price(new)
                if old_n is not None and new_n is not None:
                    if old_n != new_n:
                        changes.append(FieldChange(f, old, new))
                    continue

            # Fall back to normalized string comparison.
            if _normalize_for_compare(old) != _normalize_for_compare(new):
                # Ignore whitespace/case-only differences on plain text.
                old_s = str(old).strip() if old is not None else ""
                new_s = str(new).strip() if new is not None else ""
                if old_s.lower() == new_s.lower():
                    continue
                changes.append(FieldChange(f, old, new))

        return changes


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_default = ChangeClassifier()

def classify(previous: list[dict], current: list[dict]) -> ChangeSet:
    return _default.classify(previous, current)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    previous = [
        {"title": "A", "price": "$10.00", "note": "in stock"},
        {"title": "B", "price": "$20.00"},
        {"title": "C", "price": "$30.00", "note": "discontinued"},
    ]
    current = [
        {"title": "A", "price": "$10.00", "note": "IN STOCK"},     # case-only diff -> unchanged
        {"title": "B", "price": "$25.00"},                          # price changed
        {"title": "D", "price": "$15.00"},                          # new
        # C is gone -> removed
    ]

    cs = classify(previous, current)
    assert len(cs.new) == 1 and cs.new[0].identity == "D"
    assert len(cs.removed) == 1 and cs.removed[0].identity == "C"
    assert len(cs.modified) == 1 and cs.modified[0].identity == "B"
    assert cs.modified[0].field_changes[0].field_name == "price"
    assert len(cs.unchanged) == 1 and cs.unchanged[0].identity == "A"

    # Timestamp suppression
    prev = [{"title": "X", "updated_at": "2026-01-01T00:00:00"}]
    curr = [{"title": "X", "updated_at": "2026-09-24T00:00:00"}]
    assert not classify(prev, curr).has_changes

    # Currency-formatting equivalence
    prev = [{"title": "Y", "price": "$1,299.00"}]
    curr = [{"title": "Y", "price": "1299.00 USD"}]
    assert not classify(prev, curr).has_changes

    # to_dict
    d = cs.to_dict()
    assert d["summary"]["new"] == 1
    assert d["summary"]["removed"] == 1
    assert d["summary"]["modified"] == 1
    assert d["summary"]["unchanged"] == 1

    print("ChangeClassifier OK.")