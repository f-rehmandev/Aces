"""
Data Quality Engine — spec §22.

Every run produces a dataset. Before it can replace a previous good
version, it must pass a set of rules. This module implements:

    Baseline rules (§22.1):
        - min_records
        - max_failed_page_pct
        - max_empty_page_pct
        - min_populated_field_pct

    Extended rules (§22.2):
        - schema_consistency   — every record has the declared fields
        - type_validity        — values parse as declared types
        - range_validity       — numeric values within plausible bounds
        - duplicate_rate       — share of records sharing an identity key
        - source_disagreement  — fraction of triangulated field comparisons with conflicts
        - freshness            — fraction of records with configured timestamp fields inside the freshness window

    Anomaly detection (§22.4) lives in its own module (anomaly.py) because
    it needs historical distributions we haven't stored yet.

Each rule returns a RuleResult. `QualityEvaluator` aggregates them into
a `QualityResult` with a score, a list of failed rules, and a suggested
action ("publish" | "quarantine" | "retry" | "escalate").
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime, timezone
import re


# ---------------------------------------------------------------------------
# Rule + result
# ---------------------------------------------------------------------------

@dataclass
class RuleResult:
    name: str
    passed: bool
    detail: str = ""
    value: Any = None          # measured value, for logging


@dataclass
class QualityResult:
    passed: bool
    score: float               # 0..1
    per_rule_results: list[RuleResult] = field(default_factory=list)
    failed_rules: list[str] = field(default_factory=list)
    suggested_action: str = "publish"     # publish | quarantine | retry | escalate
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "failed_rules": list(self.failed_rules),
            "suggested_action": self.suggested_action,
            "explanation": self.explanation,
            "per_rule_results": [
                {"name": r.name, "passed": r.passed, "detail": r.detail, "value": r.value}
                for r in self.per_rule_results
            ],
        }


@dataclass
class QualityRules:
    """Configurable thresholds. Sensible defaults for a small scrape."""

    # §22.1 baseline
    min_records: int = 1
    max_failed_page_pct: float = 0.3
    max_empty_page_pct: float = 0.5
    min_populated_field_pct: float = 0.5

    # §22.2 extended
    max_duplicate_rate: float = 0.2

    # Fraction of triangulated field comparisons allowed to disagree.
    max_source_disagreement_rate: float = 0.25

    # Freshness is disabled unless a window + timestamp field(s) are supplied.
    freshness_window_seconds: int = 0
    freshness_fields: list[str] = field(default_factory=list)
    min_freshness_rate: float = 1.0

    # Runtime measurement supplied by the triangulation layer.
    source_disagreement_rate: Optional[float] = None

    required_fields: list[str] = field(default_factory=list)
    field_types: dict[str, str] = field(default_factory=dict)   # name -> type
    numeric_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # required for extended checks — pass these in when known
    failed_page_count: int = 0
    empty_page_count: int = 0
    total_page_count: int = 0


# ---------------------------------------------------------------------------
# Field type parsing (§27 gives the full normalizers; these are lightweight
# structural checks used only for validation here)
# ---------------------------------------------------------------------------

_PURE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

_CURRENCY_RE = re.compile(
    r"^[^\d\-+]{0,3}\s*[+-]?(?:\d+\.?\d*|\.\d+)\s*[A-Za-z]{0,4}$"
)

_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

_PHONE_DIGITS_RE = re.compile(r"\D")


def _looks_like_number(v: Any) -> bool:
    if isinstance(v, (int, float)):
        return True
    return isinstance(v, str) and bool(_PURE_NUMBER_RE.match(v.strip()))


def _looks_like_currency(v: Any) -> bool:
    if not isinstance(v, str) or not v.strip():
        return False
    return bool(_CURRENCY_RE.match(v.strip()))


def _looks_like_url(v: Any) -> bool:
    return isinstance(v, str) and bool(_URL_RE.match(v.strip()))


def _looks_like_email(v: Any) -> bool:
    return isinstance(v, str) and bool(_EMAIL_RE.match(v.strip()))


def _looks_like_phone(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    digits = _PHONE_DIGITS_RE.sub("", v)
    return 9 <= len(digits) <= 15


_TYPE_CHECKERS: dict[str, Callable[[Any], bool]] = {
    "text":     lambda v: isinstance(v, str),
    "number":   _looks_like_number,
    "currency": _looks_like_currency,
    "url":      _looks_like_url,
    "email":    _looks_like_email,
    "phone":    _looks_like_phone,
    "date":     lambda v: isinstance(v, str) and len(v.strip()) >= 6,
}


def value_matches_type(value: Any, type_name: str) -> bool:
    """Public helper — used by tests and by callers that want one check."""
    check = _TYPE_CHECKERS.get(type_name)
    if check is None:
        return True   # unknown type → don't fail
    if value in (None, ""):
        return True   # missing values are handled by populated_pct, not here
    return check(value)



def parse_observation_timestamp(value: Any) -> Optional[datetime]:
    """Parse common ISO-8601 timestamps into an aware UTC datetime."""
    if value in (None, ""):
        return None

    if isinstance(value, datetime):
        dt = value

    elif isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            # Accept date-only values as midnight UTC.
            try:
                dt = datetime.fromisoformat(
                    raw + "T00:00:00+00:00"
                )
            except ValueError:
                return None

    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)