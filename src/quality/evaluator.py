"""
Quality evaluator — spec §22.1 + §22.2.

Takes a list of records and a `QualityRules`, returns a `QualityResult`.

Design:
    - Each rule is a small standalone function returning a `RuleResult`.
    - The evaluator runs them all, aggregates, and decides the action.
    - No historical data required for Round 1 (anomaly detection is Round 2).
"""

from __future__ import annotations
from collections import Counter
from typing import Callable, Iterable, Optional
from datetime import datetime, timezone

from src.quality.rules import (
    QualityRules, QualityResult, RuleResult, value_matches_type,
)


# ---------------------------------------------------------------------------
# Identity for duplicate detection — same priority order as diff_engine.
# ---------------------------------------------------------------------------

_IDENTITY_PRIORITY = ("title", "business_name", "job_title", "name", "url", "source_url")


def _identity_key(record: dict) -> Optional[str]:
    for field in _IDENTITY_PRIORITY:
        v = record.get(field)
        if v:
            return str(v)
    return None


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

def rule_min_records(records: list[dict], rules: QualityRules) -> RuleResult:
    n = len(records)
    return RuleResult(
        name="min_records",
        passed=n >= rules.min_records,
        detail=f"{n} records (min {rules.min_records})",
        value=n,
    )


def rule_failed_page_pct(records: list[dict], rules: QualityRules) -> RuleResult:
    if rules.total_page_count <= 0:
        return RuleResult("max_failed_page_pct", True, "no page data", None)
    pct = rules.failed_page_count / rules.total_page_count
    return RuleResult(
        name="max_failed_page_pct",
        passed=pct <= rules.max_failed_page_pct,
        detail=f"{pct:.0%} failed (max {rules.max_failed_page_pct:.0%})",
        value=round(pct, 3),
    )


def rule_empty_page_pct(records: list[dict], rules: QualityRules) -> RuleResult:
    if rules.total_page_count <= 0:
        return RuleResult("max_empty_page_pct", True, "no page data", None)
    pct = rules.empty_page_count / rules.total_page_count
    return RuleResult(
        name="max_empty_page_pct",
        passed=pct <= rules.max_empty_page_pct,
        detail=f"{pct:.0%} empty (max {rules.max_empty_page_pct:.0%})",
        value=round(pct, 3),
    )


def rule_populated_field_pct(records: list[dict], rules: QualityRules) -> RuleResult:
    """
    For every field that appears at least once in the dataset, what fraction
    of records have a non-null, non-empty value for it? The mean across
    fields must meet `min_populated_field_pct`.
    """
    if not records:
        return RuleResult("min_populated_field_pct", False, "no records", 0.0)

    fields: set[str] = set()
    for r in records:
        fields.update(r.keys())
    # Ignore bookkeeping fields
    fields -= {"source_url", "diff_status", "_numeric_price"}

    if not fields:
        return RuleResult("min_populated_field_pct", True, "no fields to check", 1.0)

    field_pcts: list[float] = []
    for f in fields:
        populated = sum(1 for r in records if r.get(f) not in (None, ""))
        field_pcts.append(populated / len(records))

    mean_pct = sum(field_pcts) / len(field_pcts)
    return RuleResult(
        name="min_populated_field_pct",
        passed=mean_pct >= rules.min_populated_field_pct,
        detail=f"{mean_pct:.0%} populated (min {rules.min_populated_field_pct:.0%})",
        value=round(mean_pct, 3),
    )


def rule_required_fields(records: list[dict], rules: QualityRules) -> RuleResult:
    if not rules.required_fields:
        return RuleResult("required_fields", True, "none required", None)
    missing = {f: 0 for f in rules.required_fields}
    for r in records:
        for f in rules.required_fields:
            if r.get(f) in (None, ""):
                missing[f] += 1
    total_missing = sum(missing.values())
    ok = total_missing == 0
    detail = "all present" if ok else f"missing counts: {missing}"
    return RuleResult("required_fields", ok, detail, total_missing)


def rule_type_validity(records: list[dict], rules: QualityRules) -> RuleResult:
    if not rules.field_types:
        return RuleResult("type_validity", True, "no field types declared", None)
    bad = 0
    examples: list[str] = []
    for i, r in enumerate(records):
        for fname, ftype in rules.field_types.items():
            if fname not in r:
                continue
            if not value_matches_type(r[fname], ftype):
                bad += 1
                if len(examples) < 3:
                    examples.append(f"row {i} {fname}={r[fname]!r} not {ftype}")
    return RuleResult(
        name="type_validity",
        passed=bad == 0,
        detail="ok" if bad == 0 else f"{bad} type violations; e.g. {examples}",
        value=bad,
    )


def rule_range_validity(records: list[dict], rules: QualityRules) -> RuleResult:
    if not rules.numeric_bounds:
        return RuleResult("range_validity", True, "no numeric bounds", None)

    from src.quality.rules import _looks_like_number

    bad = 0
    for r in records:
        for fname, (lo, hi) in rules.numeric_bounds.items():
            v = r.get(fname)
            if v in (None, ""):
                continue
            if not _looks_like_number(v):
                continue   # type_validity's job
            # strip to number
            s = str(v).strip()
            try:
                n = float(s) if not any(c.isalpha() for c in s) else float(
                    "".join(c for c in s if c.isdigit() or c in ".-")
                )
            except (ValueError, TypeError):
                continue
            if not (lo <= n <= hi):
                bad += 1
    return RuleResult(
        name="range_validity",
        passed=bad == 0,
        detail="ok" if bad == 0 else f"{bad} out-of-range values",
        value=bad,
    )


def rule_duplicate_rate(records: list[dict], rules: QualityRules) -> RuleResult:
    if not records:
        return RuleResult("duplicate_rate", True, "no records", 0.0)
    keys = [_identity_key(r) for r in records]
    keys = [k for k in keys if k]
    if not keys:
        return RuleResult("duplicate_rate", True, "no identity keys found", 0.0)
    counts = Counter(keys)
    dupes = sum(c - 1 for c in counts.values() if c > 1)
    rate = dupes / len(records)
    return RuleResult(
        name="duplicate_rate",
        passed=rate <= rules.max_duplicate_rate,
        detail=f"{rate:.0%} duplicates (max {rules.max_duplicate_rate:.0%})",
        value=round(rate, 3),
    )

def rule_source_disagreement(
    records: list[dict],
    rules: QualityRules,
) -> RuleResult:
    """Fail when triangulation reports too many conflicting fields."""
    rate = rules.source_disagreement_rate

    if rate is None:
        return RuleResult(
            "source_disagreement",
            True,
            "no triangulation data",
            None,
        )

    rate = max(0.0, min(1.0, float(rate)))
    passed = rate <= rules.max_source_disagreement_rate

    return RuleResult(
        name="source_disagreement",
        passed=passed,
        detail=(
            f"{rate:.0%} conflicting "
            f"(max {rules.max_source_disagreement_rate:.0%})"
        ),
        value=round(rate, 3),
    )


def rule_freshness(
    records: list[dict],
    rules: QualityRules,
) -> RuleResult:
    """Check configured timestamp fields against the freshness window."""
    if (
        rules.freshness_window_seconds <= 0
        or not rules.freshness_fields
    ):
        return RuleResult(
            "freshness",
            True,
            "not configured",
            None,
        )

    if not records:
        return RuleResult(
            "freshness",
            False,
            "no records",
            0.0,
        )

    now = datetime.now(timezone.utc)

    applicable = len(records)
    fresh = 0
    missing = 0
    unparseable = 0

    for record in records:
        raw = None

        for field_name in rules.freshness_fields:
            value = record.get(field_name)
            if value not in (None, ""):
                raw = value
                break

        if raw is None:
            missing += 1
            continue

        dt = _parse_timestamp(raw)

        if dt is None:
            unparseable += 1
            continue

        age = (now - dt).total_seconds()

        if age <= rules.freshness_window_seconds:
            fresh += 1

    freshness_rate = fresh / applicable

    passed = (
        freshness_rate >= rules.min_freshness_rate
        and missing == 0
        and unparseable == 0
    )

    detail = (
        f"{fresh}/{applicable} records fresh "
        f"(window {rules.freshness_window_seconds}s; "
        f"min {rules.min_freshness_rate:.0%})"
        + (f", {missing} missing" if missing else "")
        + (f", {unparseable} unparseable" if unparseable else "")
    )

    return RuleResult(
        "freshness",
        passed,
        detail,
        round(freshness_rate, 3),
    )


def _parse_timestamp(value) -> Optional[datetime]:
    from src.quality.rules import parse_observation_timestamp
    return parse_observation_timestamp(value)

# Registry — add rules by name here.
_ALL_RULES: list[Callable[[list[dict], QualityRules], RuleResult]] = [
    rule_min_records,
    rule_failed_page_pct,
    rule_empty_page_pct,
    rule_populated_field_pct,
    rule_required_fields,
    rule_type_validity,
    rule_range_validity,
    rule_duplicate_rate,
    rule_source_disagreement,
    rule_freshness,
]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class QualityEvaluator:
    def __init__(self, rules: Optional[list[Callable]] = None):
        self._rules = list(rules) if rules is not None else list(_ALL_RULES)

    def evaluate(self, records: list[dict], rules: QualityRules) -> QualityResult:
        results = [r(records, rules) for r in self._rules]
        failed = [r for r in results if not r.passed]
        passed = len(failed) == 0

        score = self._score(results)
        action = self._suggest_action(passed, score, failed)
        explanation = self._explain(results, passed, score)

        return QualityResult(
            passed=passed,
            score=score,
            per_rule_results=results,
            failed_rules=[r.name for r in failed],
            suggested_action=action,
            explanation=explanation,
        )

    @staticmethod
    def _score(results: list[RuleResult]) -> float:
        """Fraction of rules passed, weighted equally. 0..1."""
        if not results:
            return 1.0
        return round(sum(1 for r in results if r.passed) / len(results), 3)

    @staticmethod
    def _suggest_action(passed: bool, score: float, failed: list[RuleResult]) -> str:
        if passed:
            return "publish"
        # Retry only if the failures look transient (page fetch issues)
        if any(r.name in ("max_failed_page_pct", "max_empty_page_pct") for r in failed):
            return "retry"
        if score < 0.5:
            return "escalate"
        return "quarantine"

    @staticmethod
    def _explain(results: list[RuleResult], passed: bool, score: float) -> str:
        if passed:
            return f"All {len(results)} quality rules passed (score {score:.2f})."
        failed = [r for r in results if not r.passed]
        parts = [f"{r.name}: {r.detail}" for r in failed]
        return f"Quality failed ({len(failed)} rule(s)) — " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_default = QualityEvaluator()

def evaluate(records: list[dict], rules: QualityRules | None = None) -> QualityResult:
    return _default.evaluate(records, rules or QualityRules())