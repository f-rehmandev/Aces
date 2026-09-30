"""
Anomaly detection — spec §22.4.

Statistical, not LLM-based: cheap and explainable. Four detectors:

    1. per-field z-score vs trailing distribution
    2. sudden zero-rate on a previously-populated field
    3. sudden 100%-identical-value on a previously-varied field
    4. category-distribution shift (chi-square-ish; simplified)

Anomalies do NOT automatically fail the run — they raise the uncertainty
of the quality score and are surfaced in the run summary (per §22.4).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Optional


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

@dataclass
class AnomalyFinding:
    kind: str               # "zscore" | "sudden_zero" | "sudden_identical" | "distribution_shift"
    field_name: str
    detail: str
    severity: str = "info"  # "info" | "warning" | "critical"


@dataclass
class AnomalyReport:
    findings: list[AnomalyFinding] = field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        return bool(self.findings)

    @property
    def has_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.findings)

    def summary(self) -> str:
        if not self.findings:
            return "no anomalies"
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        parts = [f"{n} × {k}" for k, n in sorted(counts.items())]
        return "anomalies: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detect_zscore_outliers(
    current_values: list[float],
    historical_values: list[float],
    field_name: str,
    z_threshold: float = 3.0,
    min_history: int = 5,
) -> list[AnomalyFinding]:
    """
    Field-level z-score of the current mean vs the trailing distribution.
    Fires when the shift is beyond `z_threshold` standard deviations.
    """
    if len(historical_values) < min_history or not current_values:
        return []

    mu = mean(historical_values)
    sigma = pstdev(historical_values)
    if sigma == 0:
        return []

    current_mean = mean(current_values)
    z = (current_mean - mu) / sigma

    if abs(z) >= z_threshold:
        return [AnomalyFinding(
            kind="zscore",
            field_name=field_name,
            detail=(
                f"field {field_name!r} mean shifted to {current_mean:.2f} "
                f"(hist mean {mu:.2f}, sigma {sigma:.2f}, z={z:.2f})"
            ),
            severity="warning" if abs(z) < 5 else "critical",
        )]
    return []


def detect_sudden_zero(
    current_values: list,
    historical_values: list,
    field_name: str,
    historical_min_population: float = 0.8,
) -> Optional[AnomalyFinding]:
    """
    A field that was ≥80% populated historically is now 0% populated.
    """
    if not historical_values or not current_values:
        return None

    hist_pop = sum(1 for v in historical_values if v not in (None, "")) / len(historical_values)
    curr_pop = sum(1 for v in current_values if v not in (None, "")) / len(current_values)

    if hist_pop >= historical_min_population and curr_pop == 0:
        return AnomalyFinding(
            kind="sudden_zero",
            field_name=field_name,
            detail=(
                f"field {field_name!r} was {hist_pop:.0%} populated historically, "
                f"now {curr_pop:.0%}"
            ),
            severity="critical",
        )
    return None


def detect_sudden_identical(
    current_values: list,
    historical_values: list,
    field_name: str,
    min_history_distinct: int = 3,
) -> Optional[AnomalyFinding]:
    """
    A field that used to vary is now 100% identical.
    """
    if not current_values or not historical_values:
        return None

    non_null_current = [v for v in current_values if v not in (None, "")]
    if len(non_null_current) < 3:
        return None

    hist_distinct = len({v for v in historical_values if v not in (None, "")})
    if hist_distinct < min_history_distinct:
        return None

    if len(set(non_null_current)) == 1:
        return AnomalyFinding(
            kind="sudden_identical",
            field_name=field_name,
            detail=(
                f"field {field_name!r} previously had {hist_distinct} distinct values, "
                f"now every record has {non_null_current[0]!r}"
            ),
            severity="critical",
        )
    return None


def detect_distribution_shift(
    current_values: list,
    historical_values: list,
    field_name: str,
    max_shift: float = 0.4,
) -> Optional[AnomalyFinding]:
    """
    Simplified chi-square-ish check on category frequencies.
    Fires when the L1 distance between the two distributions exceeds max_shift.
    """
    if not current_values or not historical_values:
        return None

    def _dist(vals: list) -> dict[str, float]:
        counts: dict[str, int] = {}
        for v in vals:
            if v in (None, ""):
                continue
            counts[str(v)] = counts.get(str(v), 0) + 1
        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: c / total for k, c in counts.items()}

    curr = _dist(current_values)
    hist = _dist(historical_values)
    if not curr or not hist:
        return None

    keys = set(curr) | set(hist)
    l1 = sum(abs(curr.get(k, 0.0) - hist.get(k, 0.0)) for k in keys)
    if l1 >= max_shift:
        return AnomalyFinding(
            kind="distribution_shift",
            field_name=field_name,
            detail=(
                f"field {field_name!r} distribution shifted "
                f"(L1 distance {l1:.2f}, threshold {max_shift})"
            ),
            severity="warning",
        )
    return None


# ---------------------------------------------------------------------------
# Detector facade
# ---------------------------------------------------------------------------

class AnomalyDetector:
    def __init__(
        self,
        z_threshold: float = 3.0,
        distribution_max_shift: float = 0.4,
    ):
        self.z_threshold = z_threshold
        self.distribution_max_shift = distribution_max_shift

    def detect(
        self,
        current_records: list[dict],
        historical_records: list[dict],
        numeric_fields: Optional[list[str]] = None,
    ) -> AnomalyReport:
        """
        Run all detectors. `historical_records` should be the trailing-N
        window (caller decides how many to pass).
        """
        findings: list[AnomalyFinding] = []

        if not historical_records:
            return AnomalyReport()

        fields = self._collect_fields(current_records, historical_records)

        for fname in fields:
            curr = [r.get(fname) for r in current_records]
            hist = [r.get(fname) for r in historical_records]

            # numeric z-score
            curr_nums = [_to_float(v) for v in curr]
            hist_nums = [_to_float(v) for v in hist]
            curr_nums = [v for v in curr_nums if v is not None]
            hist_nums = [v for v in hist_nums if v is not None]
            if curr_nums and hist_nums:
                findings.extend(detect_zscore_outliers(
                    curr_nums, hist_nums, fname, z_threshold=self.z_threshold,
                ))

            # sudden zero
            f = detect_sudden_zero(curr, hist, fname)
            if f:
                findings.append(f)

            # sudden identical
            f = detect_sudden_identical(curr, hist, fname)
            if f:
                findings.append(f)

            # distribution shift (categorical)
            f = detect_distribution_shift(
                curr, hist, fname, max_shift=self.distribution_max_shift,
            )
            if f:
                findings.append(f)

        return AnomalyReport(findings=findings)

    @staticmethod
    def _collect_fields(
        current: list[dict], historical: list[dict],
    ) -> set[str]:
        fields: set[str] = set()
        for r in current:
            fields.update(r.keys())
        for r in historical:
            fields.update(r.keys())
        fields -= {"source_url", "diff_status", "_numeric_price"}
        return fields


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    det = AnomalyDetector()

    # 1. z-score
    hist = [{"price": i} for i in range(100, 106)]   # ~100-105
    curr = [{"price": 500}] * 3                       # huge shift
    r = det.detect(curr, hist)
    assert any(f.kind == "zscore" for f in r.findings), r.findings

    # 2. sudden zero
    hist = [{"email": "a@b.com"} for _ in range(10)]
    curr = [{"email": None} for _ in range(3)]
    r = det.detect(curr, hist)
    assert any(f.kind == "sudden_zero" for f in r.findings)

    # 3. sudden identical
    hist = [{"title": f"item-{i}"} for i in range(10)]
    curr = [{"title": "SAME"} for _ in range(5)]
    r = det.detect(curr, hist)
    assert any(f.kind == "sudden_identical" for f in r.findings)

    # 4. distribution shift
    hist = [{"cat": "A"} for _ in range(8)] + [{"cat": "B"} for _ in range(2)]
    curr = [{"cat": "B"} for _ in range(10)]
    r = det.detect(curr, hist)
    assert any(f.kind == "distribution_shift" for f in r.findings)

    # 5. No history -> clean
    r = det.detect([{"price": 5}], [])
    assert not r.is_suspicious

    # 6. Similar datasets -> clean
    same = [{"t": f"x-{i}", "p": i} for i in range(5)]
    r = det.detect(same, same)
    # no sudden zero, no sudden identical (values vary), no extreme z
    assert not any(f.kind in ("sudden_zero", "sudden_identical") for f in r.findings)

    print("Anomaly detector OK.")