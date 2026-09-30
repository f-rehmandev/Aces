"""
Change reports — spec §30.

Turns a `ChangeSet` into a plain-text summary that can ship as a
first-class deliverable alongside the dataset. Also provides business-level
interpretation per §29.2 ("page HTML changed" vs "record changed").
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from src.history.change import ChangeSet, RecordChange


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------

@dataclass
class ChangeReport:
    title: str = "Change Report"
    total_records: int = 0
    new_count: int = 0
    removed_count: int = 0
    modified_count: int = 0
    unchanged_count: int = 0
    highlights: list[str] = field(default_factory=list)
    quality_passed: bool = True
    quality_score: float = 0.0
    confidence_mean: float = 0.0

    def render(self) -> str:
        lines = [self.title, "─" * 29]
        lines.append(f"Records: {self.total_records:,}")
        lines.append(f"New: {self.new_count}")
        lines.append(f"Removed: {self.removed_count}")
        lines.append(f"Modified: {self.modified_count}")
        lines.append(f"Unchanged: {self.unchanged_count}")
        if self.highlights:
            lines.append("")
            lines.append("Top changes:")
            for h in self.highlights:
                lines.append(f"  • {h}")
        lines.append("")
        q = "Passed" if self.quality_passed else "FAILED"
        lines.append(f"Quality: {q} (score {self.quality_score:.2f})")
        if self.confidence_mean:
            lines.append(f"Confidence: mean {self.confidence_mean:.2f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class ChangeReportBuilder:
    """Turn a ChangeSet into a ChangeReport with business-level highlights."""

    def __init__(self, max_highlights: int = 5):
        self.max_highlights = max_highlights

    def build(
        self,
        change_set: ChangeSet,
        quality_passed: bool = True,
        quality_score: float = 0.0,
        confidence_mean: float = 0.0,
        title: str = "Change Report",
    ) -> ChangeReport:
        report = ChangeReport(
            title=title,
            total_records=change_set.total,
            new_count=len(change_set.new),
            removed_count=len(change_set.removed),
            modified_count=len(change_set.modified),
            unchanged_count=len(change_set.unchanged),
            quality_passed=quality_passed,
            quality_score=quality_score,
            confidence_mean=confidence_mean,
        )
        report.highlights = self._build_highlights(change_set)
        return report

    # ------------------------------------------------------------------
    # Business-level highlights (§29.2)
    # ------------------------------------------------------------------
    def _build_highlights(self, cs: ChangeSet) -> list[str]:
        from collections import defaultdict

        # Group modified records by field name → count of changes.
        field_counts: dict[str, int] = defaultdict(int)
        price_up = 0
        price_down = 0
        availability_flips = 0

        for rc in cs.modified:
            for fc in rc.field_changes:
                field_counts[fc.field_name] += 1
                fname = fc.field_name.lower()
                if "price" in fname or "cost" in fname:
                    old = _to_float(fc.old_value)
                    new = _to_float(fc.new_value)
                    if old is not None and new is not None:
                        if new > old:
                            price_up += 1
                        elif new < old:
                            price_down += 1
                if "availab" in fname or "stock" in fname:
                    availability_flips += 1

        highlights: list[str] = []
        if price_down:
            highlights.append(f"{price_down} price decrease(s)")
        if price_up:
            highlights.append(f"{price_up} price increase(s)")
        if availability_flips:
            highlights.append(f"{availability_flips} availability flip(s)")

        # Any other field with changes beyond price/availability.
        for fname, count in sorted(field_counts.items(), key=lambda kv: -kv[1]):
            low = fname.lower()
            if "price" in low or "cost" in low or "availab" in low or "stock" in low:
                continue
            highlights.append(f"{count} {fname} change(s)")

        return highlights[: self.max_highlights]


def _to_float(value) -> Optional[float]:
    import re
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_report(change_set: ChangeSet, **kwargs) -> ChangeReport:
    return ChangeReportBuilder().build(change_set, **kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.history.change import classify

    previous = [
        {"title": "A", "price": "$10.00", "availability": "In stock"},
        {"title": "B", "price": "$20.00", "availability": "In stock"},
        {"title": "C", "price": "$30.00"},
    ]
    current = [
        {"title": "A", "price": "$8.00", "availability": "Out of stock"},
        {"title": "B", "price": "$25.00", "availability": "In stock"},
        {"title": "D", "price": "$15.00"},
    ]

    cs = classify(previous, current)
    report = ChangeReportBuilder().build(
        cs, quality_passed=True, quality_score=0.95, confidence_mean=0.91,
    )
    print(report.render())
    print()

    assert report.new_count == 1
    assert report.removed_count == 1
    assert report.modified_count == 2
    assert any("price decrease" in h for h in report.highlights)
    assert any("price increase" in h for h in report.highlights)
    assert any("availability flip" in h for h in report.highlights)

    # Report renders cleanly
    text = report.render()
    assert "Records:" in text
    assert "Quality: Passed" in text
    assert "Confidence: mean 0.91" in text

    print("ChangeReport OK.")