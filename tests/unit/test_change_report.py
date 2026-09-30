"""Unit tests for change reports (spec §30)."""
from src.history.change import classify
from src.history.report import ChangeReportBuilder, build_report


def _make_change_set():
    prev = [
        {"title": "A", "price": "$10.00", "availability": "In stock"},
        {"title": "B", "price": "$20.00"},
        {"title": "C", "price": "$30.00"},
    ]
    curr = [
        {"title": "A", "price": "$8.00", "availability": "Out of stock"},
        {"title": "B", "price": "$25.00"},
        {"title": "D", "price": "$15.00"},
    ]
    return classify(prev, curr)


def test_report_counts():
    cs = _make_change_set()
    r = build_report(cs)
    assert r.new_count == 1
    assert r.removed_count == 1
    assert r.modified_count == 2
    assert r.total_records == 4


def test_report_detects_price_direction():
    cs = _make_change_set()
    r = build_report(cs)
    text = " ".join(r.highlights).lower()
    assert "price decrease" in text or "price increase" in text


def test_report_detects_availability_flip():
    cs = _make_change_set()
    r = build_report(cs)
    text = " ".join(r.highlights).lower()
    assert "availability flip" in text


def test_report_renders():
    cs = _make_change_set()
    r = build_report(cs, quality_passed=True, quality_score=0.95, confidence_mean=0.91)
    text = r.render()
    assert "Records:" in text
    assert "New: 1" in text
    assert "Removed: 1" in text
    assert "Modified: 2" in text
    assert "Quality: Passed" in text
    assert "Confidence: mean 0.91" in text


def test_report_shows_failed_quality():
    cs = classify([], [{"title": "A"}])
    r = build_report(cs, quality_passed=False, quality_score=0.3)
    assert "FAILED" in r.render()


def test_report_max_highlights():
    # Build a change set with many field changes
    prev = [{"title": f"r{i}", "a": "old", "b": "old", "c": "old"} for i in range(5)]
    curr = [{"title": f"r{i}", "a": "new", "b": "new", "c": "new"} for i in range(5)]
    cs = classify(prev, curr)
    builder = ChangeReportBuilder(max_highlights=2)
    r = builder.build(cs)
    assert len(r.highlights) <= 2


def test_report_with_no_changes():
    cs = classify([{"title": "A"}], [{"title": "A"}])
    r = build_report(cs)
    assert r.highlights == []
    assert r.new_count == 0
    assert r.unchanged_count == 1