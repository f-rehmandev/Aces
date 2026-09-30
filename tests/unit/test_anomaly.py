"""Unit tests for anomaly detection (spec §22.4)."""
from src.quality.anomaly import (
    AnomalyDetector, AnomalyFinding, AnomalyReport,
    detect_zscore_outliers, detect_sudden_zero,
    detect_sudden_identical, detect_distribution_shift,
)


# --- z-score -------------------------------------------------------------

def test_zscore_fires_on_large_shift():
    hist = [100, 101, 102, 103, 104, 105]
    curr = [500, 500, 500]
    findings = detect_zscore_outliers(curr, hist, "price", z_threshold=3.0)
    assert findings
    assert findings[0].kind == "zscore"


def test_zscore_quiet_on_small_shift():
    hist = [100, 101, 102, 103, 104, 105]
    curr = [103, 104, 105]
    findings = detect_zscore_outliers(curr, hist, "price", z_threshold=3.0)
    assert not findings


def test_zscore_needs_history():
    findings = detect_zscore_outliers([500], [100], "price")
    assert not findings


def test_zscore_zero_sigma_returns_empty():
    hist = [100, 100, 100, 100, 100]
    findings = detect_zscore_outliers([500], hist, "price")
    assert not findings


def test_zscore_severity_escalates_at_extreme_z():
    hist = [100, 101, 102, 103, 104, 105]
    curr = [10000, 10000, 10000]
    findings = detect_zscore_outliers(curr, hist, "price", z_threshold=3.0)
    assert findings[0].severity == "critical"


# --- sudden zero ---------------------------------------------------------

def test_sudden_zero_fires_when_previously_populated():
    hist = ["a@b.com"] * 10
    curr = [None, None, None]
    f = detect_sudden_zero(curr, hist, "email")
    assert f is not None
    assert f.kind == "sudden_zero"
    assert f.severity == "critical"


def test_sudden_zero_quiet_when_history_also_empty():
    hist = [None] * 10
    curr = [None, None]
    assert detect_sudden_zero(curr, hist, "email") is None


def test_sudden_zero_quiet_when_partially_populated():
    hist = ["a@b.com"] * 10
    curr = ["x@y.com", None, None]
    assert detect_sudden_zero(curr, hist, "email") is None


# --- sudden identical ---------------------------------------------------

def test_sudden_identical_fires_when_previously_varied():
    hist = [f"item-{i}" for i in range(10)]
    curr = ["SAME"] * 5
    f = detect_sudden_identical(curr, hist, "title")
    assert f is not None
    assert f.kind == "sudden_identical"


def test_sudden_identical_quiet_when_history_itself_identical():
    hist = ["SAME"] * 10
    curr = ["SAME"] * 5
    assert detect_sudden_identical(curr, hist, "title") is None


def test_sudden_identical_quiet_when_current_varies():
    hist = [f"item-{i}" for i in range(10)]
    curr = ["a", "b", "c", "d"]
    assert detect_sudden_identical(curr, hist, "title") is None


def test_sudden_identical_needs_at_least_3_values():
    hist = [f"item-{i}" for i in range(10)]
    curr = ["SAME", "SAME"]
    assert detect_sudden_identical(curr, hist, "title") is None


# --- distribution shift -------------------------------------------------

def test_distribution_shift_fires_on_reverse():
    hist = ["A"] * 8 + ["B"] * 2
    curr = ["B"] * 10
    f = detect_distribution_shift(curr, hist, "cat", max_shift=0.4)
    assert f is not None
    assert f.kind == "distribution_shift"


def test_distribution_shift_quiet_when_similar():
    hist = ["A"] * 8 + ["B"] * 2
    curr = ["A"] * 7 + ["B"] * 3
    assert detect_distribution_shift(curr, hist, "cat", max_shift=0.4) is None


# --- detector facade ----------------------------------------------------

def test_detector_returns_report():
    r = AnomalyDetector().detect(
        current_records=[{"price": 500}],
        historical_records=[{"price": i} for i in range(100, 106)],
    )
    assert isinstance(r, AnomalyReport)
    assert r.is_suspicious


def test_detector_empty_history_is_clean():
    r = AnomalyDetector().detect([{"price": 5}], [])
    assert not r.is_suspicious


def test_detector_ignores_bookkeeping_fields():
    r = AnomalyDetector().detect(
        current_records=[{"source_url": "https://x.com/a"}],
        historical_records=[{"source_url": "https://x.com/b"}],
    )
    # source_url is excluded -> no findings from it
    assert not any(f.field_name == "source_url" for f in r.findings)


def test_detector_summary_readable():
    r = AnomalyDetector().detect(
        [{"price": 9999}] * 3,
        [{"price": i} for i in range(100, 106)],
    )
    assert "zscore" in r.summary()


def test_has_critical_flag():
    r = AnomalyDetector().detect(
        [{"title": "SAME"} for _ in range(5)],
        [{"title": f"item-{i}"} for i in range(10)],
    )
    assert r.has_critical