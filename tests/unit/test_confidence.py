"""Unit tests for confidence scoring (spec §25)."""
from src.trust.confidence import (
    ConfidenceScorer, ConfidenceInputs, ConfidenceScore,
    record_confidence, score, _normalize_source_count,
)


# --- normalization helper ---------------------------------------------

def test_source_count_normalization():
    assert _normalize_source_count(0) == 0.0
    assert _normalize_source_count(1) == 0.2
    assert _normalize_source_count(2) == 0.6
    assert _normalize_source_count(3) == 1.0
    assert _normalize_source_count(10) == 1.0


# --- happy path --------------------------------------------------------

def test_strong_signals_high_confidence():
    s = score(ConfidenceInputs(
        consensus=0.95,
        source_count=3,
        mean_trust=0.9,
        freshness=0.95,
        selector=0.95,
        validation_passed=True,
        historical_stability=0.98,
    ))
    assert s.value >= 0.85
    assert "High confidence" in s.explanation


def test_weak_signals_low_confidence():
    s = score(ConfidenceInputs(
        consensus=0.3,
        source_count=1,
        mean_trust=0.4,
        freshness=0.5,
    ))
    assert s.value < 0.5


def test_value_always_between_zero_and_one():
    for consensus in (0.0, 0.5, 1.0):
        s = score(ConfidenceInputs(consensus=consensus, source_count=3))
        assert 0.0 <= s.value <= 1.0


# --- missing signals ---------------------------------------------------

def test_no_signals_returns_zero_with_message():
    s = score(ConfidenceInputs())
    assert s.value == 0.0
    assert "No signals" in s.explanation


def test_partial_signals_still_scored():
    s = score(ConfidenceInputs(validation_passed=True))
    assert s.value > 0.0
    assert s.value <= 1.0


def test_missing_signals_do_not_penalize():
    # A single strong signal should still score high because weights are
    # renormalised across what's available.
    s = score(ConfidenceInputs(consensus=1.0))
    assert s.value >= 0.8


# --- specific signal semantics ----------------------------------------

def test_validation_failure_lowers_score():
    good = score(ConfidenceInputs(consensus=0.9, validation_passed=True))
    bad = score(ConfidenceInputs(consensus=0.9, validation_passed=False))
    assert bad.value < good.value


def test_anomaly_penalty_reduces_score():
    baseline = score(ConfidenceInputs(consensus=0.9, source_count=3))
    penalized = score(ConfidenceInputs(
        consensus=0.9, source_count=3, anomaly_score=1.0,
    ))
    assert penalized.value < baseline.value


def test_single_source_penalty_in_explanation():
    s = score(ConfidenceInputs(
        consensus=0.9, source_count=1, mean_trust=0.9,
    ))
    assert "only one source" in s.explanation


def test_stale_value_mentioned():
    s = score(ConfidenceInputs(
        consensus=0.9, source_count=3, freshness=0.1,
    ))
    assert "stale" in s.explanation


def test_disagreement_mentioned():
    s = score(ConfidenceInputs(consensus=0.2, source_count=3))
    assert "disagree" in s.explanation


def test_stable_history_mentioned():
    s = score(ConfidenceInputs(
        consensus=0.9, source_count=3, historical_stability=0.98,
    ))
    assert "stable" in s.explanation.lower()


# --- explanation tiers -------------------------------------------------

def test_tier_high():
    s = score(ConfidenceInputs(
        consensus=1.0, source_count=3, mean_trust=1.0,
        freshness=1.0, selector=1.0, validation_passed=True,
        historical_stability=1.0,
    ))
    assert "High confidence" in s.explanation


def test_tier_moderate():
    s = score(ConfidenceInputs(
        consensus=0.7, source_count=2, mean_trust=0.6,
    ))
    assert "Moderate" in s.explanation or "Low" in s.explanation


# --- record roll-up ----------------------------------------------------

def test_record_min_mode():
    fields = {
        "a": ConfidenceScore(value=0.95, inputs=ConfidenceInputs()),
        "b": ConfidenceScore(value=0.6, inputs=ConfidenceInputs()),
        "c": ConfidenceScore(value=0.85, inputs=ConfidenceInputs()),
    }
    assert record_confidence(fields, mode="min") == 0.6


def test_record_mean_mode():
    fields = {
        "a": ConfidenceScore(value=0.9, inputs=ConfidenceInputs()),
        "b": ConfidenceScore(value=0.6, inputs=ConfidenceInputs()),
    }
    assert record_confidence(fields, mode="mean") == 0.75


def test_record_confidence_empty():
    assert record_confidence({}, mode="min") == 0.0
    assert record_confidence({}, mode="mean") == 0.0


# --- custom weights ----------------------------------------------------

def test_custom_weights_change_outcome():
    heavy_consensus = ConfidenceScorer(weights={"consensus": 10.0})
    standard = ConfidenceScorer()
    inputs = ConfidenceInputs(consensus=1.0, validation_passed=False)
    a = heavy_consensus.score(inputs).value
    b = standard.score(inputs).value
    assert a > b