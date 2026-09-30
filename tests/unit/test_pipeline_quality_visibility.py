"""Tests: failed quality rules are surfaced as warnings, not swallowed."""
from src.quality.evaluator import QualityEvaluator
from src.quality.rules import QualityRules


def test_failed_min_records_appears_in_per_rule_results():
    records = [{"title": "A"}]
    rules = QualityRules(min_records=10)
    result = QualityEvaluator().evaluate(records, rules)
    assert not result.passed
    assert "min_records" in result.failed_rules
    # And the reason is human-readable
    min_rec = next(r for r in result.per_rule_results if r.name == "min_records")
    assert "1" in min_rec.detail  # "1 records (min 10)"
    assert "10" in min_rec.detail


def test_field_completeness_failure_visible():
    records = [{"title": "A", "price": None, "email": None}]
    rules = QualityRules(min_populated_field_pct=0.9)
    result = QualityEvaluator().evaluate(records, rules)
    assert not result.passed
    assert "min_populated_field_pct" in result.failed_rules