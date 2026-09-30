"""Unit tests for the Data Quality Engine (spec §22.1, §22.2)."""
import pytest

from src.quality.evaluator import (
    QualityEvaluator, evaluate,
    rule_min_records, rule_failed_page_pct, rule_empty_page_pct,
    rule_populated_field_pct, rule_required_fields,
    rule_type_validity, rule_range_validity, rule_duplicate_rate,
    rule_source_disagreement,
    rule_freshness,
)
from src.quality.rules import QualityRules, value_matches_type



# --- baseline: min_records ---------------------------------------------

def test_min_records_pass():
    r = rule_min_records([{"t": "a"}, {"t": "b"}], QualityRules(min_records=2))
    assert r.passed


def test_min_records_fail():
    r = rule_min_records([{"t": "a"}], QualityRules(min_records=5))
    assert not r.passed


# --- baseline: page pct ------------------------------------------------

def test_failed_page_pct_pass():
    r = rule_failed_page_pct([], QualityRules(
        total_page_count=10, failed_page_count=1, max_failed_page_pct=0.3))
    assert r.passed


def test_failed_page_pct_fail():
    r = rule_failed_page_pct([], QualityRules(
        total_page_count=10, failed_page_count=5, max_failed_page_pct=0.3))
    assert not r.passed


def test_empty_page_pct_pass_when_no_page_data():
    r = rule_empty_page_pct([], QualityRules())
    assert r.passed


# --- baseline: populated field pct -------------------------------------

def test_populated_field_pct_all_full():
    records = [{"title": "A", "price": "$1"}, {"title": "B", "price": "$2"}]
    r = rule_populated_field_pct(records, QualityRules(min_populated_field_pct=0.9))
    assert r.passed
    assert r.value == 1.0


def test_populated_field_pct_half_empty():
    records = [{"title": "A", "price": None}, {"title": "B", "price": "$2"}]
    r = rule_populated_field_pct(records, QualityRules(min_populated_field_pct=0.9))
    assert not r.passed
    assert r.value == 0.75


def test_populated_field_pct_ignores_source_url():
    # source_url is bookkeeping and shouldn't count against quality
    records = [{"title": "A", "source_url": None}]
    r = rule_populated_field_pct(records, QualityRules(min_populated_field_pct=0.9))
    assert r.passed


def test_populated_field_pct_empty_records():
    r = rule_populated_field_pct([], QualityRules())
    assert not r.passed


# --- extended: required fields -----------------------------------------

def test_required_fields_all_present():
    records = [{"title": "A", "price": "$1"}]
    r = rule_required_fields(records, QualityRules(required_fields=["title", "price"]))
    assert r.passed


def test_required_fields_missing():
    records = [{"title": "A", "price": None}]
    r = rule_required_fields(records, QualityRules(required_fields=["title", "price"]))
    assert not r.passed


def test_required_fields_none_declared_passes():
    r = rule_required_fields([{"a": 1}], QualityRules())
    assert r.passed


# --- extended: type validity -------------------------------------------

def test_type_validity_all_correct():
    records = [{"price": "$1.50", "email": "a@b.com"}]
    rules = QualityRules(field_types={"price": "currency", "email": "email"})
    r = rule_type_validity(records, rules)
    assert r.passed


def test_type_validity_email_wrong():
    records = [{"email": "not-an-email"}]
    r = rule_type_validity(records, QualityRules(field_types={"email": "email"}))
    assert not r.passed


def test_type_validity_currency_wrong():
    records = [{"price": "Sold out"}]
    r = rule_type_validity(records, QualityRules(field_types={"price": "currency"}))
    assert not r.passed


def test_type_validity_url():
    records = [{"u": "https://x.com/a"}]
    assert rule_type_validity(records, QualityRules(field_types={"u": "url"})).passed
    records = [{"u": "not-a-url"}]
    assert not rule_type_validity(records, QualityRules(field_types={"u": "url"})).passed


def test_type_validity_ignores_missing():
    records = [{"email": None}]
    r = rule_type_validity(records, QualityRules(field_types={"email": "email"}))
    assert r.passed


def test_value_matches_type_helper():
    assert value_matches_type("$1", "currency")
    assert value_matches_type(42, "number")
    assert value_matches_type("a@b.com", "email")
    assert not value_matches_type("hello", "email")


# --- extended: range validity ------------------------------------------

def test_range_validity_within_bounds():
    records = [{"rating": "4.5"}, {"rating": "3.0"}]
    rules = QualityRules(numeric_bounds={"rating": (0.0, 5.0)})
    assert rule_range_validity(records, rules).passed


def test_range_validity_out_of_bounds():
    records = [{"rating": "9.0"}]
    rules = QualityRules(numeric_bounds={"rating": (0.0, 5.0)})
    assert not rule_range_validity(records, rules).passed


def test_range_validity_no_bounds_declared():
    assert rule_range_validity([{"x": 999}], QualityRules()).passed


# --- extended: duplicate rate ------------------------------------------

def test_duplicate_rate_all_unique():
    records = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
    r = rule_duplicate_rate(records, QualityRules(max_duplicate_rate=0.1))
    assert r.passed


def test_duplicate_rate_too_many_dupes():
    records = [{"title": "A"}, {"title": "A"}, {"title": "A"}, {"title": "B"}]
    r = rule_duplicate_rate(records, QualityRules(max_duplicate_rate=0.2))
    assert not r.passed


def test_duplicate_rate_uses_fallback_identity():
    # Records with business_name but no title still get identity via fallback
    records = [{"business_name": "X"}, {"business_name": "X"}]
    r = rule_duplicate_rate(records, QualityRules(max_duplicate_rate=0.1))
    assert not r.passed


# --- evaluator: pass/fail/score/action ---------------------------------

def test_evaluator_all_pass():
    records = [{"title": f"item-{i}"} for i in range(5)]
    r = evaluate(records, QualityRules(min_records=1))
    assert r.passed
    assert r.score == 1.0
    assert r.suggested_action == "publish"
    assert r.failed_rules == []


def test_evaluator_one_fail_returns_quarantine():
    # fails min_records only
    r = evaluate([], QualityRules(min_records=5))
    assert not r.passed
    assert "min_records" in r.failed_rules
    assert r.suggested_action in ("quarantine", "escalate")


def test_evaluator_retry_action_when_page_failures():
    r = evaluate([{"title": "x"}], QualityRules(
        min_records=1,
        total_page_count=10,
        failed_page_count=8,
        max_failed_page_pct=0.1,
    ))
    assert not r.passed
    assert r.suggested_action == "retry"


def test_evaluator_escalate_when_many_failures():
    r = evaluate([], QualityRules(
        min_records=100,
        field_types={"x": "email"},
        total_page_count=1,
        failed_page_count=1,
        max_failed_page_pct=0.1,
    ))
    assert not r.passed
    assert r.suggested_action in ("retry", "escalate")


def test_evaluator_explanation_is_readable():
    r = evaluate([], QualityRules(min_records=5))
    assert "min_records" in r.explanation


def test_quality_result_to_dict():
    r = evaluate([{"title": "a"}], QualityRules(min_records=1))
    d = r.to_dict()
    assert d["passed"] is True
    assert "per_rule_results" in d



# --- extended: source disagreement -------------------------------------

def test_source_disagreement_not_configured_passes():
    r = rule_source_disagreement(
        [{"title": "A"}],
        QualityRules(),
    )
    assert r.passed
    assert r.value is None


def test_source_disagreement_passes_under_threshold():
    rules = QualityRules(
        max_source_disagreement_rate=0.25,
        source_disagreement_rate=0.20,
    )

    r = rule_source_disagreement(
        [{"title": "A"}],
        rules,
    )

    assert r.passed
    assert r.value == 0.2


def test_source_disagreement_fails_over_threshold():
    rules = QualityRules(
        max_source_disagreement_rate=0.25,
        source_disagreement_rate=0.50,
    )

    r = rule_source_disagreement(
        [{"title": "A"}],
        rules,
    )

    assert not r.passed
    assert r.value == 0.5


# --- extended: freshness -----------------------------------------------

def test_freshness_disabled_passes():
    r = rule_freshness(
        [{"published_at": "2026-09-28T12:00:00Z"}],
        QualityRules(),
    )

    assert r.passed
    assert r.value is None


def test_freshness_recent_timestamp_passes():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    rules = QualityRules(
        freshness_window_seconds=3600,
        freshness_fields=["published_at"],
    )

    r = rule_freshness(
        [{"published_at": now.isoformat()}],
        rules,
    )

    assert r.passed
    assert r.value == 1.0


def test_freshness_stale_timestamp_fails():
    from datetime import datetime, timedelta, timezone

    stale = datetime.now(timezone.utc) - timedelta(days=3)

    rules = QualityRules(
        freshness_window_seconds=3600,
        freshness_fields=["published_at"],
    )

    r = rule_freshness(
        [{"published_at": stale.isoformat()}],
        rules,
    )

    assert not r.passed
    assert r.value == 0.0


def test_freshness_missing_timestamp_fails_when_configured():
    rules = QualityRules(
        freshness_window_seconds=3600,
        freshness_fields=["published_at"],
    )

    r = rule_freshness(
        [{"title": "A"}],
        rules,
    )

    assert not r.passed
    assert r.value == 0.0
    assert "missing" in r.detail


def test_freshness_unparseable_timestamp_fails():
    rules = QualityRules(
        freshness_window_seconds=3600,
        freshness_fields=["published_at"],
    )

    r = rule_freshness(
        [{"published_at": "not-a-timestamp"}],
        rules,
    )

    assert not r.passed
    assert "unparseable" in r.detail


def test_task_spec_quality_round_trip_preserves_new_fields():
    from src.core.task_spec import TaskSpec, Quality

    spec = TaskSpec(
        quality=Quality(
            max_source_disagreement_rate=0.1,
            freshness_window_seconds=7200,
            freshness_fields=[
                "published_at",
                "updated_at",
            ],
            min_freshness_rate=0.8,
        )
    )

    restored = TaskSpec.from_dict(spec.to_dict())

    assert restored.quality.max_source_disagreement_rate == 0.1
    assert restored.quality.freshness_window_seconds == 7200
    assert restored.quality.freshness_fields == [
        "published_at",
        "updated_at",
    ]
    assert restored.quality.min_freshness_rate == 0.8