"""Unit tests for retry policy (spec §39)."""
import random

from src.ops.retry import (
    RetryClass, RetryPolicy, classify_failure, delay_for_attempt, should_retry,
)


# --- classification ---------------------------------------------------

def test_retryable_types():
    for t in ("TIMEOUT", "HTTP_5XX", "RATE_LIMIT_TEMP", "TRANSIENT", "NETWORK"):
        assert classify_failure(t) == RetryClass.RETRYABLE


def test_non_retryable_types():
    for t in ("NOT_FOUND_404", "AUTHORIZATION", "CAPTCHA_HARD",
              "SCHEMA_MISMATCH", "COMPLIANCE_REFUSED", "SSRF_BLOCKED",
              "BUDGET_TRIPPED"):
        assert classify_failure(t) == RetryClass.NON_RETRYABLE


def test_unknown_defaults_to_non_retryable():
    assert classify_failure("WTF") == RetryClass.NON_RETRYABLE


def test_classification_case_insensitive():
    assert classify_failure("timeout") == RetryClass.RETRYABLE
    assert classify_failure("not_found_404") == RetryClass.NON_RETRYABLE


# --- delay schedule ---------------------------------------------------

def test_first_attempt_has_no_delay():
    assert delay_for_attempt(1, RetryPolicy(jitter=False)) == 0.0


def test_explicit_schedule_no_jitter():
    p = RetryPolicy(jitter=False)
    assert delay_for_attempt(2, p) == 2.0
    assert delay_for_attempt(3, p) == 4.0
    assert delay_for_attempt(4, p) == 8.0
    assert delay_for_attempt(5, p) == 16.0


def test_delay_grows_with_attempt():
    p = RetryPolicy(jitter=False)
    assert delay_for_attempt(2, p) < delay_for_attempt(3, p)


def test_max_delay_cap():
    p = RetryPolicy(jitter=False, base_delay_seconds=100,
                    explicit_schedule=[], max_delay_seconds=30)
    assert delay_for_attempt(5, p) <= 30


def test_jitter_is_deterministic_with_seed():
    a = delay_for_attempt(2, RetryPolicy(jitter=True), rng=random.Random(1))
    b = delay_for_attempt(2, RetryPolicy(jitter=True), rng=random.Random(1))
    assert a == b


def test_jitter_stays_within_bounds():
    p = RetryPolicy(jitter=True)
    for seed in range(50):
        v = delay_for_attempt(2, p, rng=random.Random(seed))
        assert 1.6 <= v <= 2.4   # 2.0 ± 20%


# --- should_retry -----------------------------------------------------

def test_should_retry_retryable_under_max():
    ok, _ = should_retry("TIMEOUT", attempts_so_far=1,
                         policy=RetryPolicy(max_attempts=3))
    assert ok


def test_should_not_retry_when_max_reached():
    ok, reason = should_retry("TIMEOUT", attempts_so_far=3,
                              policy=RetryPolicy(max_attempts=3))
    assert not ok
    assert "max attempts" in reason


def test_should_not_retry_non_retryable():
    ok, reason = should_retry("CAPTCHA_HARD", attempts_so_far=1)
    assert not ok
    assert "non-retryable" in reason