"""
Retry policy — spec §39.

Not every failure should be retried. This module decides:
    - retryable vs non-retryable (§39.1, §39.2)
    - backoff schedule (§39.3: 0, 2, 4, 8, 16 seconds)
    - jitter to prevent thundering herd
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Failure classes
# ---------------------------------------------------------------------------

class RetryClass(str, Enum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"


# §39.1 retryable
_RETRYABLE = {
    "TRANSIENT", "TIMEOUT", "HTTP_5XX", "RATE_LIMIT_TEMP",
    "BROWSER_START", "NETWORK",
}

# §39.2 non-retryable
_NON_RETRYABLE = {
    "NOT_FOUND_404", "AUTHORIZATION", "CAPTCHA_HARD",
    "SCHEMA_MISMATCH", "COMPLIANCE_REFUSED", "SSRF_BLOCKED",
    "BUDGET_TRIPPED",
}


def classify_failure(failure_type: str) -> RetryClass:
    key = failure_type.upper()
    if key in _RETRYABLE:
        return RetryClass.RETRYABLE
    if key in _NON_RETRYABLE:
        return RetryClass.NON_RETRYABLE
    # Unknown → retry once, then give up (conservative default)
    return RetryClass.NON_RETRYABLE


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    max_attempts: int = 4
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 60.0
    jitter: bool = True
    # Fixed schedule from §39.3, used when `use_linear_schedule=False`
    explicit_schedule: list[float] = field(
        default_factory=lambda: [0.0, 2.0, 4.0, 8.0, 16.0]
    )


def delay_for_attempt(
    attempt: int,
    policy: RetryPolicy | None = None,
    rng: random.Random | None = None,
) -> float:
    """
    Return seconds to wait before `attempt` (1-indexed).
    Attempt 1 = the first try → delay 0 (no wait).
    Attempt 2 → base_delay.
    Then exponential: base * 2^(attempt-2), capped at max_delay.
    """
    policy = policy or RetryPolicy()
    if attempt <= 1:
        return 0.0

    # Prefer explicit schedule if provided
    if policy.explicit_schedule:
        idx = attempt - 1
        if idx < len(policy.explicit_schedule):
            base = policy.explicit_schedule[idx]
        else:
            base = min(
                policy.max_delay_seconds,
                policy.base_delay_seconds * (2 ** (attempt - 2)),
            )
    else:
        base = min(
            policy.max_delay_seconds,
            policy.base_delay_seconds * (2 ** (attempt - 2)),
        )

    if policy.jitter and base > 0:
        rng = rng or random
        # Jitter: ±20%
        base = base * rng.uniform(0.8, 1.2)

    return round(base, 3)


def should_retry(
    failure_type: str,
    attempts_so_far: int,
    policy: RetryPolicy | None = None,
) -> tuple[bool, str]:
    """
    Decide whether to retry.

    Returns (retry, reason). `attempts_so_far` is the count of tries already
    made (including the failing one).
    """
    policy = policy or RetryPolicy()
    klass = classify_failure(failure_type)
    if klass == RetryClass.NON_RETRYABLE:
        return False, f"non-retryable failure: {failure_type}"
    if attempts_so_far >= policy.max_attempts:
        return False, f"max attempts ({policy.max_attempts}) reached"
    return True, "retryable"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Classification
    assert classify_failure("TIMEOUT") == RetryClass.RETRYABLE
    assert classify_failure("HTTP_5XX") == RetryClass.RETRYABLE
    assert classify_failure("RATE_LIMIT_TEMP") == RetryClass.RETRYABLE
    assert classify_failure("NOT_FOUND_404") == RetryClass.NON_RETRYABLE
    assert classify_failure("CAPTCHA_HARD") == RetryClass.NON_RETRYABLE
    assert classify_failure("BUDGET_TRIPPED") == RetryClass.NON_RETRYABLE
    assert classify_failure("UNKNOWN_RANDOM") == RetryClass.NON_RETRYABLE

    # Delay schedule
    assert delay_for_attempt(1) == 0.0
    assert delay_for_attempt(2) > 0
    assert delay_for_attempt(3) > delay_for_attempt(2)

    # No-jitter exact values
    p = RetryPolicy(jitter=False)
    assert delay_for_attempt(1, p) == 0.0
    assert delay_for_attempt(2, p) == 2.0
    assert delay_for_attempt(3, p) == 4.0
    assert delay_for_attempt(4, p) == 8.0
    assert delay_for_attempt(5, p) == 16.0

    # Jitter deterministic with seeded RNG
    import random as _r
    r = _r.Random(42)
    d1 = delay_for_attempt(2, RetryPolicy(jitter=True), rng=r)
    d2 = delay_for_attempt(2, RetryPolicy(jitter=True), rng=_r.Random(42))
    assert d1 == d2

    # should_retry
    ok, _ = should_retry("TIMEOUT", attempts_so_far=1, policy=RetryPolicy(max_attempts=3))
    assert ok
    ok, reason = should_retry("TIMEOUT", attempts_so_far=3, policy=RetryPolicy(max_attempts=3))
    assert not ok and "max attempts" in reason
    ok, reason = should_retry("NOT_FOUND_404", attempts_so_far=1)
    assert not ok and "non-retryable" in reason

    print("Retry policy OK.")