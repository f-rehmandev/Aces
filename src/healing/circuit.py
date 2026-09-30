"""
Healing circuit breaker — spec §17.2.1.

Caps multimodal healing events per run per domain (default 3). If a
target page breaks three separate selectors within a single run,
automated vision repair halts and the system falls through to
Rung 4 (graceful failure). This prevents hallucination loops where
an LLM repeatedly generates brittle selectors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_MAX_PER_DOMAIN = 3


@dataclass
class CircuitSnapshot:
    domain: str
    healing_attempts: int
    tripped: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "healing_attempts": self.healing_attempts,
            "tripped": self.tripped,
            "reason": self.reason,
        }


class HealingCircuitBreaker:
    """
    Tracks healing events per domain. Call `record_attempt(domain)` before
    each healing try; `can_attempt(domain)` gates whether another try is
    allowed. The breaker is per-run: construct a fresh one for each run.
    """

    def __init__(self, max_per_domain: int = DEFAULT_MAX_PER_DOMAIN):
        if max_per_domain < 0:
            raise ValueError("max_per_domain must be >= 0")
        self.max_per_domain = max_per_domain
        self._counts: dict[str, int] = {}
        self._tripped: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Query / mutate
    # ------------------------------------------------------------------
    def can_attempt(self, domain: str) -> tuple[bool, str]:
        """Return (allowed, reason)."""
        d = (domain or "").lower()
        if d in self._tripped:
            return False, self._tripped[d]
        count = self._counts.get(d, 0)
        if count >= self.max_per_domain:
            reason = (
                f"healing loop detected: {count} attempts on "
                f"{d} (max {self.max_per_domain})"
            )
            self._tripped[d] = reason
            return False, reason
        return True, "under_limit"

    def record_attempt(self, domain: str) -> int:
        """Record a healing attempt. Returns the new count."""
        d = (domain or "").lower()
        self._counts[d] = self._counts.get(d, 0) + 1
        if self._counts[d] >= self.max_per_domain and d not in self._tripped:
            self._tripped[d] = (
                f"healing loop detected: {self._counts[d]} attempts on "
                f"{d} (max {self.max_per_domain})"
            )
        return self._counts[d]

    def attempts_for(self, domain: str) -> int:
        return self._counts.get((domain or "").lower(), 0)

    def is_tripped(self, domain: str) -> bool:
        return (domain or "").lower() in self._tripped

    def snapshot(self, domain: str) -> CircuitSnapshot:
        d = (domain or "").lower()
        return CircuitSnapshot(
            domain=d,
            healing_attempts=self._counts.get(d, 0),
            tripped=self.is_tripped(d),
            reason=self._tripped.get(d, ""),
        )

    # ------------------------------------------------------------------
    # Reset (useful for tests; production uses fresh instance per run)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._counts.clear()
        self._tripped.clear()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cb = HealingCircuitBreaker(max_per_domain=3)

    # Under limit
    allowed, reason = cb.can_attempt("shop.example")
    assert allowed and reason == "under_limit"

    # Record 1, 2, 3 attempts
    assert cb.record_attempt("shop.example") == 1
    assert cb.can_attempt("shop.example")[0] is True

    assert cb.record_attempt("shop.example") == 2
    assert cb.can_attempt("shop.example")[0] is True

    # Third attempt trips
    assert cb.record_attempt("shop.example") == 3
    allowed, reason = cb.can_attempt("shop.example")
    assert not allowed
    assert "healing loop detected" in reason

    # Different domain unaffected
    assert cb.can_attempt("other.example")[0] is True

    # Snapshot
    snap = cb.snapshot("shop.example")
    assert snap.tripped
    assert snap.healing_attempts == 3

    # Explicit trip persists after a reset of counters (in this design
    # the trip is sticky for the run)
    assert cb.is_tripped("shop.example")

    # reset clears everything
    cb.reset()
    assert not cb.is_tripped("shop.example")
    assert cb.attempts_for("shop.example") == 0

    # invalid max
    try:
        HealingCircuitBreaker(max_per_domain=-1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    print("Healing circuit breaker OK.")