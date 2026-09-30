"""
Crawl budget — spec §14.1 (crawl budgets) and §14.3 (hard stops).

Bounds a crawl by any combination of:
    - number of pages fetched
    - bytes fetched
    - wall-clock seconds elapsed

Injection of `now_fn` lets tests simulate time without sleeping.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class CrawlBudget:
    """Limits are set at construction; counters advance as work happens."""

    # --- limits (all optional; None means "no limit") ---
    max_pages: Optional[int] = None
    max_bytes: Optional[int] = None
    max_wall_clock_seconds: Optional[float] = None

    # --- test hook ---
    now_fn: Callable[[], float] = time.monotonic

    # --- live state ---
    pages_fetched: int = 0
    bytes_fetched: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        # Honor a custom now_fn for started_at too.
        self.started_at = self.now_fn()

    # ------------------------------------------------------------------
    # Read-side
    # ------------------------------------------------------------------
    def elapsed_seconds(self) -> float:
        return self.now_fn() - self.started_at

    def can_continue(self) -> tuple[bool, str]:
        """
        Returns (allowed, reason). `reason` is short and machine-friendly,
        matching the style used by `LinkFilter.allows`.
        """
        if self.max_pages is not None and self.pages_fetched >= self.max_pages:
            return False, "max_pages_reached"
        if self.max_bytes is not None and self.bytes_fetched >= self.max_bytes:
            return False, "max_bytes_reached"
        if (
            self.max_wall_clock_seconds is not None
            and self.elapsed_seconds() >= self.max_wall_clock_seconds
        ):
            return False, "max_wall_clock_reached"
        return True, "within_budget"

    # ------------------------------------------------------------------
    # Write-side
    # ------------------------------------------------------------------
    def consume_page(self, byte_count: int = 0) -> None:
        """Record one fetched page (and its byte size, if known)."""
        if byte_count < 0:
            raise ValueError("byte_count must be >= 0")
        self.pages_fetched += 1
        self.bytes_fetched += byte_count

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """For logging / telemetry."""
        return {
            "pages_fetched": self.pages_fetched,
            "bytes_fetched": self.bytes_fetched,
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            "limits": {
                "max_pages": self.max_pages,
                "max_bytes": self.max_bytes,
                "max_wall_clock_seconds": self.max_wall_clock_seconds,
            },
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Pages limit
    b = CrawlBudget(max_pages=2)
    assert b.can_continue() == (True, "within_budget")
    b.consume_page(100)
    b.consume_page(100)
    ok, reason = b.can_continue()
    assert not ok and reason == "max_pages_reached"

    # Bytes limit
    b = CrawlBudget(max_bytes=150)
    b.consume_page(100)
    assert b.can_continue()[0]
    b.consume_page(100)          # total 200 > 150
    ok, reason = b.can_continue()
    assert not ok and reason == "max_bytes_reached"

    # Wall clock, faked
    fake = [0.0]
    b = CrawlBudget(max_wall_clock_seconds=10, now_fn=lambda: fake[0])
    assert b.can_continue()[0]
    fake[0] = 5.0
    assert b.can_continue()[0]
    fake[0] = 10.0
    ok, reason = b.can_continue()
    assert not ok and reason == "max_wall_clock_reached"

    # Negative byte count rejected
    b = CrawlBudget()
    try:
        b.consume_page(-1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # Snapshot
    b = CrawlBudget(max_pages=10)
    b.consume_page(500)
    snap = b.snapshot()
    assert snap["pages_fetched"] == 1
    assert snap["bytes_fetched"] == 500
    assert snap["limits"]["max_pages"] == 10

    print("CrawlBudget OK.")