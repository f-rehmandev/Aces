"""
Crawl frontier — spec §14.3A.

In-memory state machine for every URL the crawler has discovered. Tracks
each target through the DISCOVERED → QUEUED → FETCHING → terminal cycle,
enforces include/exclude/depth/same-domain policy at discovery time, and
exposes a checkpoint-shaped snapshot for persistence.

The frontier does no network I/O and keeps no wall-clock state of its own
beyond asking an injected clock for the current time. That makes it fully
testable without a browser or a database.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from src.crawl.types import (
    CrawlPolicy, CrawlState, CrawlTarget,
)
from src.navigation.link_filter import LinkFilter, LinkFilterPolicy
from src.navigation.url_normalizer import normalize_url


logger = logging.getLogger("crawl.frontier")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CrawlFrontier:
    """
    Holds every URL this crawl has discovered, keyed on its canonical form.

    States (see `src.crawl.types`):
        DISCOVERED      — reserved; add() skips straight to QUEUED
        QUEUED          — ready for a worker to pick up
        FETCHING        — a worker is fetching it now
        PROCESSED       — fetched and extracted successfully
        RETRY_WAIT      — transient failure; waiting for `retry_after`
        SKIPPED         — deliberately not fetched (robots, filter, etc.)
        FAILED          — permanent failure (attempts >= policy.max_attempts)
        POLICY_REFUSED  — blocked by SSRF / compliance gate

    Every method is synchronous and side-effect-free apart from touching
    the in-memory dicts. The caller decides when to checkpoint the
    snapshot.
    """

    def __init__(
        self,
        policy: Optional[CrawlPolicy] = None,
        clock: Callable[[], float] = time.monotonic,
        link_filter: Optional[LinkFilter] = None,
    ):
        self.policy = policy or CrawlPolicy()
        self._clock = clock
        self._targets: dict[str, CrawlTarget] = {}
        self._retry_until: dict[str, float] = {}
        self._link_filter = link_filter or LinkFilter(LinkFilterPolicy(
            include_patterns=list(self.policy.include_patterns),
            exclude_patterns=list(self.policy.exclude_patterns),
            same_domain_only=self.policy.same_domain_only,
            max_depth=self.policy.max_depth,
        ))

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._targets)

    def __contains__(self, url: str) -> bool:
        canonical = normalize_url(url).normalized
        return canonical in self._targets

    def get(self, url: str) -> Optional[CrawlTarget]:
        canonical = normalize_url(url).normalized
        return self._targets.get(canonical)

    def all_targets(self) -> list[CrawlTarget]:
        """All targets in insertion (BFS) order."""
        return list(self._targets.values())

    def snapshot(self) -> list[CrawlTarget]:
        """Same content as all_targets; semantically named for checkpointing."""
        return self.all_targets()

    def stats(self) -> dict:
        """Count of targets in each state."""
        counts = {s.value: 0 for s in CrawlState}
        for t in self._targets.values():
            counts[t.state.value] += 1
        return counts

    def has_pending(self) -> bool:
        """True if any target is still QUEUED, FETCHING, or RETRY_WAIT."""
        for t in self._targets.values():
            if t.state in (
                CrawlState.QUEUED,
                CrawlState.FETCHING,
                CrawlState.RETRY_WAIT,
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Adding
    # ------------------------------------------------------------------
    def add(
        self,
        url: str,
        parent_url: str = "",
        depth: int = 0,
    ) -> Optional[CrawlTarget]:
        """
        Canonicalize and add a URL if it passes the policy filter.

        Returns the new CrawlTarget (state=QUEUED), or None if:
          - url is empty
          - url canonicalizes to something non-HTTP(S)
          - url is a duplicate (already in the frontier)
          - url is rejected by the include/exclude/depth/same-domain policy
        """
        if not url:
            return None

        canonical = normalize_url(url).normalized
        if not canonical or not canonical.startswith(("http://", "https://")):
            return None

        if canonical in self._targets:
            return None

        allowed, reason = self._link_filter.allows(
            canonical,
            from_url=parent_url or None,
            depth=depth,
        )
        if not allowed:
            logger.debug(f"Frontier rejected {canonical!r}: {reason}")
            return None

        target = CrawlTarget(
            url=canonical,
            depth=depth,
            parent_url=parent_url or "",
            state=CrawlState.QUEUED,
        )
        self._targets[canonical] = target
        return target

    def add_many(
        self,
        urls: list[str],
        parent_url: str = "",
        depth: int = 0,
    ) -> list[CrawlTarget]:
        """Add many at once; silently drops any that fail policy/dedup."""
        out: list[CrawlTarget] = []
        for u in urls:
            t = self.add(u, parent_url=parent_url, depth=depth)
            if t is not None:
                out.append(t)
        return out

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def next_target(self) -> Optional[CrawlTarget]:
        """
        Promote any RETRY_WAIT targets whose wait has elapsed, then pick
        the oldest QUEUED target, mark it FETCHING, increment attempts,
        and return it. Returns None when nothing is ready.
        """
        self._promote_ready_retries()

        for target in self._targets.values():
            if target.state == CrawlState.QUEUED:
                target.state = CrawlState.FETCHING
                target.attempts += 1
                target.last_attempt_at = _now_iso()
                return target
        return None

    def mark_processed(
        self,
        target: CrawlTarget,
        records_extracted: int = 0,
    ) -> None:
        self._assert_live(target)
        target.state = CrawlState.PROCESSED
        target.records_extracted = records_extracted
        target.error = ""
        self._retry_until.pop(target.url, None)

    def mark_failed(self, target: CrawlTarget, error: str = "") -> None:
        self._assert_live(target)
        target.state = CrawlState.FAILED
        target.error = error
        self._retry_until.pop(target.url, None)

    def mark_skipped(self, target: CrawlTarget, reason: str = "") -> None:
        self._assert_live(target)
        target.state = CrawlState.SKIPPED
        target.notes = reason
        self._retry_until.pop(target.url, None)

    def mark_policy_refused(self, target: CrawlTarget, reason: str = "") -> None:
        self._assert_live(target)
        target.state = CrawlState.POLICY_REFUSED
        target.notes = reason
        self._retry_until.pop(target.url, None)

    def mark_retry(
        self,
        target: CrawlTarget,
        error: str = "",
        retry_after_seconds: float = 0.0,
    ) -> None:
        """
        Handle a transient failure. If attempts < policy.max_attempts,
        move to RETRY_WAIT and schedule a comeback after
        `retry_after_seconds`. Otherwise mark FAILED.
        """
        self._assert_live(target)
        target.error = error

        if target.attempts >= self.policy.max_attempts:
            target.state = CrawlState.FAILED
            self._retry_until.pop(target.url, None)
            return

        target.state = CrawlState.RETRY_WAIT
        delay = max(0.0, float(retry_after_seconds))
        self._retry_until[target.url] = self._clock() + delay

    # ------------------------------------------------------------------
    # Retry introspection
    # ------------------------------------------------------------------
    def min_retry_delay(self) -> Optional[float]:
        """
        Seconds until the earliest RETRY_WAIT target becomes QUEUED again.
        Returns None when no retries are pending.
        """
        if not self._retry_until:
            return None
        now = self._clock()
        remaining = [
            max(0.0, ready_at - now)
            for ready_at in self._retry_until.values()
        ]
        return min(remaining) if remaining else None

    # ------------------------------------------------------------------
    # Checkpoint restore
    # ------------------------------------------------------------------
    def load_from_checkpoint(self, targets: list[CrawlTarget]) -> None:
        """
        Replace the current frontier with a checkpoint's contents.

        Any target that was in FETCHING when the checkpoint was taken
        (i.e. a worker crashed mid-fetch) is moved back to QUEUED so it
        will be re-attempted. RETRY_WAIT bookkeeping is reset — a fresh
        run is free to retry immediately.
        """
        self._targets.clear()
        self._retry_until.clear()
        for t in targets:
            if t.state == CrawlState.FETCHING:
                t.state = CrawlState.QUEUED
            self._targets[t.url] = t

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _promote_ready_retries(self) -> None:
        now = self._clock()
        for url, ready_at in list(self._retry_until.items()):
            if now < ready_at:
                continue
            target = self._targets.get(url)
            if target and target.state == CrawlState.RETRY_WAIT:
                target.state = CrawlState.QUEUED
            self._retry_until.pop(url, None)

    @staticmethod
    def _assert_live(target: CrawlTarget) -> None:
        if target.state in (
            CrawlState.PROCESSED,
            CrawlState.FAILED,
            CrawlState.SKIPPED,
            CrawlState.POLICY_REFUSED,
        ):
            raise ValueError(
                f"target {target.url!r} is already in terminal state "
                f"{target.state.value}"
            )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Empty frontier
    f = CrawlFrontier()
    assert len(f) == 0
    assert f.next_target() is None
    assert not f.has_pending()

    # Add one
    t = f.add("https://example.com/a")
    assert t is not None
    assert t.state == CrawlState.QUEUED
    assert len(f) == 1

    # Duplicate returns None
    assert f.add("https://example.com/a") is None
    assert f.add("https://example.com/a?utm_source=x") is None

    # Canonicalization
    t2 = f.add("HTTPS://EXAMPLE.COM/b?utm_source=y#frag")
    assert t2.url == "https://example.com/b"

    # next_target flow
    got = f.next_target()
    assert got is t
    assert t.state == CrawlState.FETCHING
    assert t.attempts == 1

    # mark_processed
    f.mark_processed(t, records_extracted=7)
    assert t.state == CrawlState.PROCESSED
    assert t.records_extracted == 7

    # Stats
    s = f.stats()
    assert s["PROCESSED"] == 1

    # Retry flow with fake clock
    now = [0.0]
    f2 = CrawlFrontier(
        policy=CrawlPolicy(max_attempts=2),
        clock=lambda: now[0],
    )
    r = f2.add("https://example.com/r")
    f2.next_target()                     # attempts=1, FETCHING
    f2.mark_retry(r, retry_after_seconds=10)
    assert r.state == CrawlState.RETRY_WAIT
    assert f2.next_target() is None      # not ready yet

    now[0] = 11.0                        # advance time
    got = f2.next_target()
    assert got is r
    assert r.attempts == 2

    f2.mark_retry(r)                     # 2 >= 2 → FAILED
    assert r.state == CrawlState.FAILED

    # Filter rejection (same-domain-only default)
    f3 = CrawlFrontier()
    f3.add("https://example.com/start")
    assert f3.add("https://other.com/x",
                  parent_url="https://example.com/start") is None

    # Checkpoint round trip — FETCHING restored as QUEUED
    f4 = CrawlFrontier()
    f4.add_many([
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ])
    f4.next_target()                     # /a → FETCHING
    snap = f4.snapshot()

    f5 = CrawlFrontier()
    f5.load_from_checkpoint(snap)
    assert len(f5) == 3
    assert f5.get("https://example.com/a").state == CrawlState.QUEUED

    print("Crawl frontier OK.")
