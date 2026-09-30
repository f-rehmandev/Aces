"""
Crawl frontier types — spec §14.3A, §14.3B.

The crawl engine tracks every discovered URL through an explicit state
machine, so progress can be persisted and a crashed crawl can resume
from the last checkpoint.

State machine (§14.3A):

    DISCOVERED → QUEUED → FETCHING → PROCESSED
                        ├→ RETRY_WAIT
                        ├→ SKIPPED
                        ├→ FAILED
                        └→ POLICY_REFUSED

This module defines the data. The state machine itself lives in
`frontier.py`, persistence in `checkpoint.py`, orchestration in
`engine.py`.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class CrawlState(str, Enum):
    DISCOVERED = "DISCOVERED"          # just added, not yet queued
    QUEUED = "QUEUED"                  # ready to be picked up by a worker
    FETCHING = "FETCHING"              # a worker is currently fetching it
    PROCESSED = "PROCESSED"            # fetched and extracted successfully
    RETRY_WAIT = "RETRY_WAIT"          # transient failure; will retry
    SKIPPED = "SKIPPED"                # excluded by an include/exclude rule
    FAILED = "FAILED"                  # permanent failure (max attempts hit)
    POLICY_REFUSED = "POLICY_REFUSED"  # blocked by compliance or SSRF gate


TERMINAL_STATES = {
    CrawlState.PROCESSED,
    CrawlState.SKIPPED,
    CrawlState.FAILED,
    CrawlState.POLICY_REFUSED,
}


def is_terminal(state: CrawlState) -> bool:
    return state in TERMINAL_STATES


# ---------------------------------------------------------------------------
# CrawlTarget — one URL in the frontier
# ---------------------------------------------------------------------------

@dataclass
class CrawlTarget:
    url: str                                       # canonical form
    depth: int = 0
    parent_url: str = ""                           # which page linked here
    state: CrawlState = CrawlState.DISCOVERED
    attempts: int = 0
    discovered_at: str = field(default_factory=_utc_now_iso)
    last_attempt_at: str = ""
    error: str = ""
    records_extracted: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CrawlTarget":
        d = dict(data)
        d["state"] = CrawlState(d.get("state", "DISCOVERED"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# CrawlPolicy — config for one crawl run
# ---------------------------------------------------------------------------

@dataclass
class CrawlPolicy:
    max_depth: int = 3
    max_pages: int = 1000
    max_bytes: int = 100_000_000
    max_wall_clock_seconds: int = 3600
    same_domain_only: bool = True
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    follow_sitemaps: bool = True
    discover_links: bool = True
    max_attempts: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CrawlPolicy":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# CrawlCheckpoint — resumable state (§14.3B)
# ---------------------------------------------------------------------------

@dataclass
class CrawlCheckpoint:
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_id: str = "default"
    task_id: str = ""
    created_at: str = field(default_factory=_utc_now_iso)

    frontier: list[CrawlTarget] = field(default_factory=list)

    # budgets consumed so far
    budget_pages_used: int = 0
    budget_bytes_used: int = 0
    budget_wall_clock_used: float = 0.0

    # stats
    stats_processed: int = 0
    stats_failed: int = 0
    stats_skipped: int = 0
    stats_policy_refused: int = 0

    last_successful_url: str = ""
    policy: Optional[CrawlPolicy] = None

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "client_id": self.client_id,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "frontier": [t.to_dict() for t in self.frontier],
            "budget_pages_used": self.budget_pages_used,
            "budget_bytes_used": self.budget_bytes_used,
            "budget_wall_clock_used": self.budget_wall_clock_used,
            "stats_processed": self.stats_processed,
            "stats_failed": self.stats_failed,
            "stats_skipped": self.stats_skipped,
            "stats_policy_refused": self.stats_policy_refused,
            "last_successful_url": self.last_successful_url,
            "policy": self.policy.to_dict() if self.policy else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CrawlCheckpoint":
        d = dict(data)
        d["frontier"] = [CrawlTarget.from_dict(t) for t in d.get("frontier", [])]
        if d.get("policy"):
            d["policy"] = CrawlPolicy.from_dict(d["policy"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # State machine
    assert is_terminal(CrawlState.PROCESSED)
    assert is_terminal(CrawlState.FAILED)
    assert not is_terminal(CrawlState.FETCHING)
    assert not is_terminal(CrawlState.QUEUED)

    # CrawlTarget round trip
    t = CrawlTarget(url="https://example.com/a", depth=1)
    assert t.state == CrawlState.DISCOVERED
    d = t.to_dict()
    assert d["state"] == "DISCOVERED"
    t2 = CrawlTarget.from_dict(d)
    assert t2.url == t.url
    assert t2.state == CrawlState.DISCOVERED

    # CrawlPolicy defaults
    p = CrawlPolicy()
    assert p.max_depth == 3
    assert p.max_pages == 1000
    assert p.same_domain_only is True

    # CrawlCheckpoint round trip
    cp = CrawlCheckpoint(
        client_id="acme",
        task_id="t-1",
        frontier=[t, CrawlTarget(url="https://example.com/b", depth=2)],
        budget_pages_used=5,
        stats_processed=3,
        policy=p,
    )
    d = cp.to_dict()
    assert len(d["frontier"]) == 2
    cp2 = CrawlCheckpoint.from_dict(d)
    assert cp2.client_id == "acme"
    assert len(cp2.frontier) == 2
    assert cp2.frontier[0].url == "https://example.com/a"
    assert cp2.policy is not None
    assert cp2.policy.max_depth == 3

    print("Crawl types OK.")
