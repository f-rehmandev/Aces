"""
Job models — spec §35.

A job is one execution of a task. It has a state machine (§35.1),
metadata (§35.2), and controls (§35.3).

The state machine is enforced: `transition()` refuses invalid moves and
raises with a clear reason.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# States (§35.1)
# ---------------------------------------------------------------------------

class JobState(str, Enum):
    DRAFT = "DRAFT"
    QUEUED = "QUEUED"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    RECOVERING = "RECOVERING"
    VALIDATING = "VALIDATING"
    QUALITY_FAILED = "QUALITY_FAILED"
    SUCCEEDED = "SUCCEEDED"
    SUCCEEDED_WITH_WARNINGS = "SUCCEEDED_WITH_WARNINGS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    QUARANTINED = "QUARANTINED"
    PAUSED_BY_CIRCUIT_BREAKER = "PAUSED_BY_CIRCUIT_BREAKER"


TERMINAL_STATES = {
    JobState.SUCCEEDED,
    JobState.SUCCEEDED_WITH_WARNINGS,
    JobState.FAILED,
    JobState.CANCELLED,
    JobState.QUALITY_FAILED,
    JobState.QUARANTINED,
}


# ---------------------------------------------------------------------------
# Allowed transitions
# ---------------------------------------------------------------------------

_ALLOWED: dict[JobState, set[JobState]] = {
    JobState.DRAFT:                      {JobState.QUEUED, JobState.CANCELLED},
    JobState.QUEUED:                     {JobState.PLANNING, JobState.CANCELLED},
    JobState.PLANNING:                   {JobState.RUNNING, JobState.FAILED,
                                          JobState.CANCELLED},
    JobState.RUNNING:                    {JobState.RECOVERING, JobState.VALIDATING,
                                          JobState.FAILED,
                                          JobState.PAUSED_BY_CIRCUIT_BREAKER,
                                          JobState.CANCELLED,
                                          JobState.SUCCEEDED_WITH_WARNINGS},
    JobState.RECOVERING:                 {JobState.RUNNING, JobState.FAILED,
                                          JobState.CANCELLED},
    JobState.VALIDATING:                 {JobState.SUCCEEDED,
                                          JobState.SUCCEEDED_WITH_WARNINGS,
                                          JobState.QUALITY_FAILED,
                                          JobState.FAILED},
    JobState.QUALITY_FAILED:             {JobState.QUARANTINED, JobState.CANCELLED,
                                          JobState.RUNNING},   # re-run allowed
    JobState.QUARANTINED:                {JobState.RUNNING, JobState.CANCELLED},
    JobState.PAUSED_BY_CIRCUIT_BREAKER:  {JobState.RUNNING, JobState.CANCELLED,
                                          JobState.FAILED},
    JobState.SUCCEEDED:                  set(),
    JobState.SUCCEEDED_WITH_WARNINGS:    set(),
    JobState.FAILED:                     {JobState.QUEUED},     # retry path
    JobState.CANCELLED:                  set(),
}


def can_transition(from_state: JobState, to_state: JobState) -> bool:
    return to_state in _ALLOWED.get(from_state, set())


# ---------------------------------------------------------------------------
# Job record (§35.2)
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    task_version: int = 1
    client_id: str = "default"
    state: JobState = JobState.DRAFT
    created_at: str = field(default_factory=_utc_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    driver_used: str = ""
    proxy_profile: str = ""

    pages_scheduled: int = 0
    pages_executed: int = 0
    pages_failed: int = 0
    pages_empty: int = 0
    pages_no_value: int = 0

    records_extracted: int = 0
    quality_score: float = 0.0
    confidence_summary: str = ""

    llm_tokens_used: int = 0
    llm_cost_usd: float = 0.0
    scraperapi_credits_used: int = 0
    scraperapi_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    strategy_ids: list[str] = field(default_factory=list)
    healing_events: list[str] = field(default_factory=list)
    circuit_breaker_tripped: bool = False
    error: str = ""

    # ------------------------------------------------------------------
    def transition(self, to_state: JobState, reason: str = "") -> None:
        if to_state == self.state:
            return
        if not can_transition(self.state, to_state):
            raise ValueError(
                f"invalid job transition {self.state.value} -> "
                f"{to_state.value}"
                + (f" ({reason})" if reason else "")
            )
        self.state = to_state
        now = _utc_iso()
        if to_state == JobState.RUNNING and not self.started_at:
            self.started_at = now
        if to_state in TERMINAL_STATES:
            self.completed_at = now

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def duration_seconds(self) -> Optional[float]:
        if not self.started_at:
            return None
        end = self.completed_at or _utc_iso()
        start_dt = _parse_iso(self.started_at)
        end_dt = _parse_iso(end)
        if not start_dt or not end_dt:
            return None
        return round((end_dt - start_dt).total_seconds(), 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        d["is_terminal"] = self.is_terminal
        d["duration_seconds"] = self.duration_seconds
        return d


def _parse_iso(s: str):
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    j = Job(task_id="t-1", client_id="acme")
    assert j.state == JobState.DRAFT
    assert not j.is_terminal

    j.transition(JobState.QUEUED)
    j.transition(JobState.PLANNING)
    j.transition(JobState.RUNNING)
    assert j.started_at

    j.transition(JobState.VALIDATING)
    j.transition(JobState.SUCCEEDED)
    assert j.is_terminal
    assert j.completed_at

    # Invalid transition
    try:
        j.transition(JobState.RUNNING)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "invalid job transition" in str(e).lower()

    # Successful workflow with warnings
    j2 = Job()
    j2.transition(JobState.QUEUED)
    j2.transition(JobState.PLANNING)
    j2.transition(JobState.RUNNING)
    j2.transition(JobState.SUCCEEDED_WITH_WARNINGS)
    assert j2.is_terminal

    # Circuit breaker path
    j3 = Job()
    j3.transition(JobState.QUEUED)
    j3.transition(JobState.PLANNING)
    j3.transition(JobState.RUNNING)
    j3.transition(JobState.PAUSED_BY_CIRCUIT_BREAKER, reason="budget exceeded")
    assert j3.state == JobState.PAUSED_BY_CIRCUIT_BREAKER
    # Resume
    j3.transition(JobState.RUNNING)
    j3.transition(JobState.VALIDATING)
    j3.transition(JobState.SUCCEEDED)

    # Retry after FAILED
    j4 = Job()
    j4.transition(JobState.QUEUED)
    j4.transition(JobState.PLANNING)
    j4.transition(JobState.FAILED)
    assert j4.is_terminal
    j4.transition(JobState.QUEUED)   # retry
    assert not j4.is_terminal

    # can_transition
    assert can_transition(JobState.DRAFT, JobState.QUEUED)
    assert not can_transition(JobState.SUCCEEDED, JobState.RUNNING)

    # to_dict
    d = j.to_dict()
    assert d["state"] == "SUCCEEDED"
    assert d["is_terminal"] is True

    print("Job model OK.")