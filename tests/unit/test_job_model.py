"""Unit tests for job model + state machine (spec §35)."""
import pytest

from src.jobs.models import (
    Job, JobState, can_transition, TERMINAL_STATES, _ALLOWED,
)


# --- construction -----------------------------------------------------

def test_new_job_defaults():
    j = Job(task_id="t-1", client_id="acme")
    assert j.state == JobState.DRAFT
    assert j.task_id == "t-1"
    assert j.client_id == "acme"
    assert not j.is_terminal
    assert j.started_at is None


# --- happy path -------------------------------------------------------

def test_happy_path_running_to_success():
    j = Job()
    j.transition(JobState.QUEUED)
    j.transition(JobState.PLANNING)
    j.transition(JobState.RUNNING)
    j.transition(JobState.VALIDATING)
    j.transition(JobState.SUCCEEDED)
    assert j.state == JobState.SUCCEEDED
    assert j.started_at and j.completed_at
    assert j.is_terminal


def test_started_at_set_on_first_running():
    j = Job()
    j.transition(JobState.QUEUED)
    j.transition(JobState.PLANNING)
    assert j.started_at is None
    j.transition(JobState.RUNNING)
    assert j.started_at is not None


# --- terminal states --------------------------------------------------

def test_all_terminal_states_recognized():
    for s in TERMINAL_STATES:
        j = Job(state=s)
        assert j.is_terminal


def test_terminal_cannot_transition_outside_retry():
    j = Job()
    for s in (JobState.QUEUED, JobState.PLANNING, JobState.RUNNING,
              JobState.VALIDATING, JobState.SUCCEEDED):
        j.transition(s)
    with pytest.raises(ValueError):
        j.transition(JobState.RUNNING)


# --- invalid transitions ----------------------------------------------

def test_invalid_transition_raises():
    j = Job()
    with pytest.raises(ValueError):
        j.transition(JobState.RUNNING)   # must be QUEUED first


def test_cannot_cancel_after_success():
    j = Job()
    j.transition(JobState.QUEUED)
    j.transition(JobState.PLANNING)
    j.transition(JobState.RUNNING)
    j.transition(JobState.VALIDATING)
    j.transition(JobState.SUCCEEDED)
    with pytest.raises(ValueError):
        j.transition(JobState.CANCELLED)


# --- failure + retry --------------------------------------------------

def test_failed_can_be_retried():
    j = Job()
    j.transition(JobState.QUEUED)
    j.transition(JobState.PLANNING)
    j.transition(JobState.FAILED)
    assert j.is_terminal
    j.transition(JobState.QUEUED)
    assert not j.is_terminal


# --- circuit breaker --------------------------------------------------

def test_circuit_breaker_pauses():
    j = Job()
    j.transition(JobState.QUEUED)
    j.transition(JobState.PLANNING)
    j.transition(JobState.RUNNING)
    j.transition(JobState.PAUSED_BY_CIRCUIT_BREAKER, reason="max_usd")
    assert j.state == JobState.PAUSED_BY_CIRCUIT_BREAKER
    # Not terminal — user can resume
    assert not j.is_terminal


def test_circuit_breaker_can_resume():
    j = Job()
    j.transition(JobState.QUEUED)
    j.transition(JobState.PLANNING)
    j.transition(JobState.RUNNING)
    j.transition(JobState.PAUSED_BY_CIRCUIT_BREAKER)
    j.transition(JobState.RUNNING)
    assert j.state == JobState.RUNNING


# --- quality failure path --------------------------------------------

def test_quality_failed_path():
    j = Job()
    j.transition(JobState.QUEUED)
    j.transition(JobState.PLANNING)
    j.transition(JobState.RUNNING)
    j.transition(JobState.VALIDATING)
    j.transition(JobState.QUALITY_FAILED)
    assert j.is_terminal
    # Can re-run
    j.transition(JobState.RUNNING)


# --- can_transition helper -------------------------------------------

def test_can_transition_true():
    assert can_transition(JobState.DRAFT, JobState.QUEUED)


def test_can_transition_false():
    assert not can_transition(JobState.DRAFT, JobState.RUNNING)


# --- serialization ----------------------------------------------------

def test_to_dict():
    j = Job(task_id="t", client_id="c")
    d = j.to_dict()
    assert d["state"] == "DRAFT"
    assert d["is_terminal"] is False
    assert d["task_id"] == "t"