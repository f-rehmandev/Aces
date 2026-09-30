"""Tests that pipeline warnings appear as trace events."""
from src.pipeline_runner import PipelineResult
from src.ui.runner import _trace_from_pipeline


def _make_result(warnings):
    return PipelineResult(
        task_id="t", records=[], quality_passed=False, quality_score=0.0,
        publication_decision=None,
        change_set_summary={"new": 0, "modified": 0, "removed": 0,
                             "unchanged": 0, "total": 0},
        workbook=None, receipt_signature=None, security_traces=[],
        warnings=warnings,
    )


def test_warnings_become_trace_events():
    result = _make_result(["fetch failed on https://x: boom"])
    events = _trace_from_pipeline(result)
    warn_events = [e for e in events if e.kind == "WARN"]
    assert len(warn_events) == 1
    assert "fetch failed" in warn_events[0].message
    assert warn_events[0].level == "warning"


def test_no_warnings_no_warn_events():
    result = _make_result([])
    events = _trace_from_pipeline(result)
    assert not [e for e in events if e.kind == "WARN"]