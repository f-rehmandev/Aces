"""Unit tests for the UI ↔ pipeline bridge (src/ui/runner.py)."""
import asyncio
from unittest.mock import patch

import pytest

from src.ui.runner import (
    run_ui_task, UiRunResult, TraceEvent,
    _build_spec, _trace_from_pipeline, _real_result, _demo_result,
)
from src.core.task_spec import TaskSpec, Target, FieldSpec
from src.pipeline_runner import PipelineResult
from src.quality.publication import PublicationDecision, DatasetState
from src.security.trace import SecurityTrace


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeScraper:
    def __init__(self, html_by_url=None):
        self.html_by_url = html_by_url or {}

    async def fetch_html(self, url, timeout=None):
        return self.html_by_url.get(url, "<html>SENTINEL</html>")

    async def fetch_screenshot(self, url, timeout=None):
        return b"fake-png"


class FakeExtractor:
    def __init__(self, records=None):
        self.records = records if records is not None else [
            {"title": "A", "price": "$10"},
            {"title": "B", "price": "$20"},
        ]

    def extract_list(self, html, instruction):
        return list(self.records)

    def extract_from_image(self, img, instr):
        return []


class BrokenScraper:
    async def fetch_html(self, url, timeout=None):
        raise RuntimeError("network down")

    async def fetch_screenshot(self, url, timeout=None):
        raise RuntimeError("no screenshot either")


# ---------------------------------------------------------------------------
# _build_spec
# ---------------------------------------------------------------------------

def test_build_spec_applies_ui_overrides(monkeypatch):
    # Bypass the LLM by patching resolve_input with a stub.
    from src.intake import dispatcher

    def fake_resolve(raw, kind=None, client_id="default"):
        from src.intake.resolver import InputResolution
        spec = TaskSpec(
            natural_language_prompt=raw,
            target=Target(start_urls=["https://auto.example/x"]),
            fields=[FieldSpec(name="title")],
        )
        return InputResolution(spec=spec, source_description="fake")

    monkeypatch.setattr(dispatcher, "resolve_input", fake_resolve)

    spec = _build_spec("find stuff", "https://manual.example/y",
                       "acme", "CSV", 3)
    # URL override wins
    assert spec.target.start_urls == ["https://manual.example/y"]
    assert spec.output.format == "csv"
    assert spec.source_requirements.min_independent_sources == 3


def test_build_spec_no_url_keeps_resolver_urls(monkeypatch):
    from src.intake import dispatcher

    def fake_resolve(raw, kind=None, client_id="default"):
        from src.intake.resolver import InputResolution
        spec = TaskSpec(
            natural_language_prompt=raw,
            target=Target(start_urls=["https://auto.example/x"]),
        )
        return InputResolution(spec=spec, source_description="fake")

    monkeypatch.setattr(dispatcher, "resolve_input", fake_resolve)
    spec = _build_spec("x", "", "acme", "XLSX", 1)
    assert spec.target.start_urls == ["https://auto.example/x"]


# ---------------------------------------------------------------------------
# Trace translation
# ---------------------------------------------------------------------------

def _fake_pipeline_result(**overrides):
    base = dict(
        task_id="t-1",
        records=[{"title": "A", "price": "$10"}],
        quality_passed=True,
        quality_score=0.95,
        publication_decision=PublicationDecision(
            state=DatasetState.READY_TO_PUBLISH, allowed=True,
        ),
        change_set_summary={"new": 1, "modified": 0, "removed": 0,
                             "unchanged": 0, "total": 1},
        workbook=None,
        receipt_signature="deadbeef",
        security_traces=[SecurityTrace(url="https://93.184.216.34/a")],
        confidence_mean=0.9,
    )
    base.update(overrides)
    return PipelineResult(**base)


def test_trace_contains_expected_kinds():
    result = _fake_pipeline_result()
    events = _trace_from_pipeline(result)
    kinds = [e.kind for e in events]
    for expected in ("FETCH", "EXTRACT", "QUALITY", "DIFF", "CONFIDENCE",
                     "RECEIPT", "DONE"):
        assert expected in kinds


def test_trace_marks_ssrf_block():
    t = SecurityTrace(url="http://10.0.0.5/secret")
    t.ssrf_allowed = False
    t.ssrf_reason = "private"
    result = _fake_pipeline_result(security_traces=[t])
    events = _trace_from_pipeline(result)
    assert any(e.kind == "SSRF" and "private" in e.message for e in events)


def test_trace_records_sanitizer_removals():
    t = SecurityTrace(url="https://x.com")
    t.sanitizer_removed = 3
    result = _fake_pipeline_result(security_traces=[t])
    events = _trace_from_pipeline(result)
    assert any(e.kind == "SANITIZE" for e in events)


def test_trace_marks_quality_failure():
    result = _fake_pipeline_result(quality_passed=False, quality_score=0.3)
    events = _trace_from_pipeline(result)
    quality_evt = [e for e in events if e.kind == "QUALITY"][0]
    assert quality_evt.level == "warning"


# ---------------------------------------------------------------------------
# Result translation
# ---------------------------------------------------------------------------

def test_real_result_shape():
    result = _fake_pipeline_result()
    ui = _real_result(result, [_evt_for_test()])
    assert ui.mode == "real"
    assert ui.record_count == 1
    assert ui.quality_score == 0.95
    assert ui.confidence_mean == 0.9
    assert ui.new_count == 1
    assert ui.source_count == 1
    assert ui.receipt_signature == "deadbeef"


def test_real_result_workbook_path():
    from src.output.workbook import WorkbookResult
    from pathlib import Path
    wb = WorkbookResult(path=Path("/tmp/r.xlsx"), sheet_names=["Data"])
    result = _fake_pipeline_result(workbook=wb)
    ui = _real_result(result, [])
    assert ui.workbook_path and ui.workbook_path.endswith("r.xlsx")


def test_demo_result_shape():
    ui = _demo_result("p", "u", "c", [], reason="x")
    assert ui.mode == "demo"
    assert len(ui.records) == 6
    assert ui.quality_passed is True
    assert any(e.kind == "DEMO" for e in ui.trace)


def _evt_for_test():
    return TraceEvent(time="00:00:00", kind="PARSE", message="x")


# ---------------------------------------------------------------------------
# run_ui_task — end to end
# ---------------------------------------------------------------------------

def test_run_ui_task_real_path(monkeypatch):
    # Stub dispatcher to skip the LLM
    from src.intake import dispatcher

    def fake_resolve(raw, kind=None, client_id="default"):
        from src.intake.resolver import InputResolution
        spec = TaskSpec(
            natural_language_prompt=raw,
            target=Target(start_urls=["https://93.184.216.34/a"]),
            fields=[FieldSpec(name="title")],
        )
        return InputResolution(spec=spec, source_description="fake")

    monkeypatch.setattr(dispatcher, "resolve_input", fake_resolve)

    result = run_ui_task(
        "find things",
        scraper=FakeScraper({"https://93.184.216.34/a": "<html>SENTINEL</html>"}),
        extractor=FakeExtractor([{"title": "X", "price": "$1"}]),
        client_id="acme",
    )
    assert result.mode == "real"
    assert result.record_count == 1
    assert result.records[0]["title"] == "X"
    assert any(e.kind == "DONE" for e in result.trace)


def test_run_ui_task_falls_back_to_demo_on_pipeline_error(monkeypatch):
    from src.intake import dispatcher

    def fake_resolve(raw, kind=None, client_id="default"):
        from src.intake.resolver import InputResolution
        spec = TaskSpec(
            natural_language_prompt=raw,
            target=Target(start_urls=["https://93.184.216.34/a"]),
            fields=[FieldSpec(name="title")],
        )
        return InputResolution(spec=spec, source_description="fake")

    monkeypatch.setattr(dispatcher, "resolve_input", fake_resolve)

    # BrokenScraper raises -> pipeline catches, adds a warning, quality
    # may still fail, and eventually run_ui_task should still return a
    # renderable result (either real with 0 records or demo).
    result = run_ui_task(
        "find things",
        scraper=BrokenScraper(),
        extractor=FakeExtractor(),
    )
    assert result.mode in ("real", "demo")
    # Either way, the trace must be present and end with something usable.
    assert result.trace
    # And the UI-facing shape must be valid.
    assert isinstance(result.records, list)


def test_run_ui_task_falls_back_on_spec_build_error(monkeypatch):
    from src.intake import dispatcher

    def boom(*args, **kwargs):
        raise RuntimeError("LLM offline")

    monkeypatch.setattr(dispatcher, "resolve_input", boom)

    result = run_ui_task("find things")
    assert result.mode == "demo"
    assert any(e.kind == "DEMO" for e in result.trace)


def test_run_ui_task_compliance_refusal_triggers_demo(monkeypatch):
    from src.intake import dispatcher

    def fake_resolve(raw, kind=None, client_id="default"):
        from src.intake.resolver import InputResolution
        spec = TaskSpec(natural_language_prompt=raw)
        spec.compliance.refusal_reason = "bypassing access control"
        return InputResolution(spec=spec, source_description="fake")

    monkeypatch.setattr(dispatcher, "resolve_input", fake_resolve)
    result = run_ui_task("bypass the login wall")
    assert result.mode == "demo"
    assert any(e.kind == "COMPLY" for e in result.trace)


# ---------------------------------------------------------------------------
# TraceEvent serialization
# ---------------------------------------------------------------------------

def test_trace_event_to_dict():
    e = TraceEvent(time="01:02:03", kind="FETCH", message="ok", level="info")
    d = e.to_dict()
    assert d == {"time": "01:02:03", "kind": "FETCH",
                 "message": "ok", "level": "info"}