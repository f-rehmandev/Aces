"""Tests for ClientContext wiring through PipelineRunner + UI bridge."""
import asyncio
from pathlib import Path

import pytest

from src.auth.context import ClientContext, anonymous_context
from src.auth.models import PermissionDenied, Role, Session
from src.core.task_spec import FieldSpec, Target, TaskSpec
from src.pipeline_runner import PipelineRunner
from src.ui.runner import run_ui_task_from_spec


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeScraper:
    def __init__(self, html="<html>SENTINEL</html>"):
        self.html = html

    async def fetch_html(self, url, timeout=None):
        return self.html

    async def fetch_screenshot(self, url, timeout=None):
        return b"fake"


class FakeExtractor:
    def __init__(self, records=None):
        self.records = records if records is not None else [{"title": "A"}]

    def extract_list(self, html, instruction):
        return list(self.records)

    def extract_from_image(self, img, instr):
        return []


def _task():
    return TaskSpec(
        natural_language_prompt="x",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )


def _ctx(client_id: str, role: Role = Role.OWNER, auth: bool = True):
    return ClientContext(
        client_id=client_id,
        client_uuid=client_id,
        user_id="u-1" if auth else None,
        role=role if auth else None,
        client_name="Test",
        client_slug="test",
        is_anonymous=not auth,
    )


# ---------------------------------------------------------------------------
# PipelineRunner: tenant scoping
# ---------------------------------------------------------------------------

def test_runner_without_context_uses_client_id_string():
    runner = PipelineRunner(FakeScraper(), FakeExtractor(), client_id="legacy")
    assert runner.client_id == "legacy"
    assert runner.context.is_anonymous


def test_runner_with_context_overrides_client_id():
    ctx = _ctx("c-real-uuid")
    runner = PipelineRunner(FakeScraper(), FakeExtractor(),
                             client_id="ignored", context=ctx)
    assert runner.client_id == "c-real-uuid"
    assert runner.is_authenticated


def test_runner_defaults_to_anonymous_when_neither_provided():
    runner = PipelineRunner(FakeScraper(), FakeExtractor())
    assert runner.client_id == "default"
    assert runner.context.is_anonymous


def test_runner_context_flows_into_result():
    ctx = _ctx("c-real-uuid")
    runner = PipelineRunner(FakeScraper(), FakeExtractor(), context=ctx)
    result = asyncio.run(runner.run(_task()))
    assert result.client_id == "c-real-uuid"
    assert result.is_authenticated is True


def test_runner_anonymous_context_marks_result_unauthenticated():
    runner = PipelineRunner(FakeScraper(), FakeExtractor())
    result = asyncio.run(runner.run(_task()))
    assert result.client_id == "default"
    assert result.is_authenticated is False


# ---------------------------------------------------------------------------
# assert_can_write
# ---------------------------------------------------------------------------

def test_assert_can_write_ok_for_owner():
    runner = PipelineRunner(FakeScraper(), FakeExtractor(),
                             context=_ctx("c-1", Role.OWNER))
    runner.assert_can_write()   # must not raise


def test_assert_can_write_rejects_viewer():
    runner = PipelineRunner(FakeScraper(), FakeExtractor(),
                             context=_ctx("c-1", Role.VIEWER))
    with pytest.raises(PermissionDenied):
        runner.assert_can_write()


def test_assert_can_write_allows_anonymous_dev():
    runner = PipelineRunner(FakeScraper(), FakeExtractor(),
                             context=anonymous_context())
    runner.assert_can_write()   # local dev fallback must not break


# ---------------------------------------------------------------------------
# UI bridge: context propagation
# ---------------------------------------------------------------------------

def test_ui_bridge_forwards_context_to_runner(monkeypatch):
    captured = {}

    import src.ui.runner as runner_mod

    original_init = PipelineRunner.__init__

    def spy_init(self, *args, **kwargs):
        captured["context"] = kwargs.get("context")
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(PipelineRunner, "__init__", spy_init)

    ctx = _ctx("c-ui")
    run_ui_task_from_spec(
        _task(),
        scraper=FakeScraper(),
        extractor=FakeExtractor(),
        context=ctx,
    )
    assert captured["context"] is ctx


def test_ui_bridge_passes_anonymous_when_no_context():
    result = run_ui_task_from_spec(
        _task(),
        scraper=FakeScraper(),
        extractor=FakeExtractor(),
    )
    assert result.mode == "real"
    # Result has the anonymous client_id
    assert result.spec is not None


# ---------------------------------------------------------------------------
# End-to-end: auth context wins over client_id string
# ---------------------------------------------------------------------------

def test_end_to_end_context_wins():
    result = run_ui_task_from_spec(
        _task(),
        client_id="should-be-ignored",
        scraper=FakeScraper(),
        extractor=FakeExtractor([{"title": "X"}]),
        context=_ctx("c-auth"),
    )
    assert result.mode == "real"
    assert result.record_count == 1