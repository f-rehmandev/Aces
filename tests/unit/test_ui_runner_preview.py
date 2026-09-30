"""Tests for the Plan Review entry points (src/ui/runner.py)."""
import pytest

from src.ui.runner import (
    build_spec_for_preview, run_ui_task_from_spec, UiRunResult,
)
from src.core.task_spec import TaskSpec, Target, FieldSpec


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeScraper:
    def __init__(self, html="<html>SENTINEL</html>"):
        self.html = html

    async def fetch_html(self, url, timeout=None):
        return self.html

    async def fetch_screenshot(self, url, timeout=None):
        return b"fake-png"


class FakeExtractor:
    def __init__(self, records=None):
        self.records = records if records is not None else [
            {"title": "A", "price": "$10"},
        ]

    def extract_list(self, html, instruction):
        return list(self.records)

    def extract_from_image(self, img, instr):
        return []


# ---------------------------------------------------------------------------
# build_spec_for_preview
# ---------------------------------------------------------------------------

def test_build_spec_for_preview_success(monkeypatch):
    from src.intake import dispatcher

    def fake_resolve(raw, kind=None, client_id="default"):
        from src.intake.resolver import InputResolution
        spec = TaskSpec(
            natural_language_prompt=raw,
            target=Target(start_urls=["https://example.com/x"]),
            fields=[FieldSpec(name="title")],
        )
        return InputResolution(spec=spec, source_description="fake")

    monkeypatch.setattr(dispatcher, "resolve_input", fake_resolve)

    spec, err = build_spec_for_preview("find things", "https://a.example/b",
                                       "acme", "CSV", 2)
    assert err == ""
    assert spec is not None
    assert spec.target.start_urls == ["https://a.example/b"]
    assert spec.output.format == "csv"
    assert spec.source_requirements.min_independent_sources == 2


def test_build_spec_for_preview_returns_error_on_exception(monkeypatch):
    from src.intake import dispatcher

    def boom(*a, **k):
        raise RuntimeError("LLM offline")

    monkeypatch.setattr(dispatcher, "resolve_input", boom)

    spec, err = build_spec_for_preview("find things")
    assert spec is None
    assert "RuntimeError" in err
    assert "LLM offline" in err


# ---------------------------------------------------------------------------
# run_ui_task_from_spec — happy path
# ---------------------------------------------------------------------------

def test_run_from_spec_real_path():
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    result = run_ui_task_from_spec(
        spec,
        scraper=FakeScraper(),
        extractor=FakeExtractor([{"title": "A", "price": "$1"}]),
        client_id="acme",
    )
    assert isinstance(result, UiRunResult)
    assert result.mode == "real"
    assert result.record_count == 1
    assert result.records[0]["title"] == "A"


def test_run_from_spec_attaches_spec_to_result():
    spec = TaskSpec(
        natural_language_prompt="x",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="t")],
    )
    result = run_ui_task_from_spec(
        spec, scraper=FakeScraper(), extractor=FakeExtractor(),
    )
    assert result.spec is spec


# ---------------------------------------------------------------------------
# run_ui_task_from_spec — compliance refusal
# ---------------------------------------------------------------------------

def test_run_from_spec_compliance_refusal_returns_demo():
    spec = TaskSpec(
        natural_language_prompt="bypass login",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="t")],
    )
    spec.compliance.refusal_reason = "bypassing access control"
    result = run_ui_task_from_spec(spec, scraper=FakeScraper(),
                                   extractor=FakeExtractor())
    assert result.mode == "demo"
    assert any(e.kind == "COMPLY" for e in result.trace)


# ---------------------------------------------------------------------------
# run_ui_task_from_spec — pipeline exception -> demo
# ---------------------------------------------------------------------------

class BrokenScraper:
    async def fetch_html(self, url, timeout=None):
        raise RuntimeError("boom")
    async def fetch_screenshot(self, url, timeout=None):
        raise RuntimeError("boom")


def test_run_from_spec_falls_back_to_demo_on_pipeline_error(monkeypatch):
    # Force PipelineRunner to raise before it even tries to fetch
    from src.ui import runner as runner_mod

    def boom_init(self, *a, **k):
        raise RuntimeError("simulated init failure")

    monkeypatch.setattr(runner_mod.PipelineRunner, "__init__", boom_init)

    spec = TaskSpec(
        natural_language_prompt="x",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="t")],
    )
    result = run_ui_task_from_spec(spec, scraper=BrokenScraper(),
                                   extractor=FakeExtractor())
    assert result.mode == "demo"
    assert any(e.kind == "ERROR" for e in result.trace)



# ---------------------------------------------------------------------------
# Source discovery wiring
# ---------------------------------------------------------------------------

def test_run_from_spec_discovers_urls_when_missing(monkeypatch):
    """When the spec has a hint but no URLs, discovery runs and populates URLs."""
    from src.ui import runner as runner_mod
    from src.intake import dispatcher

    async def fake_search(query, max_results=5):
        return ["https://93.184.216.34/a", "https://93.184.216.35/b"]

    # Patch search_products where it's imported from
    import src.discovery.product_search as ps
    monkeypatch.setattr(ps, "search_products", fake_search)

    spec = TaskSpec(
        natural_language_prompt="find wireless mouse prices",
        target=Target(start_urls=[], source_hint="wireless mouse price"),
        fields=[FieldSpec(name="title")],
    )

    result = run_ui_task_from_spec(
        spec,
        scraper=FakeScraper("<html>SENTINEL</html>"),
        extractor=FakeExtractor([{"title": "A", "price": "$1"}]),
    )

    # The spec's URLs were populated by discovery
    assert spec.target.start_urls == [
        "https://93.184.216.34/a", "https://93.184.216.35/b",
    ]
    # And the trace recorded it
    kinds = [e.kind for e in result.trace]
    assert "DISCOVER" in kinds


def test_run_from_spec_warns_when_no_urls_and_no_hint(monkeypatch):
    spec = TaskSpec(
        natural_language_prompt="x",
        target=Target(start_urls=[], source_hint=""),
        fields=[FieldSpec(name="t")],
    )
    result = run_ui_task_from_spec(
        spec, scraper=FakeScraper(), extractor=FakeExtractor(),
    )
    discover_events = [e for e in result.trace if e.kind == "DISCOVER"]
    assert discover_events
    assert "nothing to fetch" in discover_events[0].message.lower()


def test_run_from_spec_uses_urls_when_already_present(monkeypatch):
    """If URLs are already set, discovery must not run."""
    from src.discovery import product_search as ps

    called = {"n": 0}
    async def should_not_call(query, max_results=5):
        called["n"] += 1
        return []

    monkeypatch.setattr(ps, "search_products", should_not_call)

    spec = TaskSpec(
        natural_language_prompt="x",
        target=Target(start_urls=["https://93.184.216.34/a"],
                       source_hint="ignored"),
        fields=[FieldSpec(name="t")],
    )
    run_ui_task_from_spec(
        spec, scraper=FakeScraper(), extractor=FakeExtractor(),
    )
    assert called["n"] == 0


def test_discovery_failure_is_graceful(monkeypatch):
    """If discovery raises, we still run (with zero URLs) and warn."""
    from src.discovery import product_search as ps

    async def boom(query, max_results=5):
        raise RuntimeError("DuckDuckGo unreachable")

    monkeypatch.setattr(ps, "search_products", boom)

    spec = TaskSpec(
        natural_language_prompt="x",
        target=Target(start_urls=[], source_hint="something"),
        fields=[FieldSpec(name="t")],
    )
    result = run_ui_task_from_spec(
        spec, scraper=FakeScraper(), extractor=FakeExtractor(),
    )
    discover_events = [e for e in result.trace if e.kind == "DISCOVER"]
    assert any("failed" in e.message.lower() for e in discover_events)