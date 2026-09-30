"""Regression: TaskSpec.quality.min_records reaches the PipelineRunner."""
import asyncio

from src.core.task_spec import FieldSpec, Quality, Target, TaskSpec
from src.pipeline_runner import PipelineRunner
from src.ui import runner as ui_runner


class FakeScraper:
    async def fetch_html(self, url, timeout=None):
        # Realistic page: title + heading + query-term match
        return (
            "<html><head><title>Test</title></head>"
            "<body><h1>Test</h1><p>content panadol</p>"
            + "x" * 2000
            + "</body></html>"
        )
    async def fetch_screenshot(self, url, timeout=None):
        return b""


class FakeExtractor:
    def extract_list(self, html, instruction):
        # Always return exactly 3 records — deliberately below any
        # threshold we'll test with.
        return [{"title": f"R{i}"} for i in range(3)]
    def extract_from_image(self, img, instr):
        return []


def _task(min_records: int) -> TaskSpec:
    return TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
        quality=Quality(min_records=min_records),
    )


def test_runner_uses_task_min_records():
    """The runner must enforce the TaskSpec's min_records, not a default."""
    spec = _task(min_records=100)  # unreachable
    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        quality_rules=_rules_from_spec(spec),
    )
    result = asyncio.run(runner.run(spec))
    assert not result.quality_passed
    assert "min_records" in " ".join(result.warnings).lower() or \
           result.quality_score < 1.0


def test_ui_bridge_plumbs_quality_rules(monkeypatch):
    """run_ui_task_from_spec must pass quality_rules matching the TaskSpec."""
    captured = {}
    original_init = PipelineRunner.__init__

    def spy_init(self, *args, **kwargs):
        captured["quality_rules"] = kwargs.get("quality_rules")
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(PipelineRunner, "__init__", spy_init)

    spec = _task(min_records=42)
    ui_runner.run_ui_task_from_spec(
        spec,
        scraper=FakeScraper(),
        extractor=FakeExtractor(),
    )
    qr = captured.get("quality_rules")
    assert qr is not None
    assert qr.min_records == 42


def _rules_from_spec(spec):
    from src.quality.rules import QualityRules
    return QualityRules(
        min_records=spec.quality.min_records,
        min_populated_field_pct=spec.quality.min_populated_field_pct,
        max_failed_page_pct=spec.quality.max_failed_page_pct,
        max_empty_page_pct=spec.quality.max_empty_page_pct,
    )