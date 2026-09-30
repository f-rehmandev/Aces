"""
Unit tests for per-run token metering (spec §41.4).

Covers:
    - DataExtractor accumulates tokens across every router.call()
    - reset_usage() zeros the counters
    - PipelineRunner writes a TOKEN usage event alongside the PAGE event
    - Token split (input / output) is preserved in metadata
    - Zero tokens → no TOKEN event written
    - Fake extractors without token tracking don't break the pipeline
    - Extractor is reset at the start of each pipeline run
"""
import asyncio

import pytest

from src.core.task_spec import FieldSpec, Target, TaskSpec
from src.extractor.schema_extractor import DataExtractor
from src.pipeline_runner import PipelineRunner
from src.usage.store import InMemoryUsageStore
from src.usage.types import ResourceType


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Router double that reports configurable token counts
# ---------------------------------------------------------------------------

class _FakeRouter:
    def __init__(self, text='[{"title":"A"}]', in_t=100, out_t=30):
        self.text = text
        self.in_t = in_t
        self.out_t = out_t
        self.calls = 0

    def call(self, prompt, timeout=None):
        self.calls += 1
        return {
            "text": self.text,
            "provider": "fake",
            "input_tokens": self.in_t,
            "output_tokens": self.out_t,
            "total_tokens": self.in_t + self.out_t,
        }


# ===========================================================================
# DataExtractor accumulation
# ===========================================================================

def test_extractor_accumulates_tokens_across_calls():
    router = _FakeRouter(in_t=100, out_t=30)
    ex = DataExtractor(router=router)

    ex.extract_list("<html>x</html>", "instruction")
    ex.extract_list("<html>y</html>", "instruction")

    assert ex.call_count == 2
    assert ex.total_input_tokens == 200
    assert ex.total_output_tokens == 60


def test_extractor_reset_usage_zeroes_counters():
    ex = DataExtractor(router=_FakeRouter(in_t=10, out_t=5))
    ex.extract_list("<html>x</html>", "instruction")
    assert ex.total_input_tokens == 10

    ex.reset_usage()
    assert ex.total_input_tokens == 0
    assert ex.total_output_tokens == 0
    assert ex.call_count == 0


def test_extractor_tolerates_missing_token_fields():
    """A router that returns only text (older shape) must not break."""
    class _TextOnlyRouter:
        def call(self, prompt, timeout=None):
            return {"text": '[{"title":"A"}]', "provider": "old"}

    ex = DataExtractor(router=_TextOnlyRouter())
    ex.extract_list("<html>x</html>", "instruction")
    # Treated as zero tokens, not an error
    assert ex.total_input_tokens == 0
    assert ex.total_output_tokens == 0
    assert ex.call_count == 1


# ===========================================================================
# Pipeline writes a TOKEN event
# ===========================================================================

class _FakeScraper:
    async def fetch_html(self, url, timeout=None):
        return "<html><body>" + ("x" * 1000) + "</body></html>"

    async def fetch_screenshot(self, url, timeout=None):
        return b""


def _task(task_id="t-meter"):
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.task_id = task_id
    return spec


def test_pipeline_writes_page_and_token_events():
    store = InMemoryUsageStore()
    extractor = DataExtractor(router=_FakeRouter(in_t=500, out_t=120))

    runner = PipelineRunner(
        _FakeScraper(), extractor,
        client_id="acme",
        job_id="job-token-1",
        usage_store=store,
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-token-1"))
    by_type = {e.resource_type: e for e in events}

    assert ResourceType.PAGE in by_type
    assert ResourceType.TOKEN in by_type

    page = by_type[ResourceType.PAGE]
    assert page.quantity == 1.0
    assert page.client_id == "acme"

    token = by_type[ResourceType.TOKEN]
    assert token.quantity == 620.0
    assert token.metadata["input_tokens"] == 500
    assert token.metadata["output_tokens"] == 120
    assert token.metadata["llm_calls"] == 1


def test_pipeline_writes_no_token_event_when_zero():
    """A run with a text-only router writes PAGE but not TOKEN."""
    class _TextOnlyRouter:
        def call(self, prompt, timeout=None):
            return {"text": '[{"title":"A"}]', "provider": "old"}

    store = InMemoryUsageStore()
    extractor = DataExtractor(router=_TextOnlyRouter())

    runner = PipelineRunner(
        _FakeScraper(), extractor,
        client_id="acme", job_id="job-no-tokens",
        usage_store=store,
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-no-tokens"))
    types = {e.resource_type for e in events}
    assert ResourceType.PAGE in types
    assert ResourceType.TOKEN not in types


def test_pipeline_resets_extractor_usage_between_runs():
    """
    Reusing the same extractor instance across two runs must not double-
    count tokens. Each run's TOKEN event reflects only its own calls.
    """
    store = InMemoryUsageStore()
    extractor = DataExtractor(router=_FakeRouter(in_t=100, out_t=25))

    runner = PipelineRunner(
        _FakeScraper(), extractor,
        client_id="acme", job_id="job-r1",
        usage_store=store,
    )
    _run(runner.run(_task()))
    first = _run(store.events_for_job("job-r1"))

    # Same extractor instance, same runner, second run under a new job_id
    runner.job_id = "job-r2"
    _run(runner.run(_task()))
    second = _run(store.events_for_job("job-r2"))

    def token_qty(events):
        for e in events:
            if e.resource_type == ResourceType.TOKEN:
                return e.quantity
        return 0.0

    assert token_qty(first) == 125.0
    assert token_qty(second) == 125.0  # not 250


def test_pipeline_tolerates_fake_extractor_without_usage_attrs():
    """The legacy FakeExtractor pattern (no total_*_tokens) must still work."""
    class _BareExtractor:
        def extract_list(self, html, instruction):
            return [{"title": "A"}]

        def extract_from_image(self, img, instr):
            return []

    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _FakeScraper(), _BareExtractor(),
        client_id="acme", job_id="job-bare",
        usage_store=store,
    )
    result = _run(runner.run(_task()))

    # No crash, records produced, PAGE event written, no TOKEN event.
    assert len(result.records) == 1
    events = _run(store.events_for_job("job-bare"))
    types = {e.resource_type for e in events}
    assert ResourceType.PAGE in types
    assert ResourceType.TOKEN not in types
    