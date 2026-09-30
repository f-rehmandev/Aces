"""
Unit tests for per-run vision-call metering (spec §41.4).

Covers:
    - DataExtractor initializes `vision_call_count` at zero
    - reset_usage() zeroes it
    - extract_from_image increments it once per successful response
    - extract_from_image increments it even when the JSON parse fails
      (the model was still invoked and billed)
    - PipelineRunner writes a VISION_CALL event when the counter is > 0
    - No VISION_CALL event when the counter is zero
    - The extractor is reset at the start of each run
    - Extractors without the counter attribute don't break the pipeline
"""
import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from src.core.task_spec import FieldSpec, Target, TaskSpec
from src.extractor.schema_extractor import DataExtractor
from src.pipeline_runner import PipelineRunner
from src.usage.store import InMemoryUsageStore
from src.usage.types import ResourceType


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# DataExtractor vision counter
# ===========================================================================

def test_extractor_initializes_vision_counter_at_zero():
    ex = DataExtractor(router=MagicMock())
    assert ex.vision_call_count == 0


def test_extractor_reset_usage_zeroes_vision_counter():
    ex = DataExtractor(router=MagicMock())
    ex.vision_call_count = 7
    ex.reset_usage()
    assert ex.vision_call_count == 0


def test_extract_from_image_increments_counter():
    """A successful vision call bumps the counter exactly once."""
    ex = DataExtractor(router=MagicMock())

    fake_response = MagicMock()
    fake_response.text = json.dumps([{"title": "A"}])

    fake_model = MagicMock()
    fake_model.generate_content.return_value = fake_response

    with patch("src.extractor.schema_extractor.genai") as fake_genai:
        fake_genai.GenerativeModel.return_value = fake_model
        result = ex.extract_from_image(b"png", "instruction")

    assert result == [{"title": "A"}]
    assert ex.vision_call_count == 1


def test_extract_from_image_increments_even_on_parse_error():
    """
    If the model is invoked but returns un-parseable JSON, the call still
    happened — count it so cost accounting is honest.
    """
    ex = DataExtractor(router=MagicMock())

    fake_response = MagicMock()
    fake_response.text = "not json at all"

    fake_model = MagicMock()
    fake_model.generate_content.return_value = fake_response

    with patch("src.extractor.schema_extractor.genai") as fake_genai:
        fake_genai.GenerativeModel.return_value = fake_model
        with pytest.raises(ValueError):
            ex.extract_from_image(b"png", "instruction")

    assert ex.vision_call_count == 1


def test_extract_from_image_accumulates_across_calls():
    ex = DataExtractor(router=MagicMock())

    fake_response = MagicMock()
    fake_response.text = json.dumps([{"title": "A"}])

    fake_model = MagicMock()
    fake_model.generate_content.return_value = fake_response

    with patch("src.extractor.schema_extractor.genai") as fake_genai:
        fake_genai.GenerativeModel.return_value = fake_model
        ex.extract_from_image(b"png", "i1")
        ex.extract_from_image(b"png", "i2")
        ex.extract_from_image(b"png", "i3")

    assert ex.vision_call_count == 3


# ===========================================================================
# Pipeline integration
# ===========================================================================

class _FakeScraper:
    async def fetch_html(self, url, timeout=None):
        return "<html><body>" + ("x" * 1000) + "</body></html>"

    async def fetch_screenshot(self, url, timeout=None):
        return b""


class _StickyExtractor:
    """
    Extractor fake whose reset_usage is a no-op. Its vision_call_count
    stays at whatever the test set — this lets us verify the read path
    without contriving a real vision-path trigger.
    """
    def __init__(self, vision_calls: int = 0):
        self.vision_call_count = vision_calls

    def reset_usage(self):
        pass

    def extract_list(self, html, instruction):
        return [{"title": "A"}]

    def extract_from_image(self, img, instr):
        return []


def _task(task_id="t-vision"):
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
    )
    spec.task_id = task_id
    return spec


def test_pipeline_writes_vision_call_event():
    store = InMemoryUsageStore()
    extractor = _StickyExtractor(vision_calls=3)

    runner = PipelineRunner(
        _FakeScraper(), extractor,
        client_id="acme", job_id="job-vision-1",
        usage_store=store,
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-vision-1"))
    by_type = {e.resource_type: e for e in events}
    assert ResourceType.VISION_CALL in by_type

    vc = by_type[ResourceType.VISION_CALL]
    assert vc.quantity == 3.0
    assert vc.unit == "call"
    assert vc.client_id == "acme"


def test_pipeline_skips_vision_event_when_zero():
    store = InMemoryUsageStore()
    extractor = _StickyExtractor(vision_calls=0)

    runner = PipelineRunner(
        _FakeScraper(), extractor,
        client_id="acme", job_id="job-no-vision",
        usage_store=store,
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-no-vision"))
    types = {e.resource_type for e in events}
    assert ResourceType.VISION_CALL not in types
    assert ResourceType.PAGE in types


def test_pipeline_tolerates_extractor_without_vision_counter():
    """Legacy extractors without `vision_call_count` must still work."""
    class _BareExtractor:
        def extract_list(self, html, instruction):
            return [{"title": "A"}]

        def extract_from_image(self, img, instr):
            return []

    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _FakeScraper(), _BareExtractor(),
        client_id="acme", job_id="job-bare-vision",
        usage_store=store,
    )
    result = _run(runner.run(_task()))

    assert len(result.records) == 1
    events = _run(store.events_for_job("job-bare-vision"))
    types = {e.resource_type for e in events}
    assert ResourceType.VISION_CALL not in types


def test_pipeline_calls_extractor_reset_at_run_start():
    """Extractor reset_usage() must run before the first fetch."""
    reset_count = {"n": 0}

    class _CountingExtractor(_StickyExtractor):
        def reset_usage(self):
            reset_count["n"] += 1

    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _FakeScraper(), _CountingExtractor(vision_calls=0),
        client_id="acme", job_id="job-vision-reset",
        usage_store=store,
    )
    _run(runner.run(_task()))
    assert reset_count["n"] == 1


def test_pipeline_writes_all_five_resource_types_together():
    """Integration check: PAGE, TOKEN, BROWSER_SECOND, PROVIDER_CREDIT, VISION_CALL."""
    class _StickyExtractorWithAll(_StickyExtractor):
        def __init__(self):
            super().__init__(vision_calls=2)
            self.total_input_tokens = 500
            self.total_output_tokens = 120
            self.call_count = 1

    class _ScraperWithBrowser(_FakeScraper):
        def __init__(self):
            self.total_browser_seconds = 3.0
            self.browser_call_count = 1

        def reset_usage(self):
            pass

    class _StickyManager:
        def __init__(self):
            self.total_provider_credits = 5
            self.provider_credits_by_provider = {"scraperapi": 5}
            self.provider_call_count = 1

        def reset_usage(self):
            pass

        async def fetch(self, url, **kwargs):
            from src.network.types import FetchResult
            return FetchResult(url=url, html="", status_code=200)

    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _ScraperWithBrowser(), _StickyExtractorWithAll(),
        client_id="acme", job_id="job-all-five",
        usage_store=store,
        network_manager=_StickyManager(),
    )
    _run(runner.run(_task()))

    events = _run(store.events_for_job("job-all-five"))
    types = {e.resource_type for e in events}
    assert types == {
        ResourceType.PAGE,
        ResourceType.TOKEN,
        ResourceType.BROWSER_SECOND,
        ResourceType.PROVIDER_CREDIT,
        ResourceType.VISION_CALL,
    }