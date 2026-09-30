"""
Unit tests for storage-byte metering (spec §41.4).

Covers:
    - A run that writes a workbook records a STORAGE_BYTE event
    - The event quantity equals the actual file size on disk
    - `writes` metadata reflects the number of files written
    - No STORAGE_BYTE event when output_path isn't provided
    - No STORAGE_BYTE event when the quality gate blocks the workbook
    - A missing workbook file is tolerated (zero bytes, no crash)
"""
import asyncio
from pathlib import Path

import pytest

from src.core.task_spec import FieldSpec, Quality, Target, TaskSpec
from src.pipeline_runner import PipelineRunner
from src.usage.store import InMemoryUsageStore
from src.usage.types import ResourceType


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeScraper:
    async def fetch_html(self, url, timeout=None):
        return "<html><body>" + ("x" * 1000) + "</body></html>"

    async def fetch_screenshot(self, url, timeout=None):
        return b""


class _FakeExtractor:
    def extract_list(self, html, instruction):
        return [{"title": "A"}]

    def extract_from_image(self, img, instr):
        return []


def _task(task_id="t-storage"):
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
        quality=Quality(min_records=1),
    )
    spec.task_id = task_id
    return spec


# ---------------------------------------------------------------------------
# Workbook-write path
# ---------------------------------------------------------------------------

def test_run_with_workbook_writes_storage_byte_event(tmp_path: Path):
    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-storage-1",
        usage_store=store,
    )

    out = tmp_path / "report.xlsx"
    _run(runner.run(_task(), output_path=out))

    assert out.exists(), "workbook should have been written"

    events = _run(store.events_for_job("job-storage-1"))
    by_type = {e.resource_type: e for e in events}

    assert ResourceType.STORAGE_BYTE in by_type
    sb = by_type[ResourceType.STORAGE_BYTE]
    assert sb.quantity == float(out.stat().st_size)
    assert sb.metadata["writes"] == 1
    assert sb.unit == "byte"


def test_storage_event_quantity_matches_real_file(tmp_path: Path):
    """Direct comparison: event quantity == os.stat().st_size."""
    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-storage-size",
        usage_store=store,
    )
    out = tmp_path / "sized.xlsx"
    _run(runner.run(_task(), output_path=out))

    events = _run(store.events_for_job("job-storage-size"))
    sb = next(
        e for e in events
        if e.resource_type == ResourceType.STORAGE_BYTE
    )
    assert sb.quantity == float(out.stat().st_size)
    # Sanity: an xlsx is never zero bytes.
    assert sb.quantity > 0


# ---------------------------------------------------------------------------
# No workbook → no event
# ---------------------------------------------------------------------------

def test_no_output_path_means_no_storage_event():
    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-no-output",
        usage_store=store,
    )
    _run(runner.run(_task()))   # no output_path

    events = _run(store.events_for_job("job-no-output"))
    types = {e.resource_type for e in events}
    assert ResourceType.STORAGE_BYTE not in types
    assert ResourceType.PAGE in types


def test_quality_gate_blocks_workbook_and_storage_event(tmp_path: Path):
    """
    When the quality gate fails, the pipeline doesn't write a workbook,
    so no STORAGE_BYTE event should be recorded.
    """
    # Force the quality gate to fail: require more records than the fake
    # extractor produces.
    spec = _task()
    spec.quality = Quality(min_records=100)

    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-blocked-workbook",
        usage_store=store,
    )
    out = tmp_path / "never_written.xlsx"
    _run(runner.run(spec, output_path=out))

    assert not out.exists(), "quality gate should have blocked the workbook"
    events = _run(store.events_for_job("job-blocked-workbook"))
    types = {e.resource_type for e in events}
    assert ResourceType.STORAGE_BYTE not in types


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_missing_workbook_file_is_tolerated(tmp_path: Path, monkeypatch):
    """
    If WorkbookResult.path points at a file that's vanished (unlikely,
    but possible on a network share), the pipeline must not crash and
    the storage event must be skipped.
    """
    store = InMemoryUsageStore()

    # Monkeypatch WorkbookBuilder.build to return a result whose path
    # doesn't exist on disk.
    from src.output.workbook import WorkbookResult
    fake_result = WorkbookResult(
        path=tmp_path / "does_not_exist.xlsx",
        sheet_names=["Data"],
    )

    class _FakeBuilder:
        def build(self, *args, **kwargs):
            return fake_result

    import src.pipeline_runner as pr_mod
    monkeypatch.setattr(pr_mod, "WorkbookBuilder", lambda: _FakeBuilder())

    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-vanished",
        usage_store=store,
    )
    result = _run(runner.run(
        _task(), output_path=tmp_path / "whatever.xlsx",
    ))

    # Run completed, no crash.
    assert result is not None
    events = _run(store.events_for_job("job-vanished"))
    types = {e.resource_type for e in events}
    assert ResourceType.STORAGE_BYTE not in types


# ---------------------------------------------------------------------------
# Integration: STORAGE_BYTE + PAGE + TOKEN all fire together
# ---------------------------------------------------------------------------

def test_workbook_run_records_storage_and_page(tmp_path: Path):
    store = InMemoryUsageStore()
    runner = PipelineRunner(
        _FakeScraper(), _FakeExtractor(),
        client_id="acme", job_id="job-storage-and-page",
        usage_store=store,
    )
    _run(runner.run(_task(), output_path=tmp_path / "r.xlsx"))

    events = _run(store.events_for_job("job-storage-and-page"))
    types = {e.resource_type for e in events}
    assert ResourceType.PAGE in types
    assert ResourceType.STORAGE_BYTE in types