"""
Unit tests for the alert→incident bridge inside PipelineRunner
(spec §30A + §40.6).

Covers:
    - Backward compatibility: absent engine/tracker → no side effects
    - Case 1 (clean run, no alerts) → resolves prior incidents mentioning
      this task; leaves unrelated incidents open
    - Case 2 (alert fires) → opens/touches incidents; client scoping;
      task_id lands in affected_jobs
    - Full lifecycle: fail → open → succeed → resolve
    - Failure isolation: broken engine or broken tracker never crashes
      the run — errors land in `warnings`
    - Event payload shape: run metrics are actually delivered to the
      alert engine
"""
import asyncio

import pytest

from src.alerts.engine import AlertEngine
from src.alerts.rules import AlertRule, ConditionType
from src.core.task_spec import FieldSpec, Quality, Target, TaskSpec
from src.observability.store import InMemoryIncidentStore
from src.observability.tracker import IncidentTracker
from src.observability.types import Incident, IncidentStatus
from src.pipeline_runner import PipelineRunner
from src.quality.rules import QualityRules


# ---------------------------------------------------------------------------
# Fakes (same pattern as test_pipeline_runner.py so behaviour is familiar)
# ---------------------------------------------------------------------------

class FakeScraper:
    def __init__(self, html_map=None):
        self.html_map = html_map or {
            "https://93.184.216.34/a": "<html>SENTINEL</html>"
        }

    async def fetch_html(self, url, timeout=None):
        return self.html_map.get(url, "<html>SENTINEL</html>")

    async def fetch_screenshot(self, url, timeout=None):
        return b"png"


class FakeExtractor:
    def __init__(self, items=None):
        self.items = items if items is not None else [{"title": "A", "price": "$1"}]

    def extract_list(self, html, instruction):
        return list(self.items)

    def extract_from_image(self, img, instr):
        return []


def _task(task_id="t-fixed", min_records=1):
    spec = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=["https://93.184.216.34/a"]),
        fields=[FieldSpec(name="title")],
        quality=Quality(min_records=min_records),
    )
    spec.task_id = task_id
    return spec


def _run(coro):
    return asyncio.run(coro)


# =========================================================================
# Backward compatibility: bridge disabled unless BOTH are supplied
# =========================================================================

def test_default_runner_has_no_bridge():
    runner = PipelineRunner(FakeScraper(), FakeExtractor())
    assert runner.alert_engine is None
    assert runner.incident_tracker is None


def test_default_runner_produces_no_incident_warnings():
    runner = PipelineRunner(FakeScraper(), FakeExtractor())
    result = _run(runner.run(_task()))
    assert not any("incident" in w.lower() for w in result.warnings)
    assert not any("alert engine" in w.lower() for w in result.warnings)


def test_only_engine_set_is_noop():
    """Supplying only the engine must not activate the bridge."""
    engine = AlertEngine()
    engine.add_rule(AlertRule(
        condition_type=ConditionType.QUALITY_BELOW.value,
        threshold=0.99, cooldown_seconds=0,
    ))
    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=engine,
        incident_tracker=None,
    )
    result = _run(runner.run(_task()))
    assert not any("incident" in w.lower() for w in result.warnings)


def test_only_tracker_set_is_noop():
    """Supplying only the tracker must not activate the bridge."""
    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)
    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        incident_tracker=tracker,
        alert_engine=None,
    )
    _run(runner.run(_task()))
    assert _run(store.list_open()) == []


# =========================================================================
# Case 1: clean run + no alerts → resolve prior incidents
# =========================================================================

def test_clean_run_resolves_prior_incident():
    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)
    engine = AlertEngine()  # no rules; cannot fire

    prior = Incident(
        title="prior failure",
        dedup_key="k1",
        client_id="acme",
        affected_jobs=["t-A"],
    )
    _run(store.open_or_touch(prior))

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        alert_engine=engine,
        incident_tracker=tracker,
    )
    result = _run(runner.run(_task(task_id="t-A")))

    got = _run(store.get(prior.incident_id))
    assert got.status == IncidentStatus.RESOLVED
    assert got.resolved_at
    assert any("auto-resolved" in w for w in result.warnings)


def test_clean_run_leaves_unrelated_incidents_open():
    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)
    engine = AlertEngine()

    other = Incident(
        title="other", dedup_key="other",
        client_id="acme", affected_jobs=["t-OTHER"],
    )
    _run(store.open_or_touch(other))

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        alert_engine=engine,
        incident_tracker=tracker,
    )
    _run(runner.run(_task(task_id="t-MINE")))

    assert _run(store.get(other.incident_id)).status == IncidentStatus.OPEN


def test_clean_run_with_no_prior_incident_is_silent():
    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)
    engine = AlertEngine()

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        alert_engine=engine,
        incident_tracker=tracker,
    )
    result = _run(runner.run(_task()))
    assert not any("incident" in w.lower() for w in result.warnings)
    assert _run(store.list_open()) == []


# =========================================================================
# Case 2: alert fires → incident opens
# =========================================================================

def test_quality_below_alert_opens_incident():
    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)
    engine = AlertEngine()
    engine.add_rule(AlertRule(
        condition_type=ConditionType.QUALITY_BELOW.value,
        threshold=0.99, severity="critical", cooldown_seconds=0,
    ))

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),  # forces quality failure
        alert_engine=engine,
        incident_tracker=tracker,
    )
    result = _run(runner.run(_task(task_id="t-FAIL")))

    opens = _run(store.list_open(client_id="acme"))
    assert len(opens) == 1
    assert "quality_below" in opens[0].title
    assert any("incident" in w.lower() for w in result.warnings)


def test_opened_incident_is_scoped_to_client():
    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)
    engine = AlertEngine()
    engine.add_rule(AlertRule(
        condition_type=ConditionType.QUALITY_BELOW.value,
        threshold=0.99, cooldown_seconds=0,
    ))

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="tenant-x",
        quality_rules=QualityRules(min_records=100),
        alert_engine=engine,
        incident_tracker=tracker,
    )
    _run(runner.run(_task(task_id="t-x")))

    opens = _run(store.list_open(client_id="tenant-x"))
    assert len(opens) == 1
    assert opens[0].client_id == "tenant-x"
    # And it's NOT visible to another tenant
    assert _run(store.list_open(client_id="other")) == []


def test_opened_incident_has_task_in_affected_jobs():
    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)
    engine = AlertEngine()
    engine.add_rule(AlertRule(
        condition_type=ConditionType.QUALITY_BELOW.value,
        threshold=0.99, cooldown_seconds=0,
    ))

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=engine,
        incident_tracker=tracker,
    )
    _run(runner.run(_task(task_id="t-xyz")))

    opens = _run(store.list_open(client_id="acme"))
    assert "t-xyz" in opens[0].affected_jobs


# =========================================================================
# Full lifecycle: fail → open → succeed → resolve
# =========================================================================

def test_fail_then_succeed_opens_then_resolves():
    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)
    engine = AlertEngine()
    engine.add_rule(AlertRule(
        condition_type=ConditionType.QUALITY_BELOW.value,
        threshold=0.99, cooldown_seconds=0,
    ))

    # ---- Run 1: fails → incident opens ----
    bad = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=engine,
        incident_tracker=tracker,
    )
    _run(bad.run(_task(task_id="t-lifecycle")))

    opens = _run(store.list_open(client_id="acme"))
    assert len(opens) == 1
    iid = opens[0].incident_id

    # ---- Run 2: succeeds → incident resolves ----
    good = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=1),
        alert_engine=engine,
        incident_tracker=tracker,
    )
    result = _run(good.run(_task(task_id="t-lifecycle")))

    got = _run(store.get(iid))
    assert got.status == IncidentStatus.RESOLVED
    assert any("auto-resolved" in w for w in result.warnings)


# =========================================================================
# Failure isolation: bridge errors never crash the run
# =========================================================================

def test_alert_engine_exception_is_caught():
    class ExplodingEngine:
        def evaluate(self, event):
            raise RuntimeError("engine down")

    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        alert_engine=ExplodingEngine(),
        incident_tracker=tracker,
    )
    result = _run(runner.run(_task()))

    assert any("alert engine failed" in w for w in result.warnings)
    # Pipeline still completed and produced its output
    assert len(result.records) == 1


def test_tracker_open_exception_is_caught():
    class ExplodingTracker:
        async def on_alerts_fired(self, alerts, *, client_id=""):
            raise RuntimeError("tracker down")
        async def on_job_succeeded(self, job_id):
            raise RuntimeError("tracker down")

    engine = AlertEngine()
    engine.add_rule(AlertRule(
        condition_type=ConditionType.QUALITY_BELOW.value,
        threshold=0.99, cooldown_seconds=0,
    ))

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=engine,
        incident_tracker=ExplodingTracker(),
    )
    result = _run(runner.run(_task()))

    assert any("incident open failed" in w for w in result.warnings)
    assert len(result.records) == 1  # still ran to completion


def test_tracker_resolve_exception_is_caught():
    class BrokenResolveTracker:
        async def on_job_succeeded(self, job_id):
            raise RuntimeError("resolve down")
        async def on_alerts_fired(self, alerts, *, client_id=""):
            return []

    engine = AlertEngine()  # no rules; run is clean so resolve path runs

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        alert_engine=engine,
        incident_tracker=BrokenResolveTracker(),
    )
    result = _run(runner.run(_task()))

    assert any("incident resolve failed" in w for w in result.warnings)


# =========================================================================
# Event payload shape
# =========================================================================

def test_event_payload_carries_run_metrics():
    captured = []

    class CaptureEngine:
        def evaluate(self, event):
            captured.append(event)
            return []

    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        alert_engine=CaptureEngine(),
        incident_tracker=tracker,
    )
    _run(runner.run(_task(task_id="t-payload")))

    assert len(captured) == 1
    event = captured[0]
    assert event.kind == "job_completed"
    assert event.client_id == "acme"
    assert event.task_id == "t-payload"
    assert event.job_id == "t-payload"
    for key in ("records", "quality_score", "confidence_mean", "conflicts"):
        assert key in event.payload, f"missing payload key: {key}"
        