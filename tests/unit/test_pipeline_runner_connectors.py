"""
Unit tests for the alert→connector routing inside PipelineRunner
(spec §30A + §47A).

Verifies the last link in the observability chain:
    rule fires → alert opens incident → alert ALSO reaches a
    notification connector (Slack, webhook, etc.)

Backward compatibility: with no registry supplied, nothing changes.

Delivery is observed via a fake notification connector that records
every send() call. No network is touched.
"""
import asyncio

import pytest

from src.alerts.engine import AlertEngine
from src.alerts.rules import AlertRule, ConditionType
from src.core.task_spec import FieldSpec, Quality, Target, TaskSpec
from src.integrations.base import BaseConnector
from src.integrations.registry import ConnectorRegistry
from src.integrations.types import (
    ConnectorCapability,
    ConnectorTestResult,
    ConnectorType,
    DatasetReference,
    DeliveryResult,
)
from src.observability.store import InMemoryIncidentStore
from src.observability.tracker import IncidentTracker
from src.pipeline_runner import PipelineRunner
from src.quality.rules import QualityRules


# ---------------------------------------------------------------------------
# Fakes
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


class RecordingNotifier(BaseConnector):
    """
    Notification connector that records every send() payload.
    Behaves like a Slack webhook from the router's point of view.
    """
    connector_type = ConnectorType.SLACK

    def __init__(self, connector_id: str = "notifier-1", should_fail: bool = False):
        super().__init__(
            connector_id=connector_id,
            capabilities={ConnectorCapability.DELIVERS_NOTIFICATIONS},
        )
        self.should_fail = should_fail
        self.received: list[dict] = []

    async def _do_test(self):
        return ConnectorTestResult(ok=True)

    async def _do_publish(self, ref, *, metadata=None):
        return DeliveryResult(ok=False, error="not supported")

    async def _do_send(self, payload, *, metadata=None):
        self.received.append(payload)
        if self.should_fail:
            return DeliveryResult(
                ok=False, destination="simulated", error="simulated failure",
            )
        return DeliveryResult(ok=True, destination="simulated")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

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


def _fail_engine() -> AlertEngine:
    """Engine whose one rule always fires on job_completed."""
    engine = AlertEngine()
    engine.add_rule(AlertRule(
        condition_type=ConditionType.QUALITY_BELOW.value,
        threshold=0.99,
        severity="critical",
        cooldown_seconds=0,
    ))
    return engine


def _silent_engine() -> AlertEngine:
    """Engine with no rules; never fires."""
    return AlertEngine()


# ===========================================================================
# Backward compatibility
# ===========================================================================

def test_default_runner_has_no_registry():
    r = PipelineRunner(FakeScraper(), FakeExtractor())
    assert r.connector_registry is None


def test_no_registry_means_no_delivery_and_no_error():
    """Nothing changes for callers who don't supply a registry."""
    notifier = RecordingNotifier()
    reg = ConnectorRegistry()
    reg.register(notifier, client_id="acme")

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        alert_engine=_fail_engine(),
        incident_tracker=IncidentTracker(InMemoryIncidentStore()),
        # connector_registry deliberately omitted
    )
    result = _run(runner.run(_task()))
    assert notifier.received == []
    # No routing-related warnings either
    assert not any("delivered" in w.lower() for w in result.warnings)


# ===========================================================================
# Fired alerts reach the connector
# ===========================================================================

def test_fired_alert_reaches_notification_connector():
    notifier = RecordingNotifier()
    reg = ConnectorRegistry()
    reg.register(notifier, client_id="acme")

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),  # forces quality fail
        alert_engine=_fail_engine(),
        incident_tracker=IncidentTracker(InMemoryIncidentStore()),
        connector_registry=reg,
    )
    result = _run(runner.run(_task()))

    assert len(notifier.received) == 1
    env = notifier.received[0]
    assert env["event_type"] == "alert.quality_below"
    assert env["severity"] == "critical"
    assert env["client_id"] == "acme"
    assert "quality" in env["message"].lower() or "0.99" in env["message"]
    assert any("delivered 1" in w for w in result.warnings)


def test_message_carries_task_id():
    """The alert that reached the connector names the task that failed."""
    notifier = RecordingNotifier()
    reg = ConnectorRegistry()
    reg.register(notifier, client_id="acme")

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=_fail_engine(),
        incident_tracker=IncidentTracker(InMemoryIncidentStore()),
        connector_registry=reg,
    )
    _run(runner.run(_task(task_id="t-alert-payload")))

    # The envelope shape is standard; the raw event dict carries the task
    envelope = notifier.received[0]
    assert envelope["scope"] in ("task", "job", "client", "project", "domain")
    assert envelope["scope_id"] == ""  # our fake rule has no scope set


# ===========================================================================
# Clean run = no delivery
# ===========================================================================

def test_clean_run_does_not_deliver_to_connector():
    notifier = RecordingNotifier()
    reg = ConnectorRegistry()
    reg.register(notifier, client_id="acme")

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=1),  # passes
        alert_engine=_silent_engine(),              # no rules, no fires
        incident_tracker=IncidentTracker(InMemoryIncidentStore()),
        connector_registry=reg,
    )
    result = _run(runner.run(_task()))

    assert notifier.received == []
    assert not any("delivered" in w.lower() for w in result.warnings)


# ===========================================================================
# Client scoping — only the right tenant's connector gets pinged
# ===========================================================================

def test_wrong_tenant_connector_not_pinged():
    acme_notifier = RecordingNotifier("acme-notifier")
    other_notifier = RecordingNotifier("other-notifier")

    reg = ConnectorRegistry()
    reg.register(acme_notifier, client_id="acme")
    reg.register(other_notifier, client_id="other")

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=_fail_engine(),
        incident_tracker=IncidentTracker(InMemoryIncidentStore()),
        connector_registry=reg,
    )
    _run(runner.run(_task()))

    assert len(acme_notifier.received) == 1
    assert other_notifier.received == []


# ===========================================================================
# Multiple connectors → broadcast
# ===========================================================================

def test_broadcast_to_all_notification_connectors():
    n1 = RecordingNotifier("n1")
    n2 = RecordingNotifier("n2")
    n3 = RecordingNotifier("n3")

    reg = ConnectorRegistry()
    for n in (n1, n2, n3):
        reg.register(n, client_id="acme")

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=_fail_engine(),
        incident_tracker=IncidentTracker(InMemoryIncidentStore()),
        connector_registry=reg,
    )
    _run(runner.run(_task()))

    for n in (n1, n2, n3):
        assert len(n.received) == 1, f"{n.connector_id} missed the alert"


# ===========================================================================
# Failure isolation
# ===========================================================================

def test_failing_connector_does_not_crash_run():
    ok = RecordingNotifier("ok")
    broken = RecordingNotifier("broken", should_fail=True)

    reg = ConnectorRegistry()
    reg.register(ok, client_id="acme")
    reg.register(broken, client_id="acme")

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=_fail_engine(),
        incident_tracker=IncidentTracker(InMemoryIncidentStore()),
        connector_registry=reg,
    )
    result = _run(runner.run(_task()))

    # Run completed, records are still there
    assert len(result.records) == 1
    # The good notifier still received the alert
    assert len(ok.received) == 1
    # And at least one "delivered" warning is emitted
    assert any("delivered" in w.lower() for w in result.warnings)


def test_registry_with_no_notification_connectors():
    """A registry that has file-only connectors still doesn't crash."""
    class FileOnly(BaseConnector):
        connector_type = ConnectorType.LOCAL_FILE
        def __init__(self):
            super().__init__(
                connector_id="file-only",
                capabilities={ConnectorCapability.DELIVERS_FILES},
            )
        async def _do_test(self):
            return ConnectorTestResult(ok=True)
        async def _do_publish(self, ref, *, metadata=None):
            return DeliveryResult(ok=True, destination="x")
        async def _do_send(self, payload, *, metadata=None):
            return DeliveryResult(ok=False, error="not supported")

    reg = ConnectorRegistry()
    reg.register(FileOnly(), client_id="acme")

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=_fail_engine(),
        incident_tracker=IncidentTracker(InMemoryIncidentStore()),
        connector_registry=reg,
    )
    result = _run(runner.run(_task()))

    # Run still completes, warnings may mention "0 delivered"
    assert len(result.records) == 1


def test_registry_raises_does_not_crash_run():
    """A registry that explodes during lookup is caught."""
    class ExplodingRegistry(ConnectorRegistry):
        def all(self, *, client_id=None):
            raise RuntimeError("registry down")
        def by_capability(self, capability, *, client_id=None):
            raise RuntimeError("registry down")

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=_fail_engine(),
        incident_tracker=IncidentTracker(InMemoryIncidentStore()),
        connector_registry=ExplodingRegistry(),
    )
    result = _run(runner.run(_task()))

    assert len(result.records) == 1
    # Routing error should land in warnings, not propagate
    assert any("routing failed" in w.lower() for w in result.warnings)


# ===========================================================================
# Full chain: incident + delivery both fire
# ===========================================================================

def test_full_chain_incident_and_delivery_both_fire():
    """The last-mile integration test."""
    store = InMemoryIncidentStore()
    tracker = IncidentTracker(store)
    notifier = RecordingNotifier()
    reg = ConnectorRegistry()
    reg.register(notifier, client_id="acme")

    runner = PipelineRunner(
        FakeScraper(), FakeExtractor(),
        client_id="acme",
        quality_rules=QualityRules(min_records=100),
        alert_engine=_fail_engine(),
        incident_tracker=tracker,
        connector_registry=reg,
    )
    result = _run(runner.run(_task(task_id="t-full-chain")))

    # 1. Incident opened
    opens = _run(store.list_open(client_id="acme"))
    assert len(opens) == 1
    assert "t-full-chain" in opens[0].affected_jobs

    # 2. Notification delivered
    assert len(notifier.received) == 1
    envelope = notifier.received[0]
    assert envelope["severity"] == "critical"

    # 3. Both warning lines present
    joined = " ".join(result.warnings).lower()
    assert "incident" in joined
    assert "delivered" in joined