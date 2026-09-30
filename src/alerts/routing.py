"""
AlertRouter — spec §47, §30A.

Takes `FiredAlert` objects from the AlertEngine and delivers them
through one or more connectors. This is where a rule's
`notification_targets` list becomes an actual Slack message, webhook
POST, or local file.

Target resolution order (per target string):
    1. Connector ID             — exact match in the registry
    2. Connector type name      — "slack" / "s3" / "webhook" / "local_file"
    3. Client ID                — every connector scoped to that client

A rule can list multiple targets; every one that resolves gets a
delivery attempt. Failures on one target never block the others.

Payload shape (standard envelope, sent to every connector):

    {
      "event_type": "alert.job_failed",
      "title":      "job failed",
      "message":    "job j-42 failed: HTTP 503",
      "severity":   "critical",
      "scope":      "task",
      "scope_id":   "t-1",
      "rule_id":    "...",
      "client_id":  "acme",
      "occurred_at":"2026-09-27T...Z",
      "dedup_key":  "..."
    }

Each connector shapes this for its destination — Slack's auto-format
turns `severity` into an emoji + bold title; a webhook just POSTs the
raw dict; a local file writes it to disk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from src.alerts.rules import AlertRule, FiredAlert, Severity
from src.integrations.registry import ConnectorRegistry
from src.integrations.types import (
    ConnectorCapability,
    DeliveryResult,
)


logger = logging.getLogger("alerts.routing")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    rule_id: str
    condition_type: str
    severity: str
    targets_attempted: list[str] = field(default_factory=list)
    targets_resolved: list[str] = field(default_factory=list)
    deliveries: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def any_succeeded(self) -> bool:
        return any(d.get("ok") for d in self.deliveries)

    @property
    def all_succeeded(self) -> bool:
        return bool(self.deliveries) and all(
            d.get("ok") for d in self.deliveries
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def alert_to_envelope(alert: FiredAlert, client_id: str = "") -> dict:
    """
    Convert a FiredAlert into the standard routing envelope. Every
    connector receives this dict via `send()` and adapts as needed.

    `event_type` uses the `alert.<condition>` convention so downstream
    subscribers can filter on it. The `title` and `message` fields are
    already human-readable (they come from AlertEngine/AlertRule).
    """
    return {
        "event_type": f"alert.{alert.condition_type}",
        "title": alert.condition_type.replace("_", " "),
        "message": alert.message,
        "severity": alert.severity,
        "scope": alert.scope,
        "scope_id": alert.scope_id,
        "rule_id": alert.rule_id,
        "client_id": client_id,
        "occurred_at": _utc_now_iso(),
        "dedup_key": alert.dedup_key,
        # Some connectors (Slack) also read `level` — alias it so we
        # don't have to update every connector to know about severity.
        "level": alert.severity,
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class AlertRouter:
    """
    Routes FiredAlerts to connectors.

    `registry` — the ConnectorRegistry to look up connectors.
    `client_id` — tenant scope. Passed to registry lookups and included
                  in the envelope.
    `min_severity` — optional filter. Alerts below this severity are
                     dropped without any delivery. Values follow
                     Severity enum: info < warning < critical.
    """

    _SEVERITY_ORDER = {
        Severity.INFO.value: 0,
        Severity.WARNING.value: 1,
        Severity.CRITICAL.value: 2,
    }

    def __init__(
        self,
        registry: ConnectorRegistry,
        client_id: str = "default",
        min_severity: str = Severity.INFO.value,
    ):
        self.registry = registry
        self.client_id = client_id
        self.min_severity = min_severity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def route(
        self,
        alert: FiredAlert,
        rule: Optional[AlertRule] = None,
    ) -> RoutingResult:
        """
        Deliver one alert to every target the rule declares (or, if
        no rule is given, to every notification-capable connector
        visible to this client).

        Never raises. Every failure is captured in the RoutingResult.
        """
        result = RoutingResult(
            rule_id=alert.rule_id,
            condition_type=alert.condition_type,
            severity=alert.severity,
        )

        # Severity filter
        if not self._passes_severity(alert.severity):
            result.errors.append(
                f"filtered: severity {alert.severity!r} below "
                f"min_severity {self.min_severity!r}"
            )
            return result

        # Determine targets
        targets = list((rule.notification_targets if rule else []) or [])
        if not targets:
            # No explicit targets — fall back to every notification
            # connector visible to this client.
            connectors = self.registry.by_capability(
                ConnectorCapability.DELIVERS_NOTIFICATIONS,
                client_id=self.client_id,
            )
            for c in connectors:
                result.targets_attempted.append(c.connector_id)
        else:
            result.targets_attempted = targets

        # Resolve + deliver
        envelope = alert_to_envelope(alert, client_id=self.client_id)

        for target in result.targets_attempted:
            connectors = self._resolve_target(target)
            if not connectors:
                result.errors.append(
                    f"target {target!r} did not resolve to any connector"
                )
                continue
            for connector in connectors:
                result.targets_resolved.append(connector.connector_id)
                try:
                    dr = await connector.send(envelope)
                except Exception as e:
                    result.errors.append(
                        f"{connector.connector_id}: "
                        f"{type(e).__name__}: {e}"
                    )
                    continue
                result.deliveries.append({
                    "connector_id": connector.connector_id,
                    "connector_type": dr.connector_type or
                        connector.connector_type.value,
                    "ok": dr.ok,
                    "destination": dr.destination,
                    "error": dr.error,
                })

        if not result.deliveries and not result.errors:
            result.errors.append("no delivery-capable connectors resolved")

        return result

    async def route_many(
        self,
        alerts: list[FiredAlert],
        rules_by_id: Optional[dict[str, AlertRule]] = None,
    ) -> list[RoutingResult]:
        """
        Convenience: route a batch of alerts. `rules_by_id` maps
        rule_id -> AlertRule so each alert uses its rule's targets.
        """
        out: list[RoutingResult] = []
        for a in alerts:
            rule = (rules_by_id or {}).get(a.rule_id)
            out.append(await self.route(a, rule=rule))
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _passes_severity(self, severity: str) -> bool:
        current = self._SEVERITY_ORDER.get(severity, 0)
        minimum = self._SEVERITY_ORDER.get(self.min_severity, 0)
        return current >= minimum

    def _resolve_target(self, target: str):
        """
        Return a list of connectors matching this target string.
        Empty list means the target didn't resolve.
        """
        # 1. Exact connector ID
        c = self.registry.get(target, client_id=self.client_id)
        if c is not None:
            return [c]

        # 2. Connector type name
        from src.integrations.types import ConnectorType
        try:
            ctype = ConnectorType(target)
        except ValueError:
            ctype = None
        if ctype is not None:
            return [
                x for x in self.registry.by_type(
                    ctype, client_id=self.client_id,
                )
                if x.supports(ConnectorCapability.DELIVERS_NOTIFICATIONS)
            ]

        # 3. Fall back — treat as "all notification connectors for
        #    this client" if the target string matches the client id.
        if target == self.client_id:
            return self.registry.by_capability(
                ConnectorCapability.DELIVERS_NOTIFICATIONS,
                client_id=self.client_id,
            )

        return []


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    from src.integrations.base import BaseConnector
    from src.integrations.types import (
        ConnectorTestResult,
        ConnectorType,
        DatasetReference,
    )

    class RecordingNotifier(BaseConnector):
        connector_type = ConnectorType.SLACK

        def __init__(self, connector_id: str, ok: bool = True):
            super().__init__(
                connector_id=connector_id,
                capabilities={ConnectorCapability.DELIVERS_NOTIFICATIONS},
            )
            self.ok = ok
            self.received: list[dict] = []

        async def _do_test(self):
            return ConnectorTestResult(ok=True)

        async def _do_publish(self, ref, *, metadata=None):
            return DeliveryResult(ok=False, error="not supported")

        async def _do_send(self, payload, *, metadata=None):
            self.received.append(payload)
            if not self.ok:
                return DeliveryResult(
                    ok=False, destination="x", error="simulated failure",
                )
            return DeliveryResult(ok=True, destination="simulated")

    async def run():
        reg = ConnectorRegistry()
        n1 = RecordingNotifier("slack-1")
        n2 = RecordingNotifier("slack-2", ok=False)
        n3 = RecordingNotifier("slack-other")
        reg.register(n1, client_id="acme")
        reg.register(n2, client_id="acme")
        reg.register(n3, client_id="other")

        router = AlertRouter(reg, client_id="acme")

        alert = FiredAlert(
            rule_id="rule-1",
            condition_type="job_failed",
            severity="critical",
            scope="task",
            scope_id="t-1",
            message="job j-42 failed: HTTP 503",
            dedup_key="dedup-abc",
        )

        # ---- Route by explicit connector id ----
        rule = AlertRule(
            condition_type="job_failed",
            notification_targets=["slack-1"],
        )
        result = await router.route(alert, rule=rule)
        assert len(result.deliveries) == 1
        assert result.deliveries[0]["ok"] is True
        assert result.any_succeeded
        assert len(n1.received) == 1
        env = n1.received[0]
        assert env["event_type"] == "alert.job_failed"
        assert env["severity"] == "critical"
        assert env["message"] == "job j-42 failed: HTTP 503"
        assert env["client_id"] == "acme"

        # ---- Route by connector type name ----
        n1.received.clear()
        rule2 = AlertRule(
            condition_type="job_failed",
            notification_targets=["slack"],
        )
        result = await router.route(alert, rule=rule2)
        # Both acme connectors get pinged; other-tenant one does not
        assert len(result.deliveries) == 2
        assert n1.received  # got one
        assert not n3.received

        # ---- Partial failure is captured, not raised ----
        result = await router.route(alert, rule=rule2)
        assert result.any_succeeded
        assert not result.all_succeeded
        failed = [d for d in result.deliveries if not d["ok"]]
        assert len(failed) == 1
        assert "simulated failure" in failed[0]["error"]

        # ---- No targets on the rule → fallback to all notification connectors ----
        n1.received.clear()
        rule3 = AlertRule(condition_type="job_failed")
        result = await router.route(alert, rule=rule3)
        assert len(result.deliveries) == 2
        assert n1.received

        # ---- Severity filter ----
        router_strict = AlertRouter(
            reg, client_id="acme", min_severity="critical",
        )
        warning_alert = FiredAlert(
            rule_id="r", condition_type="quality_below",
            severity="warning", scope="task", scope_id="t-1",
            message="quality below", dedup_key="d",
        )
        result = await router_strict.route(warning_alert, rule=rule)
        assert len(result.deliveries) == 0
        assert any("filtered" in e for e in result.errors)

        # ---- Unresolvable target ----
        rule4 = AlertRule(
            condition_type="job_failed",
            notification_targets=["does-not-exist"],
        )
        result = await router.route(alert, rule=rule4)
        assert len(result.deliveries) == 0
        assert any("did not resolve" in e for e in result.errors)

        # ---- Connector raises → BaseConnector normalizes to ok=False;
        #      other deliveries still proceed normally. ----
        class ExplodingNotifier(RecordingNotifier):
            async def _do_send(self, payload, *, metadata=None):
                raise RuntimeError("boom")

        reg.register(ExplodingNotifier("exploder"), client_id="acme")
        rule5 = AlertRule(
            condition_type="job_failed",
            notification_targets=["exploder", "slack-1"],
        )
        n1.received.clear()
        result = await router.route(alert, rule=rule5)
        # Two deliveries recorded: one that failed (via BaseConnector
        # turning the exception into ok=False), one that succeeded.
        assert len(result.deliveries) == 2
        by_id = {d["connector_id"]: d for d in result.deliveries}
        assert by_id["exploder"]["ok"] is False
        assert "boom" in by_id["exploder"]["error"]
        assert by_id["slack-1"]["ok"] is True
        assert result.any_succeeded
        assert not result.all_succeeded
        assert n1.received

        # ---- route_many ----
        results = await router.route_many(
            [alert, alert],
            rules_by_id={"rule-1": rule},
        )
        assert len(results) == 2

        # ---- Envelope shape ----
        env = alert_to_envelope(alert, client_id="acme")
        assert env["event_type"] == "alert.job_failed"
        assert env["title"] == "job failed"
        assert env["severity"] == "critical"
        assert env["level"] == "critical"   # alias for Slack
        assert "occurred_at" in env

        print("AlertRouter OK.")

    asyncio.run(run())