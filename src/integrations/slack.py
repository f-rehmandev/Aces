"""
Slack connector — spec §47A (Tier 2 notification).

Posts a notification to a Slack Incoming Webhook. This is the simplest
Slack integration: one HTTPS URL, one POST. No OAuth, no app install,
no scopes to manage — the operator creates a webhook in Slack and
pastes the URL into .env.

Envelope → Slack message:
    The ACES alert envelope (see src/alerts/routing.py) carries
    severity / title / message / event_type / client_id / occurred_at.
    We shape it into:
        - a colored Slack attachment
          (danger=critical, warning=warning, good=info)
        - a bold title
        - the message body
        - a footer showing event_type · client · timestamp

Why this connector is thin:
    Slack is a *destination* for messages, not a workflow tool. We
    don't try to be clever. We don't parse blocks. We just deliver a
    readable message to a channel the operator already watches.

Design notes:
    - Only DELIVERS_NOTIFICATIONS. Slack is not a dataset destination;
      publish() short-circuits at BaseConnector's capability check.
    - Slack has no server-side idempotency, so we track sent dedup
      keys in-process. Best-effort; cross-restart dedup is out of scope.
    - Transport is injectable so tests never hit the network.
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol

from src.integrations.base import BaseConnector
from src.integrations.types import (
    ConnectorCapability,
    ConnectorError,
    ConnectorTestResult,
    ConnectorType,
    DatasetReference,
    DeliveryResult,
)


logger = logging.getLogger("integrations.slack")


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

class HttpTransport(Protocol):
    async def post_json(
        self, url: str, body: dict, timeout: float,
    ) -> tuple[int, str]: ...


class HttpxTransport:
    """Default transport. Lazy-imports httpx."""

    async def post_json(
        self, url: str, body: dict, timeout: float,
    ) -> tuple[int, str]:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=body)
            return r.status_code, r.text or ""


# ---------------------------------------------------------------------------
# Severity → Slack attachment color
# ---------------------------------------------------------------------------

# Slack's built-in named colors: "good" (green), "warning" (yellow),
# "danger" (red). Anything else is treated as a hex code.
_COLOR_BY_SEVERITY: dict[str, str] = {
    "critical": "danger",
    "warning":  "warning",
    "info":     "good",
}


def _color_for(severity: str) -> str:
    return _COLOR_BY_SEVERITY.get((severity or "").lower(), "#8190FF")


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class SlackConnector(BaseConnector):
    connector_type = ConnectorType.SLACK

    def __init__(
        self,
        webhook_url: str,
        channel: str = "",
        username: str = "ACES",
        icon_emoji: str = ":robot_face:",
        timeout_seconds: int = 15,
        connector_id: Optional[str] = None,
        transport: Optional[HttpTransport] = None,
        allow_http: bool = False,
    ):
        if not webhook_url:
            raise ConnectorError("slack webhook_url is required")

        url_lower = webhook_url.lower()
        if url_lower.startswith("http://"):
            if not allow_http:
                raise ConnectorError(
                    "slack webhook_url must be HTTPS "
                    "(pass allow_http=True only for local testing)"
                )
        elif not url_lower.startswith("https://"):
            raise ConnectorError(
                f"slack webhook_url must be http(s), got {webhook_url!r}"
            )

        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji
        self.timeout_seconds = int(timeout_seconds)
        self._transport = transport or HttpxTransport()
        self._sent_dedup_keys: set[str] = set()

        super().__init__(
            connector_id=connector_id,
            capabilities={ConnectorCapability.DELIVERS_NOTIFICATIONS},
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    async def _do_test(self) -> ConnectorTestResult:
        # Structural check only. Posting a real message every time
        # test_connection() runs would spam the operator's channel —
        # real delivery is validated the first time send() is called.
        return ConnectorTestResult(
            ok=True,
            message=(
                f"slack webhook configured "
                f"(channel override: {self.channel or 'default'})"
            ),
        )

    async def _do_publish(
        self,
        ref: DatasetReference,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        # Unreachable in practice — BaseConnector.publish() short-
        # circuits before we get here because SLACK isn't declared as
        # DELIVERS_FILES or DELIVERS_ROWS. Kept for ABC completeness.
        raise ConnectorError(
            "SlackConnector does not deliver datasets; use send() "
            "for notifications"
        )

    async def _do_send(
        self,
        payload: dict,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        metadata = dict(metadata or {})

        # --- process-local idempotency ---
        dedup_key = str(metadata.get("idempotency_key") or "")
        if dedup_key and dedup_key in self._sent_dedup_keys:
            return DeliveryResult(
                ok=True,
                destination=self.channel or "default",
                metadata={"idempotent_replay": True},
            )

        body = self._build_body(payload)

        try:
            status, text = await self._transport.post_json(
                self.webhook_url, body, self.timeout_seconds,
            )
        except Exception as e:
            return DeliveryResult(
                ok=False,
                destination=self.channel or "default",
                error=f"transport error: {type(e).__name__}: {e}",
                attempts=1,
            )

        # Slack returns HTTP 200 with body "ok" on success. Anything
        # else is a real failure: 400 malformed, 404 dead webhook,
        # 429 rate-limited.
        if not (200 <= status < 300):
            return DeliveryResult(
                ok=False,
                destination=self.channel or "default",
                error=f"HTTP {status}: {text[:200]}",
                attempts=1,
                metadata={"response_status": status},
            )

        if dedup_key:
            self._sent_dedup_keys.add(dedup_key)

        return DeliveryResult(
            ok=True,
            destination=self.channel or "default",
            bytes_sent=len(str(body).encode("utf-8")),
            attempts=1,
            metadata={
                "response_status": status,
                "channel": self.channel or "(webhook default)",
            },
        )

    # ------------------------------------------------------------------
    # Payload shaping
    # ------------------------------------------------------------------
    def _build_body(self, payload: dict) -> dict:
        """
        Convert the ACES alert envelope into a Slack message.

        Reads (all optional): severity, title, message, event_type,
        client_id, occurred_at. Unknown keys are ignored rather than
        dumped into the body — Slack rejects payloads with unknown
        top-level fields, so we only emit what Slack actually accepts.
        """
        severity = str(payload.get("severity") or "info").lower()
        title = str(payload.get("title") or payload.get("event_type") or "ACES alert")
        message = str(payload.get("message") or "")
        event_type = str(payload.get("event_type") or "")
        client_id = str(payload.get("client_id") or "")
        occurred_at = str(payload.get("occurred_at") or "")

        footer_parts = [p for p in (event_type, client_id, occurred_at) if p]
        footer = "  ·  ".join(footer_parts)

        attachment: dict = {
            "color": _color_for(severity),
            "title": title,
            "text": message,
            "mrkdwn_in": ["text"],
        }
        if footer:
            attachment["footer"] = footer

        body: dict = {
            # Slack's top-level `text` is the notification preview
            # (phone lockscreen, desktop notification). Keep it a
            # one-liner so it reads well truncated.
            "text": f"[{severity.upper()}] {title}: {message}",
            "attachments": [attachment],
        }
        if self.channel:
            body["channel"] = self.channel
        if self.username:
            body["username"] = self.username
        if self.icon_emoji:
            body["icon_emoji"] = self.icon_emoji

        return body


# ---------------------------------------------------------------------------
# Smoke test — injectable fake transport, no network
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    class FakeTransport:
        def __init__(self, status=200, text="ok"):
            self.status = status
            self.text = text
            self.calls: list[tuple[str, dict, float]] = []

        async def post_json(self, url, body, timeout):
            self.calls.append((url, body, timeout))
            return self.status, self.text

    async def run():
        # ---- 1. Construction validation ----
        try:
            SlackConnector(webhook_url="")
            raise AssertionError("expected ConnectorError")
        except ConnectorError:
            pass

        try:
            SlackConnector(webhook_url="http://hooks.slack.com/x")
            raise AssertionError("expected ConnectorError for http")
        except ConnectorError as e:
            assert "HTTPS" in str(e)

        try:
            SlackConnector(webhook_url="ftp://x")
            raise AssertionError("expected ConnectorError for bad scheme")
        except ConnectorError:
            pass

        # Explicit HTTP override is allowed for local testing
        SlackConnector(
            webhook_url="http://localhost:9000/hook", allow_http=True,
        )

        # ---- 2. Capabilities ----
        c = SlackConnector(webhook_url="https://hooks.slack.com/services/X")
        assert c.connector_type == ConnectorType.SLACK
        assert c.supports(ConnectorCapability.DELIVERS_NOTIFICATIONS)
        assert not c.supports(ConnectorCapability.DELIVERS_FILES)
        assert not c.supports(ConnectorCapability.DELIVERS_ROWS)

        # ---- 3. test_connection ----
        fake = FakeTransport()
        c = SlackConnector(
            webhook_url="https://hooks.slack.com/services/X",
            transport=fake,
        )
        tr = await c.test_connection()
        assert tr.ok is True
        assert "slack" in tr.message.lower()

        # ---- 4. send() happy path ----
        envelope = {
            "event_type": "alert.job_failed",
            "title": "job failed",
            "message": "job j-42 failed: HTTP 503",
            "severity": "critical",
            "scope": "task",
            "scope_id": "t-1",
            "rule_id": "rule-1",
            "client_id": "acme",
            "occurred_at": "2026-09-27T12:00:00Z",
            "dedup_key": "dedup-abc",
        }
        dr = await c.send(envelope)
        assert dr.ok is True, dr.error
        assert len(fake.calls) == 1

        url_called, body, timeout = fake.calls[0]
        assert url_called == "https://hooks.slack.com/services/X"
        assert body["username"] == "ACES"
        assert "attachments" in body and len(body["attachments"]) == 1
        attachment = body["attachments"][0]
        assert attachment["color"] == "danger"
        assert attachment["title"] == "job failed"
        assert attachment["text"] == "job j-42 failed: HTTP 503"
        assert "alert.job_failed" in attachment["footer"]
        assert "acme" in attachment["footer"]

        # ---- 5. Severity color mapping ----
        for severity, expected_color in (
            ("critical", "danger"),
            ("warning",  "warning"),
            ("info",     "good"),
            ("bogus",    "#8190FF"),
        ):
            fake2 = FakeTransport()
            c2 = SlackConnector(
                webhook_url="https://hooks.slack.com/services/X",
                transport=fake2,
            )
            await c2.send({"severity": severity, "title": "x", "message": "y"})
            _, body2, _ = fake2.calls[0]
            assert body2["attachments"][0]["color"] == expected_color, \
                f"{severity} → {body2['attachments'][0]['color']}"

        # ---- 6. Channel + username override ----
        fake3 = FakeTransport()
        c3 = SlackConnector(
            webhook_url="https://hooks.slack.com/services/X",
            channel="#alerts",
            username="ACES Ops",
            transport=fake3,
        )
        await c3.send({"title": "x", "message": "y"})
        _, body3, _ = fake3.calls[0]
        assert body3["channel"] == "#alerts"
        assert body3["username"] == "ACES Ops"

        # ---- 7. HTTP error → ok=False ----
        bad = FakeTransport(status=400, text="invalid_payload")
        c_bad = SlackConnector(
            webhook_url="https://hooks.slack.com/services/X", transport=bad,
        )
        dr = await c_bad.send({"title": "x", "message": "y"})
        assert dr.ok is False
        assert "HTTP 400" in dr.error
        assert "invalid_payload" in dr.error

        # ---- 8. Transport exception → ok=False ----
        class Boom:
            async def post_json(self, *a, **k):
                raise RuntimeError("network down")

        c_boom = SlackConnector(
            webhook_url="https://hooks.slack.com/services/X", transport=Boom(),
        )
        dr = await c_boom.send({"title": "x", "message": "y"})
        assert dr.ok is False
        assert "network down" in dr.error

        # ---- 9. Idempotency ----
        fake4 = FakeTransport()
        c4 = SlackConnector(
            webhook_url="https://hooks.slack.com/services/X", transport=fake4,
        )
        r1 = await c4.send({"title": "x"}, metadata={"idempotency_key": "K"})
        r2 = await c4.send({"title": "x"}, metadata={"idempotency_key": "K"})
        assert r1.ok and r2.ok
        assert r2.metadata.get("idempotent_replay") is True
        assert len(fake4.calls) == 1  # only one actual HTTP call

        # ---- 10. publish() is rejected cleanly ----
        ref = DatasetReference(format="json", records=[{"a": 1}])
        dr_pub = await c.publish(ref)
        assert dr_pub.ok is False
        assert "does not support publish" in dr_pub.error

        print("SlackConnector OK.")

    asyncio.run(run())