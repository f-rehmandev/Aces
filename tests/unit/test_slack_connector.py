"""
Unit tests for SlackConnector (spec §47A Tier 2 notification).

Covers:
    - Construction: required url, HTTPS enforcement, allow_http override
    - Capabilities: only DELIVERS_NOTIFICATIONS
    - test_connection: structural, does not post
    - send(): envelope → Slack body shaping, severity → color,
      channel / username / emoji overrides
    - HTTP error handling, transport exception handling
    - Idempotency (in-process)
    - publish() rejects cleanly
    - BaseConnector contract: never raises, always returns a result
"""
import asyncio

import pytest

from src.integrations.slack import SlackConnector, _color_for
from src.integrations.types import (
    ConnectorCapability,
    ConnectorError,
    ConnectorType,
    DatasetReference,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

URL = "https://hooks.slack.com/services/T000/B000/XYZ"


class FakeTransport:
    def __init__(self, status=200, text="ok"):
        self.status = status
        self.text = text
        self.calls: list[tuple[str, dict, float]] = []

    async def post_json(self, url, body, timeout):
        self.calls.append((url, body, timeout))
        return self.status, self.text


def _run(coro):
    return asyncio.run(coro)


def _conn(transport=None, **kw) -> SlackConnector:
    return SlackConnector(
        webhook_url=URL,
        transport=transport or FakeTransport(),
        **kw,
    )


def _envelope(**overrides) -> dict:
    base = {
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
    base.update(overrides)
    return base


# ===========================================================================
# Construction
# ===========================================================================

def test_construction_requires_url():
    with pytest.raises(ConnectorError):
        SlackConnector(webhook_url="")


def test_construction_rejects_http_by_default():
    with pytest.raises(ConnectorError) as exc:
        SlackConnector(webhook_url="http://hooks.slack.com/x")
    assert "HTTPS" in str(exc.value)


def test_construction_allows_http_with_override():
    c = SlackConnector(
        webhook_url="http://localhost:9000/hook", allow_http=True,
    )
    assert c.webhook_url == "http://localhost:9000/hook"


def test_construction_rejects_non_http_scheme():
    with pytest.raises(ConnectorError):
        SlackConnector(webhook_url="ftp://example.com/hook")


def test_construction_custom_connector_id():
    c = SlackConnector(webhook_url=URL, connector_id="my-slack")
    assert c.connector_id == "my-slack"


# ===========================================================================
# Capabilities
# ===========================================================================

def test_connector_type_is_slack():
    c = _conn()
    assert c.connector_type == ConnectorType.SLACK


def test_only_notification_capability():
    c = _conn()
    assert c.supports(ConnectorCapability.DELIVERS_NOTIFICATIONS)
    assert not c.supports(ConnectorCapability.DELIVERS_FILES)
    assert not c.supports(ConnectorCapability.DELIVERS_ROWS)
    assert not c.supports(ConnectorCapability.SUPPORTS_SIGNED_URLS)


# ===========================================================================
# test_connection
# ===========================================================================

def test_test_connection_ok_and_does_not_post():
    fake = FakeTransport()
    c = _conn(transport=fake)
    tr = _run(c.test_connection())
    assert tr.ok is True
    assert "slack" in tr.message.lower()
    # Critical: test_connection must NOT hit Slack's API
    assert fake.calls == []


def test_test_connection_mentions_channel_override():
    c = _conn(channel="#alerts")
    tr = _run(c.test_connection())
    assert "#alerts" in tr.message


# ===========================================================================
# send() — body shaping
# ===========================================================================

def test_send_posts_expected_body():
    fake = FakeTransport()
    c = _conn(transport=fake)
    dr = _run(c.send(_envelope()))

    assert dr.ok is True
    assert len(fake.calls) == 1

    url, body, timeout = fake.calls[0]
    assert url == URL
    assert body["username"] == "ACES"
    assert isinstance(body["attachments"], list)
    assert len(body["attachments"]) == 1
    assert timeout == c.timeout_seconds


def test_send_attachment_shape():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send(_envelope()))

    _, body, _ = fake.calls[0]
    attachment = body["attachments"][0]
    assert attachment["title"] == "job failed"
    assert attachment["text"] == "job j-42 failed: HTTP 503"
    assert attachment["mrkdwn_in"] == ["text"]
    assert "alert.job_failed" in attachment["footer"]
    assert "acme" in attachment["footer"]
    assert "2026-09-27" in attachment["footer"]


def test_send_top_level_text_is_preview_line():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send(_envelope()))
    _, body, _ = fake.calls[0]
    # Phone lock-screen preview line
    assert body["text"].startswith("[CRITICAL]")
    assert "job failed" in body["text"]


def test_send_footer_omitted_when_no_metadata():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"title": "x", "message": "y"}))
    _, body, _ = fake.calls[0]
    assert "footer" not in body["attachments"][0]


def test_send_title_falls_back_to_event_type():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"event_type": "some.event", "message": "y"}))
    _, body, _ = fake.calls[0]
    assert body["attachments"][0]["title"] == "some.event"


def test_send_title_falls_back_to_default():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"message": "y"}))
    _, body, _ = fake.calls[0]
    assert body["attachments"][0]["title"] == "ACES alert"


# ===========================================================================
# Severity → color mapping
# ===========================================================================

@pytest.mark.parametrize("severity,expected", [
    ("critical", "danger"),
    ("CRITICAL", "danger"),
    ("warning",  "warning"),
    ("info",     "good"),
    ("info",     "good"),
])
def test_severity_color_mapping(severity, expected):
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"severity": severity, "title": "x", "message": "y"}))
    _, body, _ = fake.calls[0]
    assert body["attachments"][0]["color"] == expected


def test_unknown_severity_falls_back_to_accent():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"severity": "bogus", "title": "x", "message": "y"}))
    _, body, _ = fake.calls[0]
    assert body["attachments"][0]["color"] == "#8190FF"


def test_missing_severity_defaults_to_info():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"title": "x", "message": "y"}))
    _, body, _ = fake.calls[0]
    assert body["attachments"][0]["color"] == "good"


def test_color_for_helper():
    assert _color_for("critical") == "danger"
    assert _color_for("WARNING") == "warning"
    assert _color_for("") == "#8190FF"
    assert _color_for("whatever") == "#8190FF"


# ===========================================================================
# Channel / username / emoji overrides
# ===========================================================================

def test_channel_override_added():
    fake = FakeTransport()
    c = _conn(transport=fake, channel="#alerts")
    _run(c.send(_envelope()))
    _, body, _ = fake.calls[0]
    assert body["channel"] == "#alerts"


def test_no_channel_omits_field():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send(_envelope()))
    _, body, _ = fake.calls[0]
    assert "channel" not in body


def test_username_override():
    fake = FakeTransport()
    c = _conn(transport=fake, username="ACES Ops")
    _run(c.send(_envelope()))
    _, body, _ = fake.calls[0]
    assert body["username"] == "ACES Ops"


def test_icon_emoji_override():
    fake = FakeTransport()
    c = _conn(transport=fake, icon_emoji=":fire:")
    _run(c.send(_envelope()))
    _, body, _ = fake.calls[0]
    assert body["icon_emoji"] == ":fire:"


def test_empty_username_omitted():
    fake = FakeTransport()
    c = _conn(transport=fake, username="")
    _run(c.send(_envelope()))
    _, body, _ = fake.calls[0]
    assert "username" not in body


# ===========================================================================
# HTTP errors / transport exceptions
# ===========================================================================

def test_http_400_returns_not_ok():
    bad = FakeTransport(status=400, text="invalid_payload")
    c = _conn(transport=bad)
    dr = _run(c.send(_envelope()))
    assert dr.ok is False
    assert "HTTP 400" in dr.error
    assert "invalid_payload" in dr.error


def test_http_404_dead_webhook_not_ok():
    bad = FakeTransport(status=404, text="no_service")
    c = _conn(transport=bad)
    dr = _run(c.send(_envelope()))
    assert dr.ok is False
    assert "HTTP 404" in dr.error


def test_http_429_rate_limited_not_ok():
    bad = FakeTransport(status=429, text="rate_limited")
    c = _conn(transport=bad)
    dr = _run(c.send(_envelope()))
    assert dr.ok is False
    assert "HTTP 429" in dr.error


def test_transport_exception_not_ok():
    class Boom:
        async def post_json(self, *a, **k):
            raise RuntimeError("network down")

    c = _conn(transport=Boom())
    dr = _run(c.send(_envelope()))
    assert dr.ok is False
    assert "network down" in dr.error


# ===========================================================================
# Result metadata
# ===========================================================================

def test_result_metadata_when_channel_set():
    c = _conn(channel="#alerts")
    dr = _run(c.send(_envelope()))
    assert dr.destination == "#alerts"
    assert dr.metadata.get("channel") == "#alerts"


def test_result_metadata_when_no_channel():
    c = _conn()
    dr = _run(c.send(_envelope()))
    assert dr.destination == "default"
    assert dr.metadata.get("channel") == "(webhook default)"


def test_result_connector_id_and_type_set():
    c = _conn(connector_id="slack-prod")
    dr = _run(c.send(_envelope()))
    assert dr.connector_id == "slack-prod"
    assert dr.connector_type == "slack"


# ===========================================================================
# Idempotency
# ===========================================================================

def test_idempotency_same_key_is_noop():
    fake = FakeTransport()
    c = _conn(transport=fake)
    r1 = _run(c.send(_envelope(), metadata={"idempotency_key": "K"}))
    r2 = _run(c.send(_envelope(), metadata={"idempotency_key": "K"}))
    assert r1.ok and r2.ok
    assert r2.metadata.get("idempotent_replay") is True
    assert len(fake.calls) == 1


def test_idempotency_different_keys_send_twice():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send(_envelope(), metadata={"idempotency_key": "K1"}))
    _run(c.send(_envelope(), metadata={"idempotency_key": "K2"}))
    assert len(fake.calls) == 2


def test_no_idempotency_key_always_sends():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send(_envelope()))
    _run(c.send(_envelope()))
    assert len(fake.calls) == 2


# ===========================================================================
# publish() rejected
# ===========================================================================

def test_publish_rejected_cleanly():
    c = _conn()
    ref = DatasetReference(format="json", records=[{"a": 1}])
    dr = _run(c.publish(ref))
    assert dr.ok is False
    assert "does not support publish" in dr.error


# ===========================================================================
# BaseConnector contract: never raises
# ===========================================================================

def test_send_never_raises_on_any_error():
    """
    BaseConnector.send() must convert every exception to ok=False.
    This is the contract every connector relies on.
    """
    class EveryError:
        async def post_json(self, *a, **k):
            raise ValueError("anything")

    c = _conn(transport=EveryError())
    dr = _run(c.send(_envelope()))
    assert dr.ok is False
    assert "anything" in dr.error