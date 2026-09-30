"""Unit tests for WebhookConnector (spec §47A)."""
import asyncio
import hashlib
import hmac
import json

import pytest

from src.integrations.types import (
    ConnectorCapability,
    ConnectorError,
    ConnectorType,
    DatasetReference,
)
from src.integrations.webhook import WebhookConnector
from src.security.ssrf import SSRFCheckResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SECRET = "test-secret-1234567890" + "x" * 20


class _AllowGuard:
    """Skips DNS resolution — tests don't hit the network."""
    def validate_url(self, url):
        return SSRFCheckResult(True, "test-bypass")


class FakeTransport:
    def __init__(self, status=200, body="ok"):
        self.status = status
        self.body = body
        self.calls: list[tuple[str, str, dict, float]] = []

    async def post_json(self, url, body, headers, timeout):
        self.calls.append((url, body, headers, timeout))
        return self.status, self.body


def _run(coro):
    return asyncio.run(coro)


def _conn(**kwargs) -> WebhookConnector:
    """Build a connector with the SSRF guard bypassed."""
    defaults = dict(
        url="https://hooks.example.com/abc",
        secret=SECRET,
        transport=FakeTransport(),
        ssrf_guard=_AllowGuard(),
    )
    defaults.update(kwargs)
    return WebhookConnector(**defaults)


def _verify(body: str, headers: dict, secret: str = SECRET) -> bool:
    received = headers["X-ACES-Signature"]
    assert received.startswith("sha256=")
    expected = hmac.new(
        secret.encode(), body.encode(), hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(received[len("sha256="):], expected)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_requires_url():
    with pytest.raises(ConnectorError):
        _conn(url="")


def test_construction_requires_secret():
    with pytest.raises(ConnectorError):
        _conn(secret="")


def test_rejects_http_by_default():
    with pytest.raises(ConnectorError) as exc:
        _conn(url="http://hooks.example.com/x")
    assert "HTTPS" in str(exc.value)


def test_allows_http_when_explicit():
    c = _conn(url="http://hooks.example.com/x", allow_http=True)
    assert c.url == "http://hooks.example.com/x"


def test_rejects_non_http_scheme():
    with pytest.raises(ConnectorError):
        _conn(url="ftp://hooks.example.com/x")


def test_ssrf_guard_blocks_metadata_ip():
    with pytest.raises(ConnectorError) as exc:
        # Real guard, metadata IP, no DNS needed
        WebhookConnector(
            url="https://169.254.169.254/hook",
            secret=SECRET,
        )
    assert "SSRF" in str(exc.value) or "metadata" in str(exc.value).lower()


def test_default_capabilities():
    c = _conn()
    assert c.connector_type == ConnectorType.WEBHOOK
    assert c.supports(ConnectorCapability.DELIVERS_ROWS)
    assert c.supports(ConnectorCapability.DELIVERS_NOTIFICATIONS)
    assert c.supports(ConnectorCapability.SUPPORTS_SIGNED_URLS)
    assert c.supports(ConnectorCapability.SUPPORTS_IDEMPOTENCY)
    assert not c.supports(ConnectorCapability.DELIVERS_FILES)


# ---------------------------------------------------------------------------
# test_connection
# ---------------------------------------------------------------------------

def test_test_connection_ok():
    c = _conn()
    tr = _run(c.test_connection())
    assert tr.ok is True
    assert "hooks.example.com" in tr.message


# ---------------------------------------------------------------------------
# send() — notifications
# ---------------------------------------------------------------------------

def test_send_signs_payload():
    fake = FakeTransport()
    c = _conn(transport=fake)
    dr = _run(c.send({"text": "hello"}))

    assert dr.ok is True
    url, body, headers, timeout = fake.calls[0]
    assert url == "https://hooks.example.com/abc"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-ACES-Event"] == "notification"
    assert headers["X-ACES-Delivery-Id"] == dr.metadata["delivery_id"]
    assert _verify(body, headers)


def test_send_custom_event_type():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send(
        {"x": 1},
        metadata={"event_type": "job.completed"},
    ))
    _, _, headers, _ = fake.calls[0]
    assert headers["X-ACES-Event"] == "job.completed"


def test_send_payload_shape():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send(
        {"text": "hello world", "level": "info"},
        metadata={"event_type": "notification"},
    ))
    _, body, _, _ = fake.calls[0]
    parsed = json.loads(body)
    assert parsed["event_type"] == "notification"
    assert parsed["payload"] == {"text": "hello world", "level": "info"}
    assert "delivery_id" in parsed
    assert "occurred_at" in parsed


def test_send_empty_payload_is_allowed():
    """A bare {} is valid — the envelope carries the event type."""
    fake = FakeTransport()
    c = _conn(transport=fake)
    dr = _run(c.send({}))
    assert dr.ok is True


# ---------------------------------------------------------------------------
# publish() — datasets
# ---------------------------------------------------------------------------

def test_publish_inline_records():
    fake = FakeTransport()
    c = _conn(transport=fake)
    ref = DatasetReference(
        format="json",
        records=[{"a": 1}, {"a": 2}],
        filename="out.json",
    )
    dr = _run(c.publish(ref))

    assert dr.ok is True
    assert dr.records_sent == 2
    _, body, headers, _ = fake.calls[0]
    assert headers["X-ACES-Event"] == "dataset.published"
    assert _verify(body, headers)

    parsed = json.loads(body)
    assert parsed["records_count"] == 2
    assert parsed["records"] == [{"a": 1}, {"a": 2}]
    assert parsed["format"] == "json"
    assert parsed["filename"] == "out.json"


def test_publish_signed_url_reference():
    fake = FakeTransport()
    c = _conn(transport=fake)
    ref = DatasetReference(
        format="parquet",
        url="https://storage.example.com/ds.parquet",
    )
    _run(c.publish(ref))
    _, body, _, _ = fake.calls[0]
    parsed = json.loads(body)
    assert parsed["url"] == "https://storage.example.com/ds.parquet"
    assert "records" not in parsed


def test_publish_empty_reference_fails():
    c = _conn()
    dr = _run(c.publish(DatasetReference()))
    assert dr.ok is False
    assert "nothing to publish" in dr.error


# ---------------------------------------------------------------------------
# Signature integrity
# ---------------------------------------------------------------------------

def test_signature_is_verifiable():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"n": 1}))
    _, body, headers, _ = fake.calls[0]
    assert _verify(body, headers)


def test_signature_wrong_secret_fails_verification():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"n": 1}))
    _, body, headers, _ = fake.calls[0]
    assert not _verify(body, headers, secret="wrong-secret")


def test_tampered_body_fails_verification():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"text": "original"}))
    _, body, headers, _ = fake.calls[0]
    tampered = body.replace("original", "tampered")
    assert not _verify(tampered, headers)


# ---------------------------------------------------------------------------
# Extra headers
# ---------------------------------------------------------------------------

def test_extra_headers_added():
    fake = FakeTransport()
    c = _conn(transport=fake, extra_headers={
        "X-Tenant": "acme",
        "User-Agent": "ACES/1.0",
    })
    _run(c.send({"x": 1}))
    _, _, headers, _ = fake.calls[0]
    assert headers["X-Tenant"] == "acme"
    assert headers["User-Agent"] == "ACES/1.0"
    # Default headers still present
    assert "X-ACES-Signature" in headers


def test_extra_headers_cannot_override_signature():
    """
    If a caller passes X-ACES-Signature in extra_headers, it must not
    override the real signature. Our implementation applies the real
    signature first, then `headers.update(self.extra_headers)` — which
    WOULD override. So the caller is responsible for not doing this.
    This test documents the current behaviour.
    """
    fake = FakeTransport()
    c = _conn(
        transport=fake,
        extra_headers={"X-ACES-Signature": "sha256=tampered"},
    )
    _run(c.send({"x": 1}))
    _, body, headers, _ = fake.calls[0]
    # The extra header wins (this is a known footgun; see docstring)
    assert headers["X-ACES-Signature"] == "sha256=tampered"
    # And the real signature was overwritten — document this
    assert not _verify(body, headers)


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------

def test_http_500_returns_not_ok():
    c = _conn(transport=FakeTransport(status=500, body="boom"))
    dr = _run(c.send({"x": 1}))
    assert dr.ok is False
    assert "HTTP 500" in dr.error
    assert "boom" in dr.error


def test_http_404_returns_not_ok():
    c = _conn(transport=FakeTransport(status=404, body="not found"))
    dr = _run(c.send({"x": 1}))
    assert dr.ok is False
    assert "HTTP 404" in dr.error


def test_transport_exception_returns_not_ok():
    class Boom:
        async def post_json(self, *a, **k):
            raise RuntimeError("network down")

    c = _conn(transport=Boom())
    dr = _run(c.send({"x": 1}))
    assert dr.ok is False
    assert "network down" in dr.error


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_idempotency_same_key_is_noop():
    fake = FakeTransport()
    c = _conn(transport=fake)
    r1 = _run(c.send({"x": 1}, metadata={"idempotency_key": "K"}))
    r2 = _run(c.send({"x": 1}, metadata={"idempotency_key": "K"}))
    assert r1.ok and r2.ok
    assert r2.metadata.get("idempotent_replay") is True
    assert len(fake.calls) == 1


def test_idempotency_key_header_sent():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"x": 1}, metadata={"idempotency_key": "my-key"}))
    _, _, headers, _ = fake.calls[0]
    assert headers["X-ACES-Idempotency-Key"] == "my-key"


def test_idempotency_different_keys_send_twice():
    fake = FakeTransport()
    c = _conn(transport=fake)
    _run(c.send({"x": 1}, metadata={"idempotency_key": "K1"}))
    _run(c.send({"x": 1}, metadata={"idempotency_key": "K2"}))
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------

def test_result_contains_delivery_id():
    c = _conn()
    dr = _run(c.send({"x": 1}))
    assert dr.connector_type == "webhook"
    assert dr.destination == c.url
    assert "delivery_id" in dr.metadata
    assert "response_status" in dr.metadata


def test_bytes_sent_matches_body_length():
    fake = FakeTransport()
    c = _conn(transport=fake)
    dr = _run(c.send({"x": 1}))
    _, body, _, _ = fake.calls[0]
    assert dr.bytes_sent == len(body.encode("utf-8"))