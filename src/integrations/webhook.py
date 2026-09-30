"""
WebhookConnector — spec §47A, §46.

POSTs a signed JSON body to any HTTPS endpoint. Every request carries
an HMAC-SHA256 signature header so the receiver can verify authenticity
before trusting the payload.

Body shapes:

    publish(ref)  -> dataset delivery envelope
        {
          "event_type": "dataset.published",
          "delivery_id": "...",
          "occurred_at": "...",
          "format": "csv",
          "records_count": 100,
          "records": [...],          # when ref.records is populated
          "url": "https://...",      # when ref.url is populated
          "filename": "...",
          "metadata": {...}
        }

    send(payload) -> notification envelope
        {
          "event_type": "<metadata.event_type or 'notification'>",
          "delivery_id": "...",
          "occurred_at": "...",
          "payload": {...caller's dict...},
          "metadata": {...}
        }

Signature header:  X-ACES-Signature: sha256=<hex>
Event header:      X-ACES-Event: <event_type>
Delivery header:   X-ACES-Delivery-Id: <uuid>

Safety:
    - HTTPS required by default (http allowed only when allow_http=True,
      which is intended for local testing).
    - URL is SSRF-validated via the same guard used elsewhere in ACES.
    - The request body is signed AFTER serialization, so any change to
      the payload invalidates the signature.
    - Transport is injectable for tests.
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
from src.output.receipt import canonical_json
from src.security.ssrf import SSRFGuard


logger = logging.getLogger("integrations.webhook")

# ---------------------------------------------------------------------------
# SSRF helpers
# ---------------------------------------------------------------------------

def _is_transient_dns_failure(reason: str) -> bool:
    """
    True when the SSRF guard rejected a URL because DNS could not
    resolve the hostname *right now*, rather than because the URL is
    structurally dangerous.

    Transient DNS failures must not block construction — a hostname
    that resolves fine today should not fail to register as a webhook
    just because the network was momentarily down when the process
    started. The send path re-validates anyway, which is what §50.2 /
    §50.3 require for DNS-rebinding protection.

    Hard failures (metadata IP literal, private IP literal, bad scheme,
    userinfo, forbidden suffix) do NOT match this predicate and are
    still rejected at construction.
    """
    return bool(reason) and reason.startswith("DNS resolution failed")


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

class HttpTransport(Protocol):
    async def post_json(
        self, url: str, body: str, headers: dict, timeout: float,
    ) -> tuple[int, str]: ...


class HttpxTransport:
    async def post_json(
        self, url: str, body: str, headers: dict, timeout: float,
    ) -> tuple[int, str]:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                url,
                content=body.encode("utf-8"),
                headers=headers,
            )
            return r.status_code, r.text or ""


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class WebhookConnector(BaseConnector):
    connector_type = ConnectorType.WEBHOOK

    def __init__(
        self,
        url: str,
        secret: str,
        extra_headers: Optional[dict] = None,
        timeout_seconds: int = 30,
        allow_http: bool = False,
        connector_id: Optional[str] = None,
        transport: Optional[HttpTransport] = None,
        ssrf_guard: Optional[SSRFGuard] = None,
    ):
        if not url:
            raise ConnectorError("webhook url is required")
        if not secret:
            raise ConnectorError("webhook secret is required")

        url_lower = url.lower()
        if url_lower.startswith("http://"):
            if not allow_http:
                raise ConnectorError(
                    "webhook url must be HTTPS (pass allow_http=True "
                    "only for local testing)"
                )
        elif not url_lower.startswith("https://"):
            raise ConnectorError(
                f"webhook url must be http(s), got {url!r}"
            )

        # --- SSRF gate, construction time ---
        # Only *structural* checks run here — those that can be
        # decided without a DNS lookup. Transient DNS failures are
        # deferred to the send path, which re-validates on every POST.
        guard = ssrf_guard or SSRFGuard()
        check = guard.validate_url(url)
        if not check.allowed and not _is_transient_dns_failure(check.reason):
            raise ConnectorError(
                f"webhook url rejected by SSRF guard: {check.reason}"
            )

        self.url = url
        self.secret = secret.encode("utf-8")
        self.extra_headers = dict(extra_headers or {})
        self.timeout_seconds = int(timeout_seconds)
        self._transport = transport or HttpxTransport()
        self._delivered_keys: dict[str, str] = {}
        # Store the guard so send-time can re-validate against the
        # same policy (including any test-injected permissive guard).
        self._ssrf_guard = guard

        super().__init__(
            connector_id=connector_id,
            capabilities={
                ConnectorCapability.DELIVERS_ROWS,
                ConnectorCapability.DELIVERS_NOTIFICATIONS,
                ConnectorCapability.SUPPORTS_SIGNED_URLS,
                ConnectorCapability.SUPPORTS_IDEMPOTENCY,
            },
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    async def _do_test(self) -> ConnectorTestResult:
        # We can't verify the receiver accepts payloads without sending
        # one, so a structural check is what's honest here: the URL is
        # HTTPS, the secret is set, and the transport is instantiated.
        # The real check happens on first publish/send.
        return ConnectorTestResult(
            ok=True,
            message=f"webhook configured for {self.url}",
        )

    async def _do_publish(
        self,
        ref: DatasetReference,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        metadata = dict(metadata or {})

        # ---- Idempotency ----
        idem_key = metadata.get("idempotency_key") or ""
        if idem_key and idem_key in self._delivered_keys:
            existing = self._delivered_keys[idem_key]
            return DeliveryResult(
                ok=True,
                destination=self.url,
                metadata={"idempotent_replay": True, "delivery_id": existing},
            )

        if ref.mode == "empty":
            return DeliveryResult(
                ok=False,
                destination=self.url,
                error="nothing to publish (empty reference)",
            )

        # ---- Build envelope ----
        import uuid
        from datetime import datetime, timezone
        delivery_id = str(uuid.uuid4())
        body: dict = {
            "event_type": "dataset.published",
            "delivery_id": delivery_id,
            "occurred_at": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "format": ref.format or "json",
            "records_count": len(ref.records or []),
            "filename": ref.filename or "",
            "content_type": ref.content_type or "",
            "metadata": metadata,
        }
        if ref.url:
            body["url"] = ref.url
        else:
            body["records"] = list(ref.records or [])

        return await self._post_signed(
            body=body,
            event_type="dataset.published",
            delivery_id=delivery_id,
            records_sent=body["records_count"],
            idem_key=idem_key,
        )

    async def _do_send(
        self,
        payload: dict,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        metadata = dict(metadata or {})
        event_type = str(metadata.get("event_type") or "notification")

        idem_key = metadata.get("idempotency_key") or ""
        if idem_key and idem_key in self._delivered_keys:
            existing = self._delivered_keys[idem_key]
            return DeliveryResult(
                ok=True,
                destination=self.url,
                metadata={"idempotent_replay": True, "delivery_id": existing},
            )

        import uuid
        from datetime import datetime, timezone
        delivery_id = str(uuid.uuid4())
        body = {
            "event_type": event_type,
            "delivery_id": delivery_id,
            "occurred_at": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "payload": dict(payload or {}),
            "metadata": metadata,
        }

        return await self._post_signed(
            body=body,
            event_type=event_type,
            delivery_id=delivery_id,
            records_sent=0,
            idem_key=idem_key,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _post_signed(
        self,
        body: dict,
        event_type: str,
        delivery_id: str,
        records_sent: int,
        idem_key: str,
    ) -> DeliveryResult:
        # --- SSRF re-validation on every send (§50.2 / §50.3) ---
        # A hostname can rebind between calls; a redirect chain can
        # point somewhere new. Re-check the target before every POST.
        # Only hard failures abort — transient DNS is left to the
        # HTTP client, whose error is more informative.
        check = self._ssrf_guard.validate_url(self.url)
        if not check.allowed and not _is_transient_dns_failure(check.reason):
            return DeliveryResult(
                ok=False,
                destination=self.url,
                error=f"SSRF re-check failed: {check.reason}",
                attempts=1,
            )

        serialized = canonical_json(body)
        signature = self._sign(serialized)

        headers = {
            "Content-Type": "application/json",
            "X-ACES-Signature": signature,
            "X-ACES-Event": event_type,
            "X-ACES-Delivery-Id": delivery_id,
        }
        if idem_key:
            headers["X-ACES-Idempotency-Key"] = idem_key
        headers.update(self.extra_headers)

        try:
            status, text = await self._transport.post_json(
                self.url, serialized, headers, self.timeout_seconds,
            )
        except Exception as e:
            return DeliveryResult(
                ok=False,
                destination=self.url,
                error=f"transport error: {type(e).__name__}: {e}",
                attempts=1,
                metadata={"delivery_id": delivery_id},
            )

        if not (200 <= status < 300):
            return DeliveryResult(
                ok=False,
                destination=self.url,
                error=f"HTTP {status}: {text[:200]}",
                attempts=1,
                metadata={"delivery_id": delivery_id, "response_status": status},
            )

        if idem_key:
            self._delivered_keys[idem_key] = delivery_id

        return DeliveryResult(
            ok=True,
            destination=self.url,
            bytes_sent=len(serialized.encode("utf-8")),
            records_sent=records_sent,
            attempts=1,
            metadata={
                "delivery_id": delivery_id,
                "response_status": status,
                "response_body": text[:200],
            },
        )

    def _sign(self, body_str: str) -> str:
        """
        HMAC-SHA256 over the serialized body. Uses the same
        "sha256=<hex>" format as src/api/webhook.py so receivers can
        reuse verification code.
        """
        import hashlib
        import hmac
        mac = hmac.new(
            self.secret, body_str.encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        return f"sha256={mac}"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import hashlib
    import hmac

    class FakeTransport:
        def __init__(self, status=200, body="ok"):
            self.status = status
            self.body = body
            self.calls: list[tuple[str, str, dict, float]] = []

        async def post_json(self, url, body, headers, timeout):
            self.calls.append((url, body, headers, timeout))
            return self.status, self.body

    async def run():
        SECRET = b"test-secret-1234567890" * 2
        SECRET_STR = SECRET.decode("utf-8")

        # A permissive guard for tests — skips DNS resolution.
        class _AllowAllGuard:
            def validate_url(self, url):
                from src.security.ssrf import SSRFCheckResult
                return SSRFCheckResult(True, "test-bypass")

        # ---- Construction ----
        fake = FakeTransport()
        c = WebhookConnector(
            url="https://hooks.example.com/abc",
            secret=SECRET_STR,
            transport=fake,
            ssrf_guard=_AllowAllGuard(),
        )
        assert c.connector_type == ConnectorType.WEBHOOK
        assert c.supports(ConnectorCapability.DELIVERS_NOTIFICATIONS)
        assert c.supports(ConnectorCapability.DELIVERS_ROWS)
        assert c.supports(ConnectorCapability.SUPPORTS_SIGNED_URLS)
        assert c.supports(ConnectorCapability.SUPPORTS_IDEMPOTENCY)

        # ---- Required args ----
        try:
            WebhookConnector(url="", secret="x")
            raise AssertionError("expected ConnectorError")
        except ConnectorError:
            pass
        try:
            WebhookConnector(url="https://x.com", secret="")
            raise AssertionError("expected ConnectorError")
        except ConnectorError:
            pass

        # ---- HTTPS enforcement ----
        try:
            WebhookConnector(url="http://x.com", secret="s")
            raise AssertionError("expected ConnectorError")
        except ConnectorError as e:
            assert "HTTPS" in str(e)


        # ---- SSRF — real guard blocks a metadata IP without DNS ----
        try:
            WebhookConnector(
                url="https://169.254.169.254/hook",
                secret="s",
            )
            raise AssertionError("expected ConnectorError for metadata IP")
        except ConnectorError as e:
            assert "SSRF" in str(e) or "metadata" in str(e).lower()

        # ---- test_connection ----
        tr = await c.test_connection()
        assert tr.ok is True
        assert "hooks.example.com" in tr.message

        # ---- send() notification ----
        fake.calls.clear()
        dr = await c.send({"text": "hello world"})
        assert dr.ok is True
        assert dr.records_sent == 0
        assert dr.destination == "https://hooks.example.com/abc"

        url, body, headers, timeout = fake.calls[0]
        assert url == "https://hooks.example.com/abc"
        assert headers["Content-Type"] == "application/json"
        assert headers["X-ACES-Event"] == "notification"
        assert headers["X-ACES-Signature"].startswith("sha256=")

        # ---- Signature verification (receiver side) ----
        received_sig = headers["X-ACES-Signature"][len("sha256="):]
        expected = hmac.new(
            SECRET, body.encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        assert received_sig == expected, "signature mismatch"

        # ---- Tamper detection ----
        tampered = body.replace("hello world", "goodbye world")
        tampered_sig = hmac.new(
            SECRET, tampered.encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        assert tampered_sig != received_sig

        # ---- Custom event_type via metadata ----
        fake.calls.clear()
        await c.send(
            {"msg": "x"},
            metadata={"event_type": "job.completed"},
        )
        _, _, headers2, _ = fake.calls[0]
        assert headers2["X-ACES-Event"] == "job.completed"

        # ---- publish() with inline records ----
        fake.calls.clear()
        ref = DatasetReference(
            format="json",
            records=[{"a": 1}, {"a": 2}, {"a": 3}],
            filename="out.json",
        )
        dr = await c.publish(ref)
        assert dr.ok is True
        assert dr.records_sent == 3
        _, body3, headers3, _ = fake.calls[0]
        assert headers3["X-ACES-Event"] == "dataset.published"
        parsed = __import__("json").loads(body3)
        assert parsed["records_count"] == 3
        assert parsed["records"] == [{"a": 1}, {"a": 2}, {"a": 3}]
        assert parsed["format"] == "json"
        assert parsed["filename"] == "out.json"

        # ---- publish() with signed URL ----
        fake.calls.clear()
        ref_url = DatasetReference(
            format="parquet",
            url="https://storage.example.com/dataset.parquet",
        )
        dr = await c.publish(ref_url)
        assert dr.ok is True
        _, body4, _, _ = fake.calls[0]
        parsed4 = __import__("json").loads(body4)
        assert parsed4["url"] == "https://storage.example.com/dataset.parquet"
        assert "records" not in parsed4

        # ---- HTTP error becomes ok=False ----
        fail = FakeTransport(status=500, body="boom")
        c_fail = WebhookConnector(
            url="https://hooks.example.com/fail",
            secret=SECRET_STR,
            transport=fail,
            ssrf_guard=_AllowAllGuard(),
        )
        dr = await c_fail.send({"x": 1})
        assert dr.ok is False
        assert "HTTP 500" in dr.error

        # ---- Transport exception becomes ok=False ----
        class Boom:
            async def post_json(self, *a, **k):
                raise RuntimeError("network down")
        c_boom = WebhookConnector(
            url="https://hooks.example.com/boom",
            secret=SECRET_STR,
            transport=Boom(),
            ssrf_guard=_AllowAllGuard(),
        )
        dr = await c_boom.send({"x": 1})
        assert dr.ok is False
        assert "network down" in dr.error

        # ---- Idempotency ----
        fake2 = FakeTransport()
        c2 = WebhookConnector(
            url="https://hooks.example.com/idem",
            secret=SECRET_STR,
            transport=fake2,
            ssrf_guard=_AllowAllGuard(),
        )
        r1 = await c2.send({"x": 1}, metadata={"idempotency_key": "K"})
        r2 = await c2.send({"x": 1}, metadata={"idempotency_key": "K"})
        assert r1.ok and r2.ok
        assert r2.metadata.get("idempotent_replay") is True
        # Only one HTTP call reached the transport
        assert len(fake2.calls) == 1

        # ---- Empty reference ----
        dr = await c.publish(DatasetReference())
        assert dr.ok is False
        assert "nothing to publish" in dr.error

        print("WebhookConnector OK.")

    asyncio.run(run())