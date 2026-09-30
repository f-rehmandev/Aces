"""Unit tests for webhooks (spec §46)."""
from datetime import datetime, timezone

import pytest

from src.api.webhook import (
    WebhookEvent, WebhookPayload, build_payload, sign_payload,
    verify_signature, WebhookDispatcher, DeliveryRecord,
    MAX_DELIVERY_ATTEMPTS,
)


SECRET = b"whsecret"


# --- payload ----------------------------------------------------------

def test_build_payload_shape():
    p = build_payload(
        WebhookEvent.JOB_COMPLETED, job_id="j", task_id="t",
        client_id="acme", summary={"records": 42},
    )
    assert p.event_type == "job.completed"
    assert p.job_id == "j"
    assert p.summary["records"] == 42
    assert p.event_id
    assert p.idempotency_key


def test_build_payload_accepts_string_event():
    p = build_payload("custom.event", "j", "t", "c")
    assert p.event_type == "custom.event"


def test_canonical_json_is_sorted():
    p = build_payload(WebhookEvent.JOB_COMPLETED, "j", "t", "c",
                      summary={"b": 2, "a": 1})
    js = p.canonical_json()
    assert js.index('"a"') < js.index('"b"')


# --- signing ----------------------------------------------------------

def test_sign_payload_has_algorithm_prefix():
    p = build_payload(WebhookEvent.JOB_QUEUED, "j", "t", "c")
    sig = sign_payload(p, SECRET)
    assert sig.startswith("sha256=")


def test_verify_signature_valid():
    p = build_payload(WebhookEvent.JOB_QUEUED, "j", "t", "c")
    sig = sign_payload(p, SECRET)
    assert verify_signature(p, sig, SECRET)


def test_verify_signature_wrong_secret():
    p = build_payload(WebhookEvent.JOB_QUEUED, "j", "t", "c")
    sig = sign_payload(p, SECRET)
    assert not verify_signature(p, sig, b"other")


def test_tampered_payload_fails_verification():
    p = build_payload(WebhookEvent.JOB_QUEUED, "j", "t", "c", summary={"n": 1})
    sig = sign_payload(p, SECRET)
    p.summary["n"] = 999
    assert not verify_signature(p, sig, SECRET)


def test_empty_secret_rejected():
    p = build_payload(WebhookEvent.JOB_QUEUED, "j", "t", "c")
    with pytest.raises(ValueError):
        sign_payload(p, b"")


# --- dispatcher -------------------------------------------------------

def _ok(url, body, headers):
    return 200, ""

def _fail(url, body, headers):
    return 500, "server error"


def test_dispatcher_requires_secret():
    with pytest.raises(ValueError):
        WebhookDispatcher(b"")


def test_successful_delivery_marks_delivered():
    d = WebhookDispatcher(SECRET, send_fn=_ok)
    p = build_payload(WebhookEvent.JOB_COMPLETED, "j", "t", "c")
    rec = d.dispatch("https://x.com/hook", p, client_id="acme")
    assert rec.status == "delivered"
    assert rec.attempt_count == 1
    assert rec.attempts[0].status_code == 200


def test_failed_delivery_stays_pending():
    d = WebhookDispatcher(SECRET, send_fn=_fail)
    p = build_payload(WebhookEvent.JOB_COMPLETED, "j", "t", "c")
    rec = d.dispatch("https://x.com/hook", p)
    assert rec.status == "pending"


def test_retry_exhausts_and_marks_failed():
    d = WebhookDispatcher(SECRET, send_fn=_fail)
    p = build_payload(WebhookEvent.JOB_COMPLETED, "j", "t", "c")
    rec = d.dispatch("https://x.com/hook", p)
    for _ in range(MAX_DELIVERY_ATTEMPTS - 1):
        rec = d.retry(rec.delivery_id)
    assert rec.status == "failed"
    assert d.retry(rec.delivery_id) is None


def test_next_retry_at_is_future():
    d = WebhookDispatcher(SECRET, send_fn=_fail)
    p = build_payload(WebhookEvent.JOB_COMPLETED, "j", "t", "c")
    rec = d.dispatch("https://x.com/hook", p)
    nxt = rec.next_retry_at()
    assert nxt is not None
    assert nxt > datetime.now(timezone.utc)


def test_delivered_record_has_no_next_retry():
    d = WebhookDispatcher(SECRET, send_fn=_ok)
    p = build_payload(WebhookEvent.JOB_COMPLETED, "j", "t", "c")
    rec = d.dispatch("https://x.com/hook", p)
    assert rec.next_retry_at() is None


def test_exception_in_send_treated_as_failure():
    def bad_send(url, body, headers):
        raise RuntimeError("network down")
    d = WebhookDispatcher(SECRET, send_fn=bad_send)
    p = build_payload(WebhookEvent.JOB_COMPLETED, "j", "t", "c")
    rec = d.dispatch("https://x.com/hook", p)
    assert rec.status == "pending"
    assert "network down" in rec.attempts[0].error


def test_all_returns_deliveries():
    d = WebhookDispatcher(SECRET, send_fn=_ok)
    for i in range(3):
        p = build_payload(WebhookEvent.JOB_COMPLETED, f"j{i}", "t", "c")
        d.dispatch("https://x.com/hook", p)
    assert len(d.all()) == 3