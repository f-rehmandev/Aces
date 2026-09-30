"""Unit tests for execution receipts (spec §34)."""
import hashlib
import hmac
import pytest

from src.output.receipt import (
    ReceiptSigner, ReceiptBuilder, ExecutionReceipt,
    dataset_sha256, audit_log_sha256, canonical_json,
    verify_receipt, VerificationResult,
)


RECORDS = [
    {"title": "A", "price": "$10"},
    {"title": "B", "price": "$20"},
]
SECRET = b"test-secret-key"


# --- canonical hashing -------------------------------------------------

def test_canonical_json_is_stable_regardless_of_key_order():
    a = canonical_json({"b": 2, "a": 1})
    b = canonical_json({"a": 1, "b": 2})
    assert a == b


def test_dataset_sha256_is_stable():
    h1 = dataset_sha256(RECORDS)
    h2 = dataset_sha256(RECORDS)
    assert h1 == h2 and len(h1) == 64


def test_dataset_sha256_changes_with_content():
    assert dataset_sha256(RECORDS) != dataset_sha256(RECORDS + [{"title": "X"}])


def test_dataset_sha256_key_order_independent():
    a = dataset_sha256([{"x": 1, "y": 2}])
    b = dataset_sha256([{"y": 2, "x": 1}])
    assert a == b


def test_audit_log_sha256():
    assert audit_log_sha256([]) == dataset_sha256([])
    assert audit_log_sha256([{"e": 1}]) == dataset_sha256([{"e": 1}])


# --- signer ------------------------------------------------------------

def test_signer_requires_non_empty_secret():
    with pytest.raises(ValueError):
        ReceiptSigner(b"")


def _make_signed_receipt():
    signer = ReceiptSigner(SECRET, key_version=1)
    builder = ReceiptBuilder("j", "t", "acme", signer=signer)
    receipt, sig = builder.build(
        started_at="2026-09-24T00:00:00+00:00",
        completed_at="2026-09-24T00:00:10+00:00",
        records=RECORDS,
        pages_attempted=2, pages_succeeded=2,
    )
    return signer, receipt, sig


def test_sign_returns_hex():
    _, _, sig = _make_signed_receipt()
    assert isinstance(sig, str)
    assert len(sig) == 64
    int(sig, 16)  # valid hex


def test_verify_valid_signature():
    signer, receipt, sig = _make_signed_receipt()
    assert signer.verify(receipt, sig) is True


def test_verify_rejects_tampered_field():
    signer, receipt, sig = _make_signed_receipt()
    receipt.records_extracted = 999
    assert signer.verify(receipt, sig) is False


def test_verify_rejects_wrong_secret():
    signer, receipt, sig = _make_signed_receipt()
    other = ReceiptSigner(b"different-secret")
    assert other.verify(receipt, sig) is False


def test_verify_rejects_empty_signature():
    signer, receipt, _ = _make_signed_receipt()
    assert signer.verify(receipt, "") is False


def test_sign_stamps_key_version():
    signer = ReceiptSigner(SECRET, key_version=7)
    builder = ReceiptBuilder("j", "t", "acme", signer=signer)
    receipt, _ = builder.build(
        started_at="2026-09-24T00:00:00+00:00",
        completed_at="2026-09-24T00:00:01+00:00",
        records=RECORDS,
    )
    assert receipt.key_version == 7


# --- builder -----------------------------------------------------------

def test_builder_computes_duration():
    signer = ReceiptSigner(SECRET)
    builder = ReceiptBuilder("j", "t", "c", signer=signer)
    receipt, _ = builder.build(
        started_at="2026-09-24T00:00:00+00:00",
        completed_at="2026-09-24T00:00:42+00:00",
        records=RECORDS,
    )
    assert receipt.duration_seconds == 42.0


def test_builder_without_signer_returns_no_signature():
    builder = ReceiptBuilder("j", "t", "c", signer=None)
    receipt, sig = builder.build(
        started_at="2026-09-24T00:00:00+00:00",
        completed_at="2026-09-24T00:00:05+00:00",
        records=RECORDS,
    )
    assert sig is None
    assert receipt.records_extracted == 2


def test_builder_records_all_inputs():
    signer = ReceiptSigner(SECRET)
    builder = ReceiptBuilder("job-1", "task-1", "acme", signer=signer)
    receipt, _ = builder.build(
        started_at="2026-09-24T00:00:00+00:00",
        completed_at="2026-09-24T00:00:10+00:00",
        records=RECORDS,
        pages_attempted=5,
        pages_succeeded=4,
        pages_failed=1,
        sources=[{"url": "https://x.com/a"}],
        quality_score=0.95,
        strategy_ids=["s1", "s2"],
        model_calls={"gemini": 2},
        audit_log=[{"e": "started"}],
    )
    assert receipt.pages_attempted == 5
    assert receipt.pages_failed == 1
    assert receipt.quality_score == 0.95
    assert receipt.strategy_ids == ["s1", "s2"]
    assert receipt.model_calls == {"gemini": 2}
    assert receipt.sources == [{"url": "https://x.com/a"}]


def test_receipt_to_json_is_canonical():
    signer = ReceiptSigner(SECRET)
    builder = ReceiptBuilder("j", "t", "c", signer=signer)
    r, _ = builder.build(
        started_at="2026-09-24T00:00:00+00:00",
        completed_at="2026-09-24T00:00:01+00:00",
        records=RECORDS,
    )
    js1 = r.to_json()
    js2 = r.to_json()
    assert js1 == js2
    # Sorted keys
    assert '"client_id"' in js1


# --- verify_receipt (public API) ---------------------------------------

def test_verify_receipt_with_matching_dataset():
    signer = ReceiptSigner(SECRET)
    builder = ReceiptBuilder("j", "t", "c", signer=signer)
    receipt, sig = builder.build(
        started_at="2026-09-24T00:00:00+00:00",
        completed_at="2026-09-24T00:00:01+00:00",
        records=RECORDS,
    )
    v = verify_receipt(receipt, sig, signer, delivered_records=RECORDS)
    assert isinstance(v, VerificationResult)
    assert v.ok
    assert v.valid_signature and v.dataset_hash_matches


def test_verify_receipt_detects_tampered_dataset():
    signer = ReceiptSigner(SECRET)
    builder = ReceiptBuilder("j", "t", "c", signer=signer)
    receipt, sig = builder.build(
        started_at="2026-09-24T00:00:00+00:00",
        completed_at="2026-09-24T00:00:01+00:00",
        records=RECORDS,
    )
    v = verify_receipt(receipt, sig, signer,
                       delivered_records=RECORDS + [{"title": "X"}])
    assert not v.ok
    assert "hash mismatch" in v.reason


def test_verify_receipt_detects_tampered_receipt():
    signer = ReceiptSigner(SECRET)
    builder = ReceiptBuilder("j", "t", "c", signer=signer)
    receipt, sig = builder.build(
        started_at="2026-09-24T00:00:00+00:00",
        completed_at="2026-09-24T00:00:01+00:00",
        records=RECORDS,
    )
    receipt.quality_score = 0.99  # tampered
    v = verify_receipt(receipt, sig, signer)
    assert not v.valid_signature
    assert not v.ok


def test_verification_result_to_dict():
    v = VerificationResult(valid_signature=True, dataset_hash_matches=True)
    d = v.to_dict()
    assert d["ok"] is True
    assert "reason" in d