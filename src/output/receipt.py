"""
Execution receipts — spec §34.

A verifiable artifact proving the dataset was actually produced by the
job that claims to have produced it, at the time it claims, from the
sources it claims.

Contains (§34.1):
    - job / task / client IDs
    - start and completion timestamps
    - pages attempted / succeeded / failed
    - records extracted
    - sources with fetch timestamps
    - quality score
    - strategy IDs
    - model call counts (not content)
    - dataset SHA-256 (canonical serialization)
    - audit log SHA-256
    - HMAC-SHA256 signature over all of the above, keyed by a per-client secret
      with a key version included for rotation (§34.2)
"""

from __future__ import annotations
import hashlib
import hmac
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """
    Deterministic JSON: sorted keys, no extra whitespace, UTF-8 safe.
    Used everywhere we need a stable hash.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def dataset_sha256(records: list[dict]) -> str:
    """SHA-256 over the canonical serialization of a list of records."""
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def audit_log_sha256(entries: list[dict]) -> str:
    """SHA-256 over the canonical serialization of audit log entries."""
    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

@dataclass
class ExecutionReceipt:
    receipt_id: str
    job_id: str
    task_id: str
    client_id: str
    started_at: str
    completed_at: str
    duration_seconds: float
    pages_attempted: int
    pages_succeeded: int
    pages_failed: int
    records_extracted: int
    sources: list[dict] = field(default_factory=list)
    quality_score: float = 0.0
    strategy_ids: list[str] = field(default_factory=list)
    model_calls: dict = field(default_factory=dict)
    dataset_sha256: str = ""
    audit_log_sha256: str = ""
    key_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


# ---------------------------------------------------------------------------
# Signer
# ---------------------------------------------------------------------------

class ReceiptSigner:
    """
    Signs and verifies receipts using HMAC-SHA256 with a per-client secret
    and a key version for rotation.

    In production the secret would come from a secret manager, and key
    versions would be tracked so old receipts remain verifiable after
    rotation. That plumbing is deferred — the signer accepts a key + version.
    """

    def __init__(self, secret: bytes, key_version: int = 1):
        if not secret:
            raise ValueError("signing secret must be non-empty")
        self._secret = secret
        self.key_version = int(key_version)

    def _payload_bytes(self, receipt: ExecutionReceipt) -> bytes:
        # Sign over the receipt's canonical JSON (which already contains
        # key_version), so tampering with any field invalidates the sig.
        return canonical_json(receipt.to_dict()).encode("utf-8")

    def sign(self, receipt: ExecutionReceipt) -> str:
        """Return the hex signature. Also stamps key_version on the receipt."""
        receipt.key_version = self.key_version
        mac = hmac.new(self._secret, self._payload_bytes(receipt), hashlib.sha256)
        return mac.hexdigest()

    def verify(self, receipt: ExecutionReceipt, signature: str) -> bool:
        """Constant-time signature verification."""
        if not signature:
            return False
        expected = hmac.new(
            self._secret, self._payload_bytes(receipt), hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class ReceiptBuilder:
    """
    Collect fields during a run and emit a signed receipt.
    """

    def __init__(
        self,
        job_id: str,
        task_id: str,
        client_id: str,
        signer: Optional[ReceiptSigner] = None,
    ):
        self.job_id = job_id
        self.task_id = task_id
        self.client_id = client_id
        self.signer = signer

    def build(
        self,
        started_at: str,
        completed_at: str,
        records: list[dict],
        pages_attempted: int = 0,
        pages_succeeded: int = 0,
        pages_failed: int = 0,
        sources: Optional[list[dict]] = None,
        quality_score: float = 0.0,
        strategy_ids: Optional[list[str]] = None,
        model_calls: Optional[dict] = None,
        audit_log: Optional[list[dict]] = None,
    ) -> tuple[ExecutionReceipt, Optional[str]]:
        """
        Returns (receipt, signature_or_None). Signature is None when no
        signer is configured.
        """
        import uuid
        start_dt = _parse_iso(started_at)
        end_dt = _parse_iso(completed_at)
        duration = (end_dt - start_dt).total_seconds() if start_dt and end_dt else 0.0

        receipt = ExecutionReceipt(
            receipt_id=str(uuid.uuid4()),
            job_id=self.job_id,
            task_id=self.task_id,
            client_id=self.client_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=round(duration, 3),
            pages_attempted=pages_attempted,
            pages_succeeded=pages_succeeded,
            pages_failed=pages_failed,
            records_extracted=len(records),
            sources=sources or [],
            quality_score=quality_score,
            strategy_ids=strategy_ids or [],
            model_calls=model_calls or {},
            dataset_sha256=dataset_sha256(records),
            audit_log_sha256=audit_log_sha256(audit_log or []),
            key_version=1,
        )

        signature = self.signer.sign(receipt) if self.signer else None
        return receipt, signature


def _parse_iso(s: str):
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Verifier (public-facing)
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    valid_signature: bool
    dataset_hash_matches: bool
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.valid_signature and self.dataset_hash_matches

    def to_dict(self) -> dict:
        return {
            "valid_signature": self.valid_signature,
            "dataset_hash_matches": self.dataset_hash_matches,
            "ok": self.ok,
            "reason": self.reason,
        }


def verify_receipt(
    receipt: ExecutionReceipt,
    signature: str,
    signer: ReceiptSigner,
    delivered_records: Optional[list[dict]] = None,
) -> VerificationResult:
    """
    Verify both the signature and (optionally) that a delivered dataset
    matches the receipt's hash. Never returns the secret.
    """
    sig_ok = signer.verify(receipt, signature)
    ds_ok = True
    reason = ""

    if delivered_records is not None:
        delivered_hash = dataset_sha256(delivered_records)
        ds_ok = delivered_hash == receipt.dataset_sha256
        if not ds_ok:
            reason = "dataset hash mismatch"

    if not sig_ok:
        reason = (reason + "; " if reason else "") + "signature invalid"

    return VerificationResult(
        valid_signature=sig_ok,
        dataset_hash_matches=ds_ok,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    secret = b"super-secret-test-key"
    signer = ReceiptSigner(secret, key_version=1)
    builder = ReceiptBuilder(
        job_id="job-1", task_id="task-1", client_id="acme", signer=signer,
    )

    records = [
        {"title": "A", "price": "$10"},
        {"title": "B", "price": "$20"},
    ]
    audit = [{"event": "run_started"}, {"event": "run_completed"}]

    receipt, sig = builder.build(
        started_at="2026-09-24T00:00:00+00:00",
        completed_at="2026-09-24T00:00:30+00:00",
        records=records,
        pages_attempted=3,
        pages_succeeded=3,
        pages_failed=0,
        sources=[{"url": "https://x.com/a", "fetched_at": "2026-09-24T00:00:10+00:00"}],
        quality_score=0.95,
        strategy_ids=["s1"],
        model_calls={"gemini": 2, "groq": 1},
        audit_log=audit,
    )

    assert sig is not None
    assert receipt.records_extracted == 2
    assert receipt.duration_seconds == 30.0
    assert receipt.dataset_sha256

    # Verify
    v = verify_receipt(receipt, sig, signer, delivered_records=records)
    assert v.ok, v.reason

    # Tampered dataset
    v = verify_receipt(receipt, sig, signer,
                       delivered_records=records + [{"title": "X"}])
    assert not v.dataset_hash_matches
    assert not v.ok

    # Tampered receipt
    receipt.records_extracted = 999
    v = verify_receipt(receipt, sig, signer)
    assert not v.valid_signature

    print("Receipt signer OK.")