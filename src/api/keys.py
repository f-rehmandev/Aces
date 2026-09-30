"""
API key model — spec §45.1.

Keys are hashed at rest; only the prefix is displayed after creation.
Revocation is immediate. Rate limiting is tracked per key.
"""

from __future__ import annotations
import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Key format
# ---------------------------------------------------------------------------

KEY_PREFIX = "aces_"
SECRET_BYTES = 32          # base64-ish secret
VISIBLE_PREFIX_LEN = 12    # how much of the key we show to the user after creation


def generate_key() -> str:
    """Return a fresh key: 'aces_' + 43 chars of urlsafe random."""
    return KEY_PREFIX + secrets.token_urlsafe(SECRET_BYTES)


def hash_key(raw_key: str, pepper: bytes = b"") -> str:
    """
    Hash a key for storage. We use SHA-256 with an optional server-side
    pepper. HMAC-SHA256 is used when a pepper is provided so the pepper
    actually affects the digest.
    """
    if pepper:
        return hmac.new(pepper, raw_key.encode("utf-8"), hashlib.sha256).hexdigest()
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def key_prefix(raw_key: str) -> str:
    """The visible prefix shown to the user after creation."""
    return raw_key[:VISIBLE_PREFIX_LEN] + "…"


# ---------------------------------------------------------------------------
# Key record
# ---------------------------------------------------------------------------

@dataclass
class ApiKey:
    key_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_id: str = ""
    key_prefix: str = ""           # visible-only prefix for display
    key_hash: str = ""             # server-side hash
    created_at: str = field(default_factory=_utc_iso)
    revoked: bool = False
    revoked_at: Optional[str] = None
    last_used_at: Optional[str] = None
    note: str = ""
    scopes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Never expose the hash in serialized output
        d.pop("key_hash", None)
        return d


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ApiKeyStore:
    """
    In-memory API key store. Persistence can be layered on top via
    `to_rows` / `from_rows` (the DB would never receive the raw key).
    """

    def __init__(self, pepper: bytes = b""):
        self._pepper = pepper
        self._by_id: dict[str, ApiKey] = {}
        # hash -> key_id (for O(1) lookup on incoming requests)
        self._by_hash: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------
    def create(
        self,
        client_id: str,
        note: str = "",
        scopes: Optional[list[str]] = None,
    ) -> tuple[ApiKey, str]:
        """
        Returns (key_record, raw_key). The raw key is shown ONCE to the
        caller and never stored.
        """
        raw = generate_key()
        record = ApiKey(
            client_id=client_id,
            key_prefix=key_prefix(raw),
            key_hash=hash_key(raw, self._pepper),
            note=note,
            scopes=list(scopes or []),
        )
        self._by_id[record.key_id] = record
        self._by_hash[record.key_hash] = record.key_id
        return record, raw

    # ------------------------------------------------------------------
    # Lookup / verification
    # ------------------------------------------------------------------
    def verify(self, raw_key: str) -> Optional[ApiKey]:
        """Return the ApiKey if valid and not revoked, else None."""
        h = hash_key(raw_key, self._pepper)
        kid = self._by_hash.get(h)
        if not kid:
            return None
        record = self._by_id.get(kid)
        if not record or record.revoked:
            return None
        record.last_used_at = _utc_iso()
        return record

    # ------------------------------------------------------------------
    # Revocation
    # ------------------------------------------------------------------
    def revoke(self, key_id: str) -> bool:
        record = self._by_id.get(key_id)
        if not record or record.revoked:
            return False
        record.revoked = True
        record.revoked_at = _utc_iso()
        return True

    def get(self, key_id: str) -> Optional[ApiKey]:
        return self._by_id.get(key_id)

    def list_for_client(self, client_id: str) -> list[ApiKey]:
        return [k for k in self._by_id.values() if k.client_id == client_id]

    # ------------------------------------------------------------------
    # Persistence (no raw keys are ever exported)
    # ------------------------------------------------------------------
    def to_rows(self) -> list[dict]:
        return [asdict(k) for k in self._by_id.values()]

    @classmethod
    def from_rows(
        cls, rows: list[dict], pepper: bytes = b"",
    ) -> "ApiKeyStore":
        store = cls(pepper=pepper)
        for row in rows:
            k = ApiKey(
                key_id=row.get("key_id", str(uuid.uuid4())),
                client_id=row.get("client_id", ""),
                key_prefix=row.get("key_prefix", ""),
                key_hash=row.get("key_hash", ""),
                created_at=row.get("created_at", _utc_iso()),
                revoked=bool(row.get("revoked", False)),
                revoked_at=row.get("revoked_at"),
                last_used_at=row.get("last_used_at"),
                note=row.get("note", ""),
                scopes=list(row.get("scopes", [])),
            )
            store._by_id[k.key_id] = k
            if k.key_hash:
                store._by_hash[k.key_hash] = k.key_id
        return store


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = ApiKeyStore(pepper=b"test-pepper")

    record, raw = store.create("acme", note="dev key", scopes=["jobs:read"])
    assert raw.startswith("aces_")
    assert record.key_prefix.endswith("…")
    assert "key_hash" not in record.to_dict()

    # Verify works
    verified = store.verify(raw)
    assert verified is not None and verified.key_id == record.key_id
    assert verified.last_used_at is not None

    # Wrong key rejected
    assert store.verify("aces_wrong") is None

    # Revoked key rejected
    assert store.revoke(record.key_id)
    assert store.verify(raw) is None
    # Idempotent revoke returns False
    assert not store.revoke(record.key_id)

    # List
    rec2, raw2 = store.create("acme", note="second")
    rec3, _ = store.create("other")
    assert len(store.list_for_client("acme")) == 2
    assert len(store.list_for_client("other")) == 1

    # Persistence roundtrip
    rows = store.to_rows()
    store2 = ApiKeyStore.from_rows(rows, pepper=b"test-pepper")
    assert store2.verify(raw2) is not None

    # Same raw key hashed twice yields the same value with the same pepper
    assert hash_key(raw, b"p") == hash_key(raw, b"p")
    # Different pepper yields different hash
    assert hash_key(raw, b"p1") != hash_key(raw, b"p2")

    print("API keys OK.")