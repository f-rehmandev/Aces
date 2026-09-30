"""
Connector types — spec §47A.

A Connector is a destination. It receives either:
    - a validated dataset (as inline records or as a DatasetReference), or
    - a notification payload (for Slack / email / webhook channels).

The Connector itself never decides whether the data is trustworthy —
that's the publication gate's job (§22.6). The connector only ships it.

Design contract:

    class Connector(Protocol):
        connector_id: str
        connector_type: ConnectorType
        capabilities: set[ConnectorCapability]
        async def test_connection(self) -> ConnectorTestResult: ...
        async def publish(self, ref: DatasetReference, *,
                          metadata: dict | None = None) -> DeliveryResult: ...
        async def send(self, payload: dict, *,
                       metadata: dict | None = None) -> DeliveryResult: ...

Concrete connectors live in sibling modules:
    local_file.py       CSV / JSON / XLSX / Parquet to disk
    webhook.py          HTTP POST with HMAC signing
    s3.py               S3-compatible object storage (Tier 1)
    google_sheets.py    Google Sheets API (Tier 1)
    slack.py            Slack Incoming Webhook / API (Tier 2)
    email.py            SMTP or transactional API (Tier 2)
    airtable.py         Airtable REST (Tier 2)
    postgres.py         Postgres / Supabase table insert (Tier 3)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConnectorType(str, Enum):
    LOCAL_FILE = "local_file"
    WEBHOOK = "webhook"
    S3 = "s3"
    GOOGLE_SHEETS = "google_sheets"
    SLACK = "slack"
    EMAIL = "email"
    AIRTABLE = "airtable"
    POSTGRES = "postgres"
    SFTP = "sftp"


class ConnectorCapability(str, Enum):
    """
    What a connector can do. Callers should check capabilities rather
    than isinstance, because a connector can support a subset of these.

        DELIVERS_FILES          - writes a binary file (xlsx/csv/json/parquet)
        DELIVERS_ROWS           - appends rows to a table / sheet / DB
        DELIVERS_NOTIFICATIONS  - sends a text payload (Slack message, email body)
        SUPPORTS_SIGNED_URLS    - can consume a signed URL instead of inline data
        SUPPORTS_IDEMPOTENCY    - accepts and honours an idempotency key
    """
    DELIVERS_FILES = "delivers_files"
    DELIVERS_ROWS = "delivers_rows"
    DELIVERS_NOTIFICATIONS = "delivers_notifications"
    SUPPORTS_SIGNED_URLS = "supports_signed_urls"
    SUPPORTS_IDEMPOTENCY = "supports_idempotency"


# ---------------------------------------------------------------------------
# Dataset reference
# ---------------------------------------------------------------------------

@dataclass
class DatasetReference:
    """
    A handle to a dataset the connector should deliver.

    Two delivery modes:
        INLINE   — `records` is populated. Small datasets. Connector
                   writes them directly.
        BY_URL   — `url` is populated. Large datasets. Connector fetches
                   from the URL (typically a signed Supabase Storage URL).

    Exactly one of `records` or `url` is populated, unless the connector
    only supports notifications (in which case neither matters).

    `format` is a hint — the connector uses it when it needs to know
    how to serialize (e.g. Google Sheets expects rows, S3 might get a
    parquet file, email might get CSV).
    """
    format: str = "json"           # json | csv | xlsx | parquet | jsonl
    records: list[dict] = field(default_factory=list)
    url: str = ""                  # signed URL for large payloads
    filename: str = ""             # suggested filename
    content_type: str = ""         # suggested MIME type

    @property
    def mode(self) -> str:
        if self.url:
            return "url"
        if self.records:
            return "inline"
        return "empty"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ConnectorTestResult:
    """
    Result of a `test_connection()` call. Never raises — a connector
    that can't reach its destination returns ok=False with a message.
    """
    ok: bool
    message: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeliveryResult:
    """
    Result of a `publish()` / `send()` call.

    Connectors must never raise to the caller — they return a
    DeliveryResult with ok=False. The caller (delivery manager, job
    runner) decides whether to retry, dead-letter, or surface an error.
    """
    ok: bool
    connector_type: str = ""
    connector_id: str = ""
    destination: str = ""          # human-readable target (path, URL, sheet)
    delivery_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    bytes_sent: int = 0
    records_sent: int = 0
    attempts: int = 0
    latency_ms: int = 0
    error: str = ""
    metadata: dict = field(default_factory=dict)
    occurred_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Connector(Protocol):
    """
    Every concrete connector satisfies this shape. Uses `runtime_checkable`
    so callers can `isinstance(obj, Connector)` at runtime, though
    capability checks are preferred over type checks.
    """
    connector_id: str
    connector_type: ConnectorType
    capabilities: set

    async def test_connection(self) -> ConnectorTestResult: ...

    async def publish(
        self,
        ref: DatasetReference,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult: ...

    async def send(
        self,
        payload: dict,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult: ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConnectorError(Exception):
    """Raised for configuration-time errors (bad params, missing creds)."""


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ConnectorType
    assert ConnectorType.S3.value == "s3"
    assert ConnectorType.GOOGLE_SHEETS.value == "google_sheets"

    # Capability enum
    caps = {
        ConnectorCapability.DELIVERS_FILES,
        ConnectorCapability.SUPPORTS_IDEMPOTENCY,
    }
    assert ConnectorCapability.DELIVERS_FILES in caps

    # DatasetReference inline mode
    ref = DatasetReference(
        format="csv",
        records=[{"a": 1}, {"b": 2}],
        filename="out.csv",
    )
    assert ref.mode == "inline"
    assert len(ref.records) == 2

    # DatasetReference URL mode
    ref2 = DatasetReference(
        format="parquet",
        url="https://example.com/signed.parquet",
    )
    assert ref2.mode == "url"
    assert ref2.records == []

    # Empty
    empty = DatasetReference()
    assert empty.mode == "empty"

    # ConnectorTestResult
    tr = ConnectorTestResult(ok=True, message="ping ok", latency_ms=42)
    assert tr.to_dict()["ok"] is True
    assert tr.to_dict()["latency_ms"] == 42

    # DeliveryResult
    dr = DeliveryResult(
        ok=True,
        connector_type="s3",
        connector_id="conn-1",
        destination="s3://bucket/key",
        records_sent=100,
        bytes_sent=4096,
    )
    d = dr.to_dict()
    assert d["ok"] is True
    assert d["records_sent"] == 100
    assert d["destination"] == "s3://bucket/key"
    assert "delivery_id" in d

    # DeliveryResult failure shape
    dr_fail = DeliveryResult(ok=False, error="auth failed")
    assert dr_fail.ok is False
    assert dr_fail.error == "auth failed"

    # Protocol check: a minimal object that satisfies the shape
    class DummyConnector:
        connector_id = "d1"
        connector_type = ConnectorType.LOCAL_FILE
        capabilities = {ConnectorCapability.DELIVERS_FILES}
        async def test_connection(self):
            return ConnectorTestResult(ok=True)
        async def publish(self, ref, *, metadata=None):
            return DeliveryResult(ok=True)
        async def send(self, payload, *, metadata=None):
            return DeliveryResult(ok=True)

    assert isinstance(DummyConnector(), Connector)

    print("Connector types OK.")