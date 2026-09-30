"""Unit tests for S3Connector (spec §47A)."""
import asyncio
import json

import pytest

from src.integrations.s3 import S3Connector, _serialize_records
from src.integrations.types import (
    ConnectorCapability,
    ConnectorError,
    ConnectorType,
    DatasetReference,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeS3Client:
    """Minimal S3 client fake. Records all put_object calls."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.puts: list[dict] = []
        self.head_bucket_fails = False
        self.list_fails = False
        self.put_fails = False

    def head_bucket(self, Bucket):   # noqa: N803
        if self.head_bucket_fails:
            raise RuntimeError("no such bucket")
        return {}

    def list_objects_v2(self, Bucket, MaxKeys=1):   # noqa: N803
        if self.list_fails:
            raise RuntimeError("list failed")
        return {"Contents": []}

    def head_object(self, Bucket, Key):   # noqa: N803
        if Key in self.objects:
            return {"ContentLength": len(self.objects[Key])}
        raise FileNotFoundError(f"{Key} not found")

    def put_object(self, Bucket, Key, Body, **kwargs):   # noqa: N803
        if self.put_fails:
            raise RuntimeError("access denied")
        self.objects[Key] = Body
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body, **kwargs})
        return {}


def _conn(**kw) -> tuple[S3Connector, FakeS3Client]:
    fake = kw.pop("fake", None) or FakeS3Client()
    c = S3Connector(
        bucket=kw.pop("bucket", "test-bucket"),
        prefix=kw.pop("prefix", "ds"),
        client_factory=lambda: fake,
        **kw,
    )
    return c, fake


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_requires_bucket():
    with pytest.raises(ConnectorError):
        S3Connector(bucket="")


def test_default_capabilities():
    c, _ = _conn()
    assert c.connector_type == ConnectorType.S3
    assert c.supports(ConnectorCapability.DELIVERS_FILES)
    assert c.supports(ConnectorCapability.SUPPORTS_IDEMPOTENCY)
    assert not c.supports(ConnectorCapability.DELIVERS_NOTIFICATIONS)


def test_is_configured_with_client_factory():
    c, _ = _conn()
    assert c.is_configured() is True


# ---------------------------------------------------------------------------
# test_connection
# ---------------------------------------------------------------------------

def test_test_connection_uses_head_bucket():
    c, fake = _conn()
    tr = _run(c.test_connection())
    assert tr.ok is True
    assert "test-bucket" in tr.message


def test_test_connection_falls_back_to_list():
    c, fake = _conn()
    fake.head_bucket_fails = True
    tr = _run(c.test_connection())
    assert tr.ok is True
    assert "list_objects_v2" in tr.message


def test_test_connection_fails_cleanly():
    c, fake = _conn()
    fake.head_bucket_fails = True
    fake.list_fails = True
    tr = _run(c.test_connection())
    assert tr.ok is False
    assert "bucket check failed" in tr.message


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def test_serialize_json():
    payload = _serialize_records([{"a": 1}], "json")
    assert json.loads(payload.decode()) == [{"a": 1}]


def test_serialize_jsonl():
    payload = _serialize_records([{"a": 1}, {"a": 2}], "jsonl")
    lines = payload.decode().splitlines()
    assert len(lines) == 2


def test_serialize_csv():
    payload = _serialize_records(
        [{"name": "Alice", "age": 30}], "csv",
    )
    text = payload.decode()
    assert "name,age" in text
    assert "Alice,30" in text


def test_serialize_empty_csv():
    payload = _serialize_records([], "csv")
    assert payload == b""


def test_serialize_unsupported_format():
    with pytest.raises(ConnectorError):
        _serialize_records([{"x": 1}], "weird")


# ---------------------------------------------------------------------------
# publish()
# ---------------------------------------------------------------------------

def test_publish_json_uploads():
    c, fake = _conn()
    ref = DatasetReference(
        format="json",
        records=[{"a": 1}],
        filename="out.json",
    )
    dr = _run(c.publish(ref))
    assert dr.ok is True
    assert dr.destination == "s3://test-bucket/ds/out.json"
    assert "ds/out.json" in fake.objects
    assert dr.metadata["bucket"] == "test-bucket"
    assert dr.metadata["key"] == "ds/out.json"
    assert dr.metadata["format"] == "json"


def test_publish_csv_sets_content_type():
    c, fake = _conn()
    ref = DatasetReference(
        format="csv",
        records=[{"a": 1}],
        filename="d.csv",
    )
    _run(c.publish(ref))
    put = fake.puts[0]
    assert put["ContentType"] == "text/csv"


def test_publish_xlsx_via_tempfile():
    c, fake = _conn()
    ref = DatasetReference(
        format="xlsx",
        records=[{"a": 1}, {"b": 2}],
        filename="report.xlsx",
    )
    dr = _run(c.publish(ref))
    assert dr.ok is True
    assert len(fake.objects["ds/report.xlsx"]) > 0


def test_publish_parquet_graceful_when_pyarrow_missing():
    """If pyarrow is unavailable, parquet publish should fail cleanly."""
    try:
        import pyarrow  # noqa: F401
        has_pyarrow = True
    except ImportError:
        has_pyarrow = False

    c, fake = _conn()
    ref = DatasetReference(
        format="parquet",
        records=[{"x": 1}],
        filename="d.parquet",
    )
    dr = _run(c.publish(ref))
    if has_pyarrow:
        assert dr.ok is True
    else:
        assert dr.ok is False
        assert "serialization failed" in dr.error


def test_publish_auto_filename_when_none_given():
    c, fake = _conn()
    ref = DatasetReference(format="json", records=[{"a": 1}])
    dr = _run(c.publish(ref))
    assert dr.ok is True
    key = dr.metadata["key"]
    assert key.startswith("ds/delivery_")
    assert key.endswith(".json")


def test_publish_no_prefix_uses_bare_key():
    c, fake = _conn(prefix="")
    ref = DatasetReference(
        format="json",
        records=[{"a": 1}],
        filename="bare.json",
    )
    dr = _run(c.publish(ref))
    assert dr.metadata["key"] == "bare.json"


def test_publish_signed_url_rejected():
    c, fake = _conn()
    dr = _run(c.publish(DatasetReference(
        format="json",
        url="https://example.com/signed.json",
    )))
    assert dr.ok is False
    assert "inline records only" in dr.error


def test_publish_empty_records_rejected():
    c, fake = _conn()
    dr = _run(c.publish(DatasetReference(format="json")))
    assert dr.ok is False
    assert "no records" in dr.error


def test_publish_unsupported_format_rejected():
    c, fake = _conn()
    dr = _run(c.publish(DatasetReference(
        format="weird", records=[{"x": 1}], filename="x.weird",
    )))
    assert dr.ok is False
    assert "unsupported format" in dr.error


# ---------------------------------------------------------------------------
# SSE / storage class
# ---------------------------------------------------------------------------

def test_publish_with_sse_aes256():
    c, fake = _conn(sse="AES256")
    _run(c.publish(DatasetReference(
        format="json", records=[{"x": 1}], filename="e.json",
    )))
    assert fake.puts[0].get("ServerSideEncryption") == "AES256"


def test_publish_with_kms_and_storage_class():
    c, fake = _conn(
        sse="aws:kms",
        kms_key_id="arn:aws:kms:...:key/abc",
        storage_class="STANDARD_IA",
    )
    _run(c.publish(DatasetReference(
        format="json", records=[{"x": 1}], filename="k.json",
    )))
    put = fake.puts[0]
    assert put.get("ServerSideEncryption") == "aws:kms"
    assert put.get("SSEKMSKeyId") == "arn:aws:kms:...:key/abc"
    assert put.get("StorageClass") == "STANDARD_IA"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_idempotency_skips_second_upload():
    c, fake = _conn(prefix="p")
    ref = DatasetReference(
        format="json",
        records=[{"once": True}],
        filename="idem.json",
    )
    r1 = _run(c.publish(ref, metadata={"idempotency_key": "K-1"}))
    r2 = _run(c.publish(ref, metadata={"idempotency_key": "K-1"}))
    assert r1.ok and r2.ok
    assert r2.metadata.get("idempotent_replay") is True
    # Two objects: the data + the marker. Not three.
    assert len(fake.puts) == 2


def test_idempotency_writes_marker_object():
    c, fake = _conn(prefix="p")
    _run(c.publish(
        DatasetReference(
            format="json", records=[{"x": 1}], filename="x.json",
        ),
        metadata={"idempotency_key": "K-2"},
    ))
    marker_keys = [k for k in fake.objects if k.startswith("p/.idem/")]
    assert "p/.idem/K-2" in marker_keys


def test_idempotency_different_keys_upload_twice():
    c, fake = _conn(prefix="p")
    ref = DatasetReference(
        format="json", records=[{"x": 1}], filename="x.json",
    )
    _run(c.publish(ref, metadata={"idempotency_key": "A"}))
    _run(c.publish(ref, metadata={"idempotency_key": "B"}))
    # 2 data objects + 2 markers
    assert len(fake.puts) == 4


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_put_object_failure_returns_not_ok():
    c, fake = _conn()
    fake.put_fails = True
    dr = _run(c.publish(DatasetReference(
        format="json", records=[{"x": 1}], filename="x.json",
    )))
    assert dr.ok is False
    assert "put_object failed" in dr.error
    assert dr.records_sent == 1


def test_send_not_supported():
    c, _ = _conn()
    dr = _run(c.send({"text": "hi"}))
    assert dr.ok is False
    assert "does not support send" in dr.error