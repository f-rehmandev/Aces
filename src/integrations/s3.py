"""
S3Connector — spec §47A.

Writes a serialized dataset to any S3-compatible bucket:
    - AWS S3
    - Cloudflare R2
    - Backblaze B2 (S3-compatible endpoint)
    - MinIO / self-hosted

Uses `boto3` which is imported lazily — the module loads even without
it installed, and `is_configured()` returns False in that case so
`build_production_registry()` skips it cleanly.

Serialization:
    - json / jsonl / csv     → produced in-memory, uploaded as bytes
    - xlsx / parquet         → produced via a temp file (their writers
                                require a path), then uploaded as bytes

Safety:
    - The object key is prefixed with the configured prefix.
    - Idempotency is tracked via a small marker object in
      `<prefix>/.idem/<key>` — a second publish with the same
      idempotency_key skips the upload and reports a replay.
    - Optional SSE (`AES256` or `aws:kms`) and KMS key id.
    - Optional storage class.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
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


logger = logging.getLogger("integrations.s3")


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

_FORMAT_TO_EXT = {
    "json": "json",
    "jsonl": "jsonl",
    "csv": "csv",
    "xlsx": "xlsx",
    "parquet": "parquet",
}

_CONTENT_TYPES = {
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "parquet": "application/vnd.apache.parquet",
}


def _serialize_records(records: list[dict], fmt: str) -> bytes:
    """Serialize records to bytes in the requested format."""
    if fmt == "json":
        return json.dumps(
            records, ensure_ascii=False, default=str,
        ).encode("utf-8")

    if fmt == "jsonl":
        lines = [
            json.dumps(r, ensure_ascii=False, default=str)
            for r in records
        ]
        return ("\n".join(lines)).encode("utf-8")

    if fmt == "csv":
        if not records:
            return b""
        columns: list[str] = []
        seen: set[str] = set()
        for r in records:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    columns.append(k)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {c: ("" if r.get(c) is None else r.get(c)) for c in columns}
            writer.writerow(row)
        return buf.getvalue().encode("utf-8")

    # xlsx / parquet need a real file
    if fmt in ("xlsx", "parquet"):
        from src.output.formats import write_records
        with tempfile.NamedTemporaryFile(
            suffix=f".{fmt}", delete=False,
        ) as f:
            tmp = Path(f.name)
        try:
            result = write_records(records, tmp, format=fmt)
            if result.warnings:
                raise ConnectorError(
                    f"serialization failed: {'; '.join(result.warnings)}"
                )
            return tmp.read_bytes()
        finally:
            tmp.unlink(missing_ok=True)

    raise ConnectorError(f"unsupported format {fmt!r}")


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class S3Connector(BaseConnector):
    connector_type = ConnectorType.S3

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "",
        endpoint_url: str = "",
        access_key: str = "",
        secret_key: str = "",
        session_token: str = "",
        storage_class: str = "",
        sse: str = "",                # "AES256" | "aws:kms" | ""
        kms_key_id: str = "",
        connector_id: Optional[str] = None,
        client_factory=None,
    ):
        if not bucket:
            raise ConnectorError("s3 connector requires a bucket")

        self.bucket = bucket
        self.prefix = (prefix or "").strip("/")
        self.region = region
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token
        self.storage_class = storage_class
        self.sse = sse
        self.kms_key_id = kms_key_id
        self._client = None
        self._client_factory = client_factory

        super().__init__(
            connector_id=connector_id,
            capabilities={
                ConnectorCapability.DELIVERS_FILES,
                ConnectorCapability.SUPPORTS_IDEMPOTENCY,
            },
        )

    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        if self._client_factory is not None:
            return True
        try:
            import boto3  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        try:
            import boto3
        except ImportError as e:
            raise ConnectorError(f"boto3 not installed: {e}") from e
        self._client = boto3.client(
            "s3",
            region_name=self.region or None,
            endpoint_url=self.endpoint_url or None,
            aws_access_key_id=self.access_key or None,
            aws_secret_access_key=self.secret_key or None,
            aws_session_token=self.session_token or None,
        )
        return self._client

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    async def _do_test(self) -> ConnectorTestResult:
        client = self._get_client()
        # head_bucket is the cheapest access check; some S3-compatibles
        # don't implement it, so fall back to a 1-key list.
        try:
            client.head_bucket(Bucket=self.bucket)
            return ConnectorTestResult(
                ok=True,
                message=f"bucket {self.bucket!r} reachable",
            )
        except Exception as head_err:
            try:
                client.list_objects_v2(Bucket=self.bucket, MaxKeys=1)
                return ConnectorTestResult(
                    ok=True,
                    message=(
                        f"bucket {self.bucket!r} reachable "
                        f"(via list_objects_v2)"
                    ),
                )
            except Exception as list_err:
                return ConnectorTestResult(
                    ok=False,
                    message=(
                        f"bucket check failed: "
                        f"head={type(head_err).__name__}: {head_err}; "
                        f"list={type(list_err).__name__}: {list_err}"
                    ),
                )

    async def _do_publish(
        self,
        ref: DatasetReference,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        metadata = dict(metadata or {})

        if ref.url:
            return DeliveryResult(
                ok=False,
                destination=f"s3://{self.bucket}",
                error=(
                    "S3Connector writes inline records only; "
                    "server-side copy from a URL is not supported"
                ),
            )
        if not ref.records:
            return DeliveryResult(
                ok=False,
                destination=f"s3://{self.bucket}",
                error="no records to write",
            )

        fmt = (ref.format or "json").lower()
        if fmt not in _FORMAT_TO_EXT:
            raise ConnectorError(
                f"unsupported format {fmt!r}; "
                f"expected one of {sorted(_FORMAT_TO_EXT)}"
            )

        client = self._get_client()

        # ---- Idempotency via marker object ----
        idem_key = metadata.get("idempotency_key") or ""
        marker_key = ""
        if idem_key:
            marker_key = self._idem_marker_key(idem_key)
            try:
                client.head_object(Bucket=self.bucket, Key=marker_key)
                logger.info(
                    f"s3: idempotency hit for {idem_key!r} "
                    f"(marker {marker_key})"
                )
                return DeliveryResult(
                    ok=True,
                    destination=f"s3://{self.bucket}/{self._key_for(ref, fmt)}",
                    metadata={"idempotent_replay": True},
                )
            except Exception:
                # Marker doesn't exist — proceed with the upload
                pass

        # ---- Serialize ----
        try:
            payload = _serialize_records(ref.records, fmt)
        except ConnectorError:
            raise
        except Exception as e:
            raise ConnectorError(
                f"serialization failed: {type(e).__name__}: {e}"
            ) from e

        # ---- Upload ----
        key = self._key_for(ref, fmt)
        extra_args: dict = {"ContentType": _CONTENT_TYPES.get(
            fmt, "application/octet-stream")}
        if self.sse:
            extra_args["ServerSideEncryption"] = self.sse
        if self.kms_key_id:
            extra_args["SSEKMSKeyId"] = self.kms_key_id
        if self.storage_class:
            extra_args["StorageClass"] = self.storage_class

        try:
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=payload,
                **extra_args,
            )
        except Exception as e:
            return DeliveryResult(
                ok=False,
                destination=f"s3://{self.bucket}/{key}",
                error=f"put_object failed: {type(e).__name__}: {e}",
                records_sent=len(ref.records),
            )

        # ---- Write marker if idempotency was requested ----
        if idem_key:
            try:
                client.put_object(
                    Bucket=self.bucket,
                    Key=marker_key,
                    Body=b"1",
                )
            except Exception as e:
                logger.warning(
                    f"s3: idempotency marker write failed for "
                    f"{idem_key!r}: {e}"
                )

        return DeliveryResult(
            ok=True,
            destination=f"s3://{self.bucket}/{key}",
            bytes_sent=len(payload),
            records_sent=len(ref.records),
            metadata={
                "bucket": self.bucket,
                "key": key,
                "format": fmt,
                "content_type": extra_args["ContentType"],
            },
        )

    async def _do_send(
        self,
        payload: dict,
        *,
        metadata: Optional[dict] = None,
    ) -> DeliveryResult:
        raise ConnectorError(
            "S3Connector does not support send(); "
            "use publish() to write datasets"
        )

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------
    def _key_for(self, ref: DatasetReference, fmt: str) -> str:
        """s3://bucket/<prefix>/<filename>"""
        ext = _FORMAT_TO_EXT[fmt]
        if ref.filename:
            filename = ref.filename
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            filename = f"delivery_{ts}_{uuid.uuid4().hex[:8]}.{ext}"

        if self.prefix:
            return f"{self.prefix}/{filename}"
        return filename

    def _idem_marker_key(self, idem_key: str) -> str:
        marker = f".idem/{idem_key}"
        if self.prefix:
            return f"{self.prefix}/{marker}"
        return marker


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    class FakeClient:
        """
        Minimal S3 client fake. Tracks put_object calls and supports
        head_object for idempotency-marker tests.
        """
        def __init__(self):
            self.objects: dict[str, bytes] = {}
            self.puts: list[dict] = []
            self.head_bucket_should_fail = False
            self.list_should_fail = False

        def head_bucket(self, Bucket):   # noqa: N803
            if self.head_bucket_should_fail:
                raise RuntimeError("no such bucket")
            return {}

        def list_objects_v2(self, Bucket, MaxKeys=1):   # noqa: N803
            if self.list_should_fail:
                raise RuntimeError("list failed")
            return {"Contents": []}

        def head_object(self, Bucket, Key):   # noqa: N803
            if Key in self.objects:
                return {"ContentLength": len(self.objects[Key])}
            raise FileNotFoundError(f"{Key} not found")

        def put_object(self, Bucket, Key, Body, **kwargs):   # noqa: N803
            self.objects[Key] = Body
            self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body, **kwargs})
            return {}

    async def run():
        # ---- Construction ----
        try:
            S3Connector(bucket="")
            raise AssertionError("expected ConnectorError")
        except ConnectorError:
            pass

        fake = FakeClient()
        c = S3Connector(
            bucket="my-bucket",
            prefix="datasets/2026",
            client_factory=lambda: fake,
        )
        assert c.connector_type == ConnectorType.S3
        assert c.supports(ConnectorCapability.DELIVERS_FILES)
        assert c.supports(ConnectorCapability.SUPPORTS_IDEMPOTENCY)
        assert c.is_configured() is True

        # ---- test_connection via head_bucket ----
        tr = await c.test_connection()
        assert tr.ok is True
        assert "my-bucket" in tr.message

        # ---- test_connection falls back to list ----
        fake.head_bucket_should_fail = True
        tr = await c.test_connection()
        assert tr.ok is True
        assert "list_objects_v2" in tr.message
        fake.head_bucket_should_fail = False

        # ---- publish json ----
        ref = DatasetReference(
            format="json",
            records=[{"a": 1}, {"a": 2}],
            filename="out.json",
        )
        dr = await c.publish(ref)
        assert dr.ok is True, dr.error
        assert dr.records_sent == 2
        assert dr.destination == "s3://my-bucket/datasets/2026/out.json"
        assert dr.bytes_sent > 0
        assert dr.metadata["content_type"] == "application/json"
        assert "datasets/2026/out.json" in fake.objects
        body = fake.objects["datasets/2026/out.json"]
        assert json.loads(body.decode()) == [{"a": 1}, {"a": 2}]

        # ---- publish csv ----
        ref_csv = DatasetReference(
            format="csv",
            records=[{"name": "Alice", "age": 30}],
            filename="people.csv",
        )
        dr2 = await c.publish(ref_csv)
        assert dr2.ok is True
        csv_body = fake.objects["datasets/2026/people.csv"].decode()
        assert "name,age" in csv_body
        assert "Alice,30" in csv_body

        # ---- publish xlsx (needs temp file) ----
        ref_xlsx = DatasetReference(
            format="xlsx",
            records=[{"a": 1}],
            filename="report.xlsx",
        )
        dr3 = await c.publish(ref_xlsx)
        assert dr3.ok is True
        assert len(fake.objects["datasets/2026/report.xlsx"]) > 0

        # ---- publish jsonl ----
        ref_jsonl = DatasetReference(
            format="jsonl",
            records=[{"x": 1}, {"x": 2}],
            filename="lines.jsonl",
        )
        dr4 = await c.publish(ref_jsonl)
        assert dr4.ok is True
        jsonl_body = fake.objects["datasets/2026/lines.jsonl"].decode()
        assert len(jsonl_body.splitlines()) == 2

        # ---- Auto-generated filename ----
        ref_auto = DatasetReference(format="json", records=[{"n": 1}])
        dr5 = await c.publish(ref_auto)
        assert dr5.ok is True
        auto_key = dr5.metadata["key"]
        assert auto_key.startswith("datasets/2026/delivery_")
        assert auto_key.endswith(".json")

        # ---- SSE + storage class ----
        fake2 = FakeClient()
        c_sse = S3Connector(
            bucket="secure-bucket",
            sse="AES256",
            storage_class="STANDARD_IA",
            client_factory=lambda: fake2,
        )
        ref_sse = DatasetReference(
            format="json",
            records=[{"x": 1}],
            filename="enc.json",
        )
        await c_sse.publish(ref_sse)
        put = fake2.puts[0]
        assert put.get("ServerSideEncryption") == "AES256"
        assert put.get("StorageClass") == "STANDARD_IA"

        # ---- Idempotency ----
        fake3 = FakeClient()
        c3 = S3Connector(bucket="b", prefix="p", client_factory=lambda: fake3)
        ref_idem = DatasetReference(
            format="json",
            records=[{"once": True}],
            filename="idem.json",
        )
        r1 = await c3.publish(ref_idem, metadata={"idempotency_key": "K-1"})
        r2 = await c3.publish(ref_idem, metadata={"idempotency_key": "K-1"})
        assert r1.ok and r2.ok
        assert r2.metadata.get("idempotent_replay") is True
        # Only one real object stored (plus marker)
        real_objects = [k for k in fake3.objects if not k.startswith("p/.idem/")]
        assert len(real_objects) == 1
        assert "p/.idem/K-1" in fake3.objects

        # ---- Signed URL rejected ----
        dr_bad = await c.publish(DatasetReference(
            format="json",
            url="https://example.com/x.json",
        ))
        assert dr_bad.ok is False
        assert "inline records only" in dr_bad.error

        # ---- Empty records rejected ----
        dr_empty = await c.publish(DatasetReference(format="json"))
        assert dr_empty.ok is False
        assert "no records" in dr_empty.error

        # ---- Unsupported format ----
        dr_fmt = await c.publish(DatasetReference(
            format="weird",
            records=[{"x": 1}],
            filename="x.weird",
        ))
        assert dr_fmt.ok is False
        assert "unsupported format" in dr_fmt.error

        # ---- put_object failure is graceful ----
        class FailPut(FakeClient):
            def put_object(self, *a, **k):
                raise RuntimeError("access denied")
        c_fail = S3Connector(bucket="b", client_factory=lambda: FailPut())
        dr_fail = await c_fail.publish(DatasetReference(
            format="json",
            records=[{"x": 1}],
            filename="x.json",
        ))
        assert dr_fail.ok is False
        assert "put_object failed" in dr_fail.error

        # ---- send() not supported ----
        dr_send = await c.send({"text": "hi"})
        assert dr_send.ok is False
        assert "does not support send" in dr_send.error

        print("S3Connector OK.")

    asyncio.run(run())