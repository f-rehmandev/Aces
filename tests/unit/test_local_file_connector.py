"""Unit tests for LocalFileConnector (spec §47A)."""
import asyncio
import json
from pathlib import Path

import pytest

from src.integrations.local_file import LocalFileConnector
from src.integrations.types import (
    ConnectorCapability,
    ConnectorType,
    DatasetReference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    return LocalFileConnector(base_dir=tmp_path)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_requires_base_dir():
    with pytest.raises(Exception):
        LocalFileConnector(base_dir="")


def test_default_capabilities(conn):
    assert conn.connector_type == ConnectorType.LOCAL_FILE
    assert conn.supports(ConnectorCapability.DELIVERS_FILES)
    assert conn.supports(ConnectorCapability.SUPPORTS_IDEMPOTENCY)
    assert not conn.supports(ConnectorCapability.DELIVERS_NOTIFICATIONS)
    assert not conn.supports(ConnectorCapability.SUPPORTS_SIGNED_URLS)


def test_custom_connector_id(tmp_path):
    c = LocalFileConnector(base_dir=tmp_path, connector_id="my-conn")
    assert c.connector_id == "my-conn"


# ---------------------------------------------------------------------------
# test_connection
# ---------------------------------------------------------------------------

def test_test_connection_creates_missing_dir(tmp_path):
    missing = tmp_path / "nested" / "output"
    assert not missing.exists()
    c = LocalFileConnector(base_dir=missing)
    tr = _run(c.test_connection())
    assert tr.ok is True
    assert missing.exists()
    assert "writable" in tr.message


def test_test_connection_reports_latency(conn):
    tr = _run(conn.test_connection())
    assert tr.ok is True
    assert tr.latency_ms >= 0


# ---------------------------------------------------------------------------
# Happy-path publish: each format
# ---------------------------------------------------------------------------

def test_publish_json(conn, tmp_path):
    ref = DatasetReference(
        format="json",
        records=[{"a": 1}, {"a": 2}],
        filename="out.json",
    )
    dr = _run(conn.publish(ref))
    assert dr.ok is True
    assert dr.records_sent == 2
    assert dr.bytes_sent > 0
    assert dr.connector_type == "local_file"
    assert dr.connector_id == conn.connector_id

    data = json.loads(Path(dr.destination).read_text(encoding="utf-8"))
    assert data == [{"a": 1}, {"a": 2}]


def test_publish_csv(conn):
    ref = DatasetReference(
        format="csv",
        records=[{"name": "Alice", "age": 30}],
        filename="out.csv",
    )
    dr = _run(conn.publish(ref))
    assert dr.ok is True
    text = Path(dr.destination).read_text(encoding="utf-8")
    assert "name,age" in text
    assert "Alice,30" in text


def test_publish_jsonl(conn):
    ref = DatasetReference(
        format="jsonl",
        records=[{"x": 1}, {"x": 2}, {"x": 3}],
        filename="out.jsonl",
    )
    dr = _run(conn.publish(ref))
    lines = Path(dr.destination).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"x": 1}


def test_publish_xlsx(conn):
    ref = DatasetReference(
        format="xlsx",
        records=[{"a": 1, "b": 2}],
        filename="out.xlsx",
    )
    dr = _run(conn.publish(ref))
    assert dr.ok is True
    assert Path(dr.destination).exists()
    assert Path(dr.destination).stat().st_size > 0


# ---------------------------------------------------------------------------
# Filename resolution
# ---------------------------------------------------------------------------

def test_explicit_filename_wins(conn):
    ref = DatasetReference(
        format="json",
        records=[{"x": 1}],
        filename="chosen.json",
    )
    dr = _run(conn.publish(ref))
    assert Path(dr.destination).name == "chosen.json"


def test_filename_template(tmp_path):
    c = LocalFileConnector(
        base_dir=tmp_path,
        filename_template="report_{date}.{ext}",
    )
    ref = DatasetReference(format="csv", records=[{"z": 1}])
    dr = _run(c.publish(ref))
    name = Path(dr.destination).name
    assert name.startswith("report_")
    assert name.endswith(".csv")


def test_auto_filename_when_nothing_provided(conn):
    ref = DatasetReference(format="json", records=[{"x": 1}])
    dr = _run(conn.publish(ref))
    name = Path(dr.destination).name
    assert name.startswith("delivery_")
    assert name.endswith(".json")


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------

def test_second_publish_does_not_clobber_by_default(conn):
    ref = DatasetReference(
        format="json",
        records=[{"v": 1}],
        filename="dup.json",
    )
    dr1 = _run(conn.publish(ref))
    dr2 = _run(conn.publish(ref))
    assert dr1.destination != dr2.destination
    # Both files exist
    assert Path(dr1.destination).exists()
    assert Path(dr2.destination).exists()


def test_overwrite_true_replaces_file(tmp_path):
    c = LocalFileConnector(base_dir=tmp_path, overwrite=True)
    ref = DatasetReference(
        format="json",
        records=[{"v": 1}],
        filename="same.json",
    )
    dr1 = _run(c.publish(ref))
    dr2 = _run(c.publish(ref))
    assert dr1.destination == dr2.destination
    data = json.loads(Path(dr2.destination).read_text(encoding="utf-8"))
    assert data == [{"v": 1}]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_idempotency_same_key_is_noop(conn):
    ref = DatasetReference(
        format="json",
        records=[{"x": 1}],
        filename="idem.json",
    )
    dr1 = _run(conn.publish(ref, metadata={"idempotency_key": "K-1"}))
    dr2 = _run(conn.publish(ref, metadata={"idempotency_key": "K-1"}))
    assert dr1.ok is True
    assert dr2.ok is True
    assert dr2.metadata.get("idempotent_replay") is True
    assert dr1.destination == dr2.destination


def test_idempotency_different_keys_produce_different_files(conn):
    ref = DatasetReference(
        format="json",
        records=[{"x": 1}],
        filename="idem2.json",
    )
    dr1 = _run(conn.publish(ref, metadata={"idempotency_key": "K-A"}))
    dr2 = _run(conn.publish(ref, metadata={"idempotency_key": "K-B"}))
    assert dr1.destination != dr2.destination


def test_no_idempotency_key_means_each_call_is_fresh(conn):
    ref = DatasetReference(
        format="json",
        records=[{"x": 1}],
        filename="fresh.json",
    )
    dr1 = _run(conn.publish(ref))
    dr2 = _run(conn.publish(ref))
    assert dr1.destination != dr2.destination


# ---------------------------------------------------------------------------
# Safety: path traversal blocked
# ---------------------------------------------------------------------------

def test_path_traversal_blocked(conn):
    evil = DatasetReference(
        format="json",
        records=[{"x": 1}],
        filename="../../etc/passwd",
    )
    dr = _run(conn.publish(evil))
    assert dr.ok is False
    assert "escapes base_dir" in dr.error


def test_absolute_path_blocked(conn):
    evil = DatasetReference(
        format="json",
        records=[{"x": 1}],
        filename="/tmp/evil.json",
    )
    dr = _run(conn.publish(evil))
    assert dr.ok is False
    assert "escapes base_dir" in dr.error


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_empty_records_returns_not_ok(conn):
    ref = DatasetReference(format="json", records=[])
    dr = _run(conn.publish(ref))
    assert dr.ok is False
    assert "no records" in dr.error


def test_signed_url_only_returns_error(conn):
    ref = DatasetReference(
        format="json",
        url="https://example.com/signed.json",
    )
    dr = _run(conn.publish(ref))
    assert dr.ok is False
    assert "requires inline records" in dr.error


def test_unsupported_format_returns_error(conn):
    ref = DatasetReference(
        format="weird",
        records=[{"x": 1}],
        filename="x.weird",
    )
    dr = _run(conn.publish(ref))
    assert dr.ok is False
    assert "unsupported format" in dr.error


def test_send_not_supported(conn):
    dr = _run(conn.send({"text": "hi"}))
    assert dr.ok is False
    assert "does not support send" in dr.error


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------

def test_publish_result_metadata(conn):
    ref = DatasetReference(
        format="csv",
        records=[{"x": 1}],
        filename="meta.csv",
    )
    dr = _run(conn.publish(ref))
    assert dr.metadata.get("format") == "csv"
    assert dr.metadata.get("filename") == "meta.csv"
    assert "base_dir" in dr.metadata