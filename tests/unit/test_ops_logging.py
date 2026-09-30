"""Unit tests for structured logging (spec §40.1)."""
import json
import logging
from datetime import datetime, timezone

from src.ops.logging import LogEntry, JSONFormatter, get_json_logger, emit


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --- entry serialization ---------------------------------------------

def test_entry_to_json_has_schema():
    e = LogEntry(
        ts=_now(), run_id="r", job_id="j", client_id="acme",
        url="https://example.com/a", stage="extract",
        selector_id="price", action="extract_field",
        result="success", latency_ms=142,
    )
    d = json.loads(e.to_json())
    for k in ("ts", "run_id", "job_id", "client_id", "url", "stage",
              "selector_id", "action", "result", "latency_ms"):
        assert k in d
    assert d["latency_ms"] == 142


def test_error_message_is_secret_scrubbed():
    e = LogEntry(
        ts=_now(), stage="fetch", result="failed",
        error_type="HTTP_ERROR",
        error_message="key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789 rejected",
    )
    d = json.loads(e.to_json())
    assert "sk-proj-" not in d["error_message"]
    assert "<OPENAI_KEY_REDACTED>" in d["error_message"]


def test_url_is_secret_scrubbed():
    e = LogEntry(
        ts=_now(),
        url="https://example.com?token=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
    )
    d = json.loads(e.to_json())
    assert "sk-proj-" not in d["url"]


def test_empty_entry_serializes():
    e = LogEntry(ts=_now())
    d = json.loads(e.to_json())
    assert d["ts"] == e.ts


# --- formatter --------------------------------------------------------

def test_formatter_uses_structured_entry(caplog):
    logger = logging.getLogger("test_structured")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    e = LogEntry(ts=_now(), client_id="acme", stage="extract", result="success")
    with caplog.at_level(logging.INFO, logger="test_structured"):
        emit(logger, e)

    # caplog captures the record object
    records = [r for r in caplog.records if r.name == "test_structured"]
    assert records
    # The formatted message should be a JSON string
    formatted = formatter.format(records[-1])
    parsed = json.loads(formatted)
    assert parsed["client_id"] == "acme"


def test_formatter_falls_back_for_unstructured_record():
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname="", lineno=0,
        msg="plain message", args=(), exc_info=None,
    )
    out = JSONFormatter().format(record)
    d = json.loads(out)
    assert d["message"] == "plain message"
    assert d["level"] == "INFO"


def test_formatter_scrubs_unstructured_message():
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname="", lineno=0,
        msg="leaked sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
        args=(), exc_info=None,
    )
    out = JSONFormatter().format(record)
    assert "sk-proj-" not in out