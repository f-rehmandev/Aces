"""
Structured logging — spec §40.1.

Emits JSON log records with a fixed schema so downstream aggregation
(Datadog, Loki, etc.) can parse them without regex surgery.

Every log entry is scrubbed of secrets via §52 before emission.
"""

from __future__ import annotations
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Record shape (§40.1)
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    ts: str
    run_id: str = ""
    job_id: str = ""
    client_id: str = ""
    url: str = ""
    stage: str = ""            # "fetch" | "extract" | "quality" | ...
    selector_id: str = ""
    action: str = ""
    result: str = ""           # "success" | "failed" | "skipped"
    latency_ms: Optional[int] = None
    error_type: Optional[str] = None
    error_message: str = ""

    def to_json(self) -> str:
        from src.security.secrets import scrub
        d = asdict(self)
        # Only scrub the free-text fields
        for k in ("error_message", "url"):
            if d.get(k):
                d[k] = scrub(str(d[k])).scrubbed
        return json.dumps(d, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# JSON formatter for stdlib logging
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """
    Attach a JSON payload as `extra={"structured": LogEntry(...)}` to a
    stdlib log record; this formatter emits it.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = getattr(record, "structured", None)
        if isinstance(entry, LogEntry):
            return entry.to_json()
        # Fallback: generic JSON with message
        from src.security.secrets import scrub
        fallback = {
            "ts": _utc_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": scrub(record.getMessage()).scrubbed,
        }
        return json.dumps(fallback, ensure_ascii=False, default=str)


def get_json_logger(name: str = "aces") -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def emit(logger: logging.Logger, entry: LogEntry, level: int = logging.INFO) -> None:
    logger.log(level, entry.action or entry.stage, extra={"structured": entry})


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    e = LogEntry(
        ts=_utc_iso(),
        run_id="r-1",
        job_id="j-1",
        client_id="acme",
        url="https://example.com/p?q=1",
        stage="extract",
        selector_id="price",
        action="extract_field",
        result="success",
        latency_ms=142,
    )
    js = e.to_json()
    parsed = json.loads(js)
    assert parsed["client_id"] == "acme"
    assert parsed["selector_id"] == "price"
    assert parsed["latency_ms"] == 142

    # Secret in error_message gets scrubbed
    e2 = LogEntry(
        ts=_utc_iso(),
        stage="fetch",
        result="failed",
        error_type="HTTP_ERROR",
        error_message="provider key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789 rejected",
    )
    parsed2 = json.loads(e2.to_json())
    assert "sk-proj-" not in parsed2["error_message"]
    assert "<OPENAI_KEY_REDACTED>" in parsed2["error_message"]

    # Formatter works
    logger = get_json_logger("test_aces")
    emit(logger, e)
    # No exception = OK

    print("Structured logging OK.")