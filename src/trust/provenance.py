"""
Provenance — spec §26.

Every value ACES publishes must trace back to a source URL, a moment,
and a method. The `ProvenanceRecord` is that chain. A `ProvenanceStore`
holds them keyed by (record_id, field_name) so a client can ask
"where did this number come from?" and get a real answer.
"""

from __future__ import annotations
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceRecord:
    record_id: str
    field_name: str
    source_url: str
    source_domain: str
    observed_at: str                      # ISO 8601 UTC
    fetched_at: str                       # ISO 8601 UTC
    extraction_method: str                # "css" | "jsonld" | "rung2_llm" | "vision" | ...
    selector_or_strategy_id: Optional[str] = None
    raw_value: Optional[str] = None
    normalized_value: Optional[str] = None
    confidence: float = 0.0
    consensus_group: Optional[str] = None
    run_id: str = ""
    job_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Record identity helper (used by callers)
# ---------------------------------------------------------------------------

def make_record_id(record: dict, prefix: str = "rec") -> str:
    """
    Stable per-record identity: hash of the identity fields, in priority
    order (same as diff/quality).
    """
    for key in ("title", "business_name", "job_title", "name", "url", "source_url"):
        v = record.get(key)
        if v:
            h = hashlib.sha256(str(v).encode("utf-8")).hexdigest()[:12]
            return f"{prefix}_{h}"
    # Last resort — random; caller should have a stable identity field.
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def domain_of(url: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(url).netloc or "").lower()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ProvenanceStore:
    """
    In-memory provenance store. Callers can persist the `to_dict()` output
    to Supabase/Postgres when the schema supports it.

    The store does not deduplicate or reconcile — it just records.
    Reconciliation belongs to triangulation (§23).
    """

    def __init__(
        self,
        run_id: str = "",
        job_id: str = "",
    ):
        self.run_id = run_id
        self.job_id = job_id
        self._records: list[ProvenanceRecord] = []

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def record(
        self,
        record_id: str,
        field_name: str,
        source_url: str,
        extraction_method: str,
        raw_value,
        normalized_value=None,
        selector_or_strategy_id: Optional[str] = None,
        confidence: float = 0.0,
        consensus_group: Optional[str] = None,
        observed_at: Optional[str] = None,
    ) -> ProvenanceRecord:
        rec = ProvenanceRecord(
            record_id=record_id,
            field_name=field_name,
            source_url=source_url,
            source_domain=domain_of(source_url),
            observed_at=observed_at or _utc_now(),
            fetched_at=_utc_now(),
            extraction_method=extraction_method,
            selector_or_strategy_id=selector_or_strategy_id,
            raw_value=_stringify(raw_value),
            normalized_value=_stringify(normalized_value)
                              if normalized_value is not None else None,
            confidence=confidence,
            consensus_group=consensus_group,
            run_id=self.run_id,
            job_id=self.job_id,
        )
        self._records.append(rec)
        return rec

    def record_batch(
        self,
        records: list[dict],
        source_url: str,
        extraction_method: str,
        ignored_fields: Optional[set[str]] = None,
    ) -> list[ProvenanceRecord]:
        """Record provenance for every field of every record."""
        ignored = ignored_fields or {"diff_status", "_numeric_price"}
        out = []
        for r in records:
            rid = make_record_id(r)
            for fname, value in r.items():
                if fname in ignored:
                    continue
                out.append(self.record(
                    record_id=rid,
                    field_name=fname,
                    source_url=source_url,
                    extraction_method=extraction_method,
                    raw_value=value,
                    normalized_value=None,
                ))
        return out

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def for_record(self, record_id: str) -> list[ProvenanceRecord]:
        return [r for r in self._records if r.record_id == record_id]

    def for_field(self, record_id: str, field_name: str) -> list[ProvenanceRecord]:
        return [
            r for r in self._records
            if r.record_id == record_id and r.field_name == field_name
        ]

    def __len__(self) -> int:
        return len(self._records)

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self._records]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stringify(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = ProvenanceStore(run_id="run-1", job_id="job-1")

    records = [
        {"title": "Wireless Mouse", "price": "$24.99", "source_url": "https://shop.example/a"},
        {"title": "Mechanical Keyboard", "price": "$75.00", "source_url": "https://shop.example/b"},
    ]
    store.record_batch(records, source_url="https://shop.example/a",
                       extraction_method="css")

    assert len(store) > 0

    # A specific field lookup
    rid = make_record_id(records[0])
    price_prov = store.for_field(rid, "price")
    assert len(price_prov) == 1
    p = price_prov[0]
    assert p.source_url == "https://shop.example/a"
    assert p.source_domain == "shop.example"
    assert p.raw_value == "$24.99"
    assert p.extraction_method == "css"

    # Bookkeeping fields are ignored
    assert not any(r.field_name == "diff_status" for r in store._records)

    # Explicit raw + normalized preservation
    store.record(
        record_id="r2",
        field_name="price",
        source_url="https://shop.example/b",
        extraction_method="rung2_llm",
        raw_value="£51.77",
        normalized_value="51.77",
        confidence=0.9,
    )
    p = store.for_field("r2", "price")[0]
    assert p.raw_value == "£51.77"
    assert p.normalized_value == "51.77"
    assert p.confidence == 0.9

    # to_list serialization
    serialized = store.to_list()
    assert isinstance(serialized, list)
    assert "field_name" in serialized[0]

    # domain_of
    assert domain_of("https://Example.com/x") == "example.com"

    print("Provenance OK.")