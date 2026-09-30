"""
Multi-format writers — spec §32.

Baseline formats:
    XLSX (via src.diff.excel_writer), CSV, JSON, JSONL
Advanced:
    Parquet (if pyarrow is available)

Every writer accepts a list of records and writes to a path.
Returns the actual path written, so callers can log it.
"""

from __future__ import annotations
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.trust.cleaning import MISSING


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class WriteResult:
    path: Path
    format: str
    record_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "format": self.format,
            "record_count": self.record_count,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_records(records: list[dict]) -> list[dict]:
    """
    Convert internal sentinel values into JSON-safe values. MISSING
    becomes None, everything else passes through.
    """
    out: list[dict] = []
    for r in records:
        clean: dict = {}
        for k, v in r.items():
            if v is MISSING:
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], output_path: str | Path) -> WriteResult:
    """
    Write records to CSV. Column order is the union of keys in first-seen
    order across all records.
    """
    path = Path(output_path)
    prepared = _prepare_records(records)

    if not prepared:
        path.write_text("", encoding="utf-8")
        return WriteResult(path=path, format="csv", record_count=0)

    # Union of keys, first-seen order
    columns: list[str] = []
    seen: set[str] = set()
    for r in prepared:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in prepared:
            # Fill missing columns with empty string
            row = {col: ("" if r.get(col) is None else r.get(col)) for col in columns}
            writer.writerow(row)

    return WriteResult(path=path, format="csv", record_count=len(prepared))


def write_json(
    records: list[dict],
    output_path: str | Path,
    indent: int = 2,
) -> WriteResult:
    """Write records as a single JSON array."""
    path = Path(output_path)
    prepared = _prepare_records(records)
    with path.open("w", encoding="utf-8") as f:
        json.dump(prepared, f, ensure_ascii=False, indent=indent, default=str)
    return WriteResult(path=path, format="json", record_count=len(prepared))


def write_jsonl(records: list[dict], output_path: str | Path) -> WriteResult:
    """Write records as newline-delimited JSON (one object per line)."""
    path = Path(output_path)
    prepared = _prepare_records(records)
    with path.open("w", encoding="utf-8") as f:
        for r in prepared:
            f.write(json.dumps(r, ensure_ascii=False, default=str))
            f.write("\n")
    return WriteResult(path=path, format="jsonl", record_count=len(prepared))


def write_parquet(records: list[dict], output_path: str | Path) -> WriteResult:
    """
    Write records as Parquet. Requires pyarrow; if not installed, we
    return a WriteResult with a warning and DO NOT raise — the caller
    can fall back to CSV or JSON.
    """
    path = Path(output_path)
    prepared = _prepare_records(records)

    try:
        import pandas as pd
    except ImportError:
        return WriteResult(
            path=path, format="parquet", record_count=0,
            warnings=["pandas is not installed; cannot write parquet"],
        )

    try:
        df = pd.DataFrame(prepared)
        df.to_parquet(path, index=False)
    except ImportError:
        return WriteResult(
            path=path, format="parquet", record_count=0,
            warnings=["pyarrow is not installed; run: pip install pyarrow"],
        )
    except Exception as e:
        return WriteResult(
            path=path, format="parquet", record_count=0,
            warnings=[f"parquet write failed: {e}"],
        )

    return WriteResult(path=path, format="parquet", record_count=len(prepared))


def write_xlsx(records: list[dict], output_path: str | Path) -> WriteResult:
    """
    Thin wrapper over the existing `write_excel` writer, so callers can
    use a uniform interface. Multi-sheet Excel lives in `workbook.py`
    (Round 2).
    """
    from src.diff.excel_writer import write_excel
    path = Path(output_path)
    prepared = _prepare_records(records)
    try:
        write_excel(prepared, str(path))
    except Exception as e:
        return WriteResult(
            path=path, format="xlsx", record_count=0,
            warnings=[f"xlsx write failed: {e}"],
        )
    return WriteResult(path=path, format="xlsx", record_count=len(prepared))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_WRITERS = {
    "csv":     write_csv,
    "json":    write_json,
    "jsonl":   write_jsonl,
    "parquet": write_parquet,
    "xlsx":    write_xlsx,
}


SUPPORTED_FORMATS = tuple(_WRITERS.keys())


def write_records(
    records: list[dict],
    output_path: str | Path,
    format: Optional[str] = None,
) -> WriteResult:
    """
    Write records in the requested format. If `format` is None, it is
    inferred from the output_path's extension.
    """
    if format is None:
        suffix = Path(output_path).suffix.lower().lstrip(".")
        format = suffix or "xlsx"

    fmt = format.lower()
    if fmt not in _WRITERS:
        raise ValueError(
            f"unsupported format {fmt!r}; choose one of {SUPPORTED_FORMATS}"
        )

    return _WRITERS[fmt](records, output_path)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    records = [
        {"title": "Wireless Mouse", "price": "$24.99", "stock": 5},
        {"title": "Keyboard", "price": "$75.00", "stock": 0},
        {"title": "Cable", "price": None, "stock": 12},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # CSV
        r = write_records(records, tmp / "out.csv")
        assert r.record_count == 3
        text = (tmp / "out.csv").read_text(encoding="utf-8")
        assert "Wireless Mouse" in text
        assert "title" in text
        print(f"csv: {r.record_count} records")

        # JSON
        r = write_records(records, tmp / "out.json")
        assert r.record_count == 3
        data = json.loads((tmp / "out.json").read_text(encoding="utf-8"))
        assert isinstance(data, list) and len(data) == 3
        print(f"json: {r.record_count} records")

        # JSONL
        r = write_records(records, tmp / "out.jsonl")
        lines = (tmp / "out.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["title"] == "Wireless Mouse"
        print(f"jsonl: {r.record_count} records")

        # Parquet — graceful if pyarrow missing
        r = write_records(records, tmp / "out.parquet")
        if r.warnings:
            print(f"parquet: skipped ({r.warnings[0]})")
        else:
            assert r.record_count == 3
            print(f"parquet: {r.record_count} records")

        # Explicit format override
        r = write_records(records, tmp / "anything.txt", format="json")
        assert r.format == "json"

        # MISSING sentinel becomes null
        from src.trust.cleaning import MISSING
        r = write_records([{"x": MISSING, "y": "ok"}], tmp / "sentinel.json")
        data = json.loads((tmp / "sentinel.json").read_text(encoding="utf-8"))
        assert data[0]["x"] is None
        assert data[0]["y"] == "ok"

        # Bad format
        try:
            write_records(records, tmp / "x.badformat")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

        # Empty records
        r = write_records([], tmp / "empty.csv")
        assert r.record_count == 0

    print("Formats OK.")