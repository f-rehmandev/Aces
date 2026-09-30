"""Unit tests for multi-format output writers (spec §32)."""
import csv
import json
from pathlib import Path

import pytest

from src.output.formats import (
    write_csv, write_json, write_jsonl, write_parquet, write_xlsx,
    write_records, WriteResult, SUPPORTED_FORMATS,
)
from src.trust.cleaning import MISSING


SAMPLE = [
    {"title": "A", "price": "$1", "stock": 5},
    {"title": "B", "price": "$2", "stock": 0},
    {"title": "C", "price": None, "stock": 12},
]


# --- CSV ---------------------------------------------------------------

def test_csv_writes_header_and_rows(tmp_path: Path):
    r = write_csv(SAMPLE, tmp_path / "out.csv")
    assert r.record_count == 3
    text = (tmp_path / "out.csv").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "title,price,stock"
    assert "Wireless" not in text  # sanity: not this fixture
    assert "A,$1,5" in text


def test_csv_union_of_columns(tmp_path: Path):
    records = [{"a": 1}, {"b": 2}]
    write_csv(records, tmp_path / "u.csv")
    rows = list(csv.reader((tmp_path / "u.csv").read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["a", "b"]
    # Missing cells become empty
    assert rows[1] == ["1", ""]
    assert rows[2] == ["", "2"]


def test_csv_empty(tmp_path: Path):
    r = write_csv([], tmp_path / "e.csv")
    assert r.record_count == 0
    assert (tmp_path / "e.csv").read_text(encoding="utf-8") == ""


# --- JSON --------------------------------------------------------------

def test_json_writes_array(tmp_path: Path):
    r = write_json(SAMPLE, tmp_path / "out.json")
    data = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["title"] == "A"
    assert r.record_count == 3


def test_json_empty(tmp_path: Path):
    write_json([], tmp_path / "e.json")
    data = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
    assert data == []


# --- JSONL -------------------------------------------------------------

def test_jsonl_one_object_per_line(tmp_path: Path):
    r = write_jsonl(SAMPLE, tmp_path / "out.jsonl")
    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        obj = json.loads(line)
        assert obj["title"] == chr(ord("A") + i)
    assert r.record_count == 3


def test_jsonl_empty(tmp_path: Path):
    write_jsonl([], tmp_path / "e.jsonl")
    assert (tmp_path / "e.jsonl").read_text(encoding="utf-8") == ""


# --- Parquet -----------------------------------------------------------

def test_parquet_succeeds_or_warns(tmp_path: Path):
    r = write_parquet(SAMPLE, tmp_path / "out.parquet")
    # Either it wrote, or it gave a clear warning (pyarrow missing)
    assert r.format == "parquet"
    if r.warnings:
        assert "pyarrow" in r.warnings[0].lower() or "parquet" in r.warnings[0].lower()
    else:
        assert r.record_count == 3
        assert (tmp_path / "out.parquet").exists()


# --- XLSX --------------------------------------------------------------

def test_xlsx_writes_file(tmp_path: Path):
    r = write_xlsx(SAMPLE, tmp_path / "out.xlsx")
    assert r.record_count == 3
    assert (tmp_path / "out.xlsx").exists()


# --- MISSING sentinel --------------------------------------------------

def test_missing_sentinel_becomes_null_in_json(tmp_path: Path):
    write_json([{"a": MISSING, "b": "ok"}], tmp_path / "s.json")
    data = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert data[0]["a"] is None
    assert data[0]["b"] == "ok"


def test_missing_sentinel_becomes_empty_in_csv(tmp_path: Path):
    write_csv([{"a": MISSING, "b": "ok"}], tmp_path / "s.csv")
    text = (tmp_path / "s.csv").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "a,b"
    assert lines[1] == ",ok"


# --- dispatcher --------------------------------------------------------

def test_write_records_infers_format_from_extension(tmp_path: Path):
    r = write_records(SAMPLE, tmp_path / "x.json")
    assert r.format == "json"


def test_write_records_explicit_format_overrides_extension(tmp_path: Path):
    r = write_records(SAMPLE, tmp_path / "x.csv", format="json")
    assert r.format == "json"
    # Actually written as JSON
    data = json.loads((tmp_path / "x.csv").read_text(encoding="utf-8"))
    assert isinstance(data, list)


def test_write_records_unknown_format_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        write_records(SAMPLE, tmp_path / "x.doc")


def test_supported_formats_list():
    for fmt in ("csv", "json", "jsonl", "parquet", "xlsx"):
        assert fmt in SUPPORTED_FORMATS


# --- WriteResult -------------------------------------------------------

def test_write_result_to_dict():
    r = write_json(SAMPLE, Path("x.json"))
    d = r.to_dict()
    assert d["format"] == "json"
    assert d["record_count"] == 3
    assert "path" in d