"""Unit tests for UrlListResolver (spec §10)."""
import pytest
from pathlib import Path

from src.intake.resolver import UrlListResolver


# --- inline list -------------------------------------------------------

def test_inline_list_canonicalizes_and_dedupes():
    r = UrlListResolver().resolve([
        "https://x.com/a",
        "HTTPS://X.com/b?utm_source=foo#frag",
        "https://x.com/a",  # dup after normalization
    ])
    assert r.spec.target.start_urls == ["https://x.com/a", "https://x.com/b"]


def test_inline_list_drops_non_urls_with_warning():
    r = UrlListResolver().resolve(["https://x.com/a", "not a url"])
    assert r.spec.target.start_urls == ["https://x.com/a"]
    assert any("non-URL" in w for w in r.warnings)


def test_inline_list_all_invalid():
    r = UrlListResolver().resolve(["junk", "nope"])
    assert r.spec.target.start_urls == []
    assert any("no valid URLs" in w for w in r.warnings)


# --- text string -------------------------------------------------------

def test_multiline_string_with_comments_and_blanks():
    text = """
    # this is a comment
    https://x.com/p1

    https://x.com/p2
    # another comment
    https://x.com/p1
    """
    r = UrlListResolver().resolve(text)
    assert r.spec.target.start_urls == ["https://x.com/p1", "https://x.com/p2"]


def test_empty_string_yields_empty_spec():
    r = UrlListResolver().resolve("")
    assert r.spec.target.start_urls == []
    assert any("no valid URLs" in w for w in r.warnings)


# --- file: text --------------------------------------------------------

def test_txt_file(tmp_path: Path):
    p = tmp_path / "urls.txt"
    p.write_text("https://x.com/a\n# skip\nhttps://x.com/b\n", encoding="utf-8")
    r = UrlListResolver().resolve(p)
    assert r.spec.target.start_urls == ["https://x.com/a", "https://x.com/b"]


# --- file: CSV ---------------------------------------------------------

def test_csv_with_named_url_column(tmp_path: Path):
    p = tmp_path / "urls.csv"
    p.write_text(
        "name,url\nA,https://x.com/a\nB,https://x.com/b\n",
        encoding="utf-8",
    )
    r = UrlListResolver().resolve(p)
    assert r.spec.target.start_urls == ["https://x.com/a", "https://x.com/b"]


def test_csv_without_header_uses_first_column(tmp_path: Path):
    p = tmp_path / "urls.csv"
    p.write_text(
        "https://x.com/a,label-a\nhttps://x.com/b,label-b\n",
        encoding="utf-8",
    )
    r = UrlListResolver().resolve(p)
    assert r.spec.target.start_urls == ["https://x.com/a", "https://x.com/b"]


def test_csv_with_header_but_no_named_url_column(tmp_path: Path):
    p = tmp_path / "urls.csv"
    p.write_text(
        "link_target,note\nhttps://x.com/a,hello\n",
        encoding="utf-8",
    )
    r = UrlListResolver().resolve(p)
    assert r.spec.target.start_urls == ["https://x.com/a"]


def test_csv_with_short_rows_are_skipped(tmp_path: Path):
    p = tmp_path / "urls.csv"
    p.write_text(
        "url\nhttps://x.com/a\n\n",
        encoding="utf-8",
    )
    r = UrlListResolver().resolve(p)
    assert r.spec.target.start_urls == ["https://x.com/a"]


# --- error handling ----------------------------------------------------

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        UrlListResolver().resolve("definitely-not-here-12345.csv")


def test_unsupported_type_raises():
    with pytest.raises(TypeError):
        UrlListResolver().resolve(12345)


# --- result shape ------------------------------------------------------

def test_resolution_carries_source_description():
    r = UrlListResolver().resolve("https://x.com/a")
    assert "inline" in r.source_description.lower()
    assert isinstance(r.spec.target.start_urls, list)