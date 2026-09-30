"""Unit tests for URL normalization (spec §14.2)."""
import pytest

from src.navigation.url_normalizer import (
    normalize_url, canonicalize, urls_are_equivalent,
)


# --- scheme + host ------------------------------------------------------

def test_lowercases_scheme_and_host():
    r = normalize_url("HTTP://Example.COM/path")
    assert r.normalized == "http://example.com/path"
    assert "lowercased_scheme" in r.changes
    assert "lowercased_host" in r.changes


def test_preserves_path_case():
    r = normalize_url("https://example.com/Some/Path")
    assert r.normalized == "https://example.com/Some/Path"


# --- default ports ------------------------------------------------------

def test_strips_default_port_http():
    assert canonicalize("http://example.com:80/a") == "http://example.com/a"


def test_strips_default_port_https():
    assert canonicalize("https://example.com:443/a") == "https://example.com/a"


def test_keeps_non_default_port():
    assert canonicalize("https://example.com:8443/a") == "https://example.com:8443/a"


# --- dot segments -------------------------------------------------------

def test_resolves_dot_segments():
    r = normalize_url("https://example.com/a/b/../c")
    assert r.normalized == "https://example.com/a/c"
    assert "resolved_dot_segments" in r.changes


def test_resolves_leading_dot_segment():
    assert canonicalize("https://example.com/./a") == "https://example.com/a"


# --- fragments ----------------------------------------------------------

def test_strips_fragment():
    r = normalize_url("https://example.com/p#section-3")
    assert r.normalized == "https://example.com/p"
    assert "stripped_fragment" in r.changes


# --- tracking params ----------------------------------------------------

def test_removes_utm_params():
    r = normalize_url("https://example.com/p?utm_source=x&utm_medium=y&id=5")
    assert r.normalized == "https://example.com/p?id=5"
    assert "removed_tracking_params" in r.changes


def test_removes_fbclid_and_gclid():
    assert canonicalize(
        "https://example.com/p?fbclid=abc&gclid=xyz"
    ) == "https://example.com/p"


def test_keeps_meaningful_params_even_if_short():
    assert canonicalize("https://example.com/p?p=2&q=shoes") == "https://example.com/p?p=2&q=shoes"


def test_sorts_query_params_alphabetically():
    assert canonicalize("https://example.com/p?b=2&a=1") == "https://example.com/p?a=1&b=2"


def test_preserve_query_override():
    r = normalize_url("https://example.com/p?utm_source=x&utm_medium=y",
                      preserve_query={"utm_source"})
    # utm_source is kept; utm_medium is not
    assert "utm_source=x" in r.normalized
    assert "utm_medium" not in r.normalized


# --- relative URLs ------------------------------------------------------

def test_resolves_relative_url_against_base():
    r = normalize_url("/a/b", base="https://example.com/")
    assert r.normalized == "https://example.com/a/b"
    assert "resolved_against_base" in r.changes


def test_relative_parent_traversal():
    r = normalize_url("../c", base="https://example.com/a/b/")
    assert r.normalized == "https://example.com/a/c"


# --- stability ----------------------------------------------------------

def test_normalizing_twice_is_idempotent():
    once = canonicalize("HTTP://Example.COM:80/a/../b/?utm_source=x&b=2&a=1#f")
    twice = canonicalize(once)
    assert once == twice


def test_unchanged_url_reports_no_changes():
    r = normalize_url("https://example.com/p?a=1")
    assert r.normalized == "https://example.com/p?a=1"
    assert r.changes == []
    assert not r.was_modified


# --- equivalence --------------------------------------------------------

def test_equivalent_urls():
    assert urls_are_equivalent(
        "https://example.com/p?id=5&utm_source=x",
        "https://example.com/p?id=5",
    )
    assert urls_are_equivalent(
        "HTTP://EXAMPLE.COM/p",
        "http://example.com/p",
    )


def test_not_equivalent_when_path_differs():
    assert not urls_are_equivalent(
        "https://example.com/a",
        "https://example.com/b",
    )


def test_not_equivalent_when_id_differs():
    assert not urls_are_equivalent(
        "https://example.com/p?id=5",
        "https://example.com/p?id=6",
    )


def test_not_equivalent_when_path_case_differs():
    # §14.2 explicitly says path case is preserved
    assert not urls_are_equivalent(
        "https://example.com/a",
        "https://example.com/A",
    )


# --- edge cases ---------------------------------------------------------

def test_empty_path_becomes_slash():
    assert canonicalize("https://example.com") == "https://example.com/"


def test_preserves_blank_query_values():
    # keep_blank_values=True — don't silently drop "?q="
    assert canonicalize("https://example.com/p?q=") == "https://example.com/p?q="


def test_unparseable_url_returned_unchanged():
    r = normalize_url("")
    assert r.normalized == ""