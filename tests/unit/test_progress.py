"""Unit tests for pagination/scroll progress detection (spec §14.3)."""
from src.navigation.progress import html_unchanged, records_unchanged


# --- html ---------------------------------------------------------------

def test_html_identical():
    assert html_unchanged("<p>a</p>", "<p>a</p>")


def test_html_equivalent_after_whitespace():
    assert html_unchanged("<p>a\n  b</p>", "<p>a b</p>")
    assert html_unchanged("  <p>x</p>  ", "<p>x</p>")


def test_html_different_content():
    assert not html_unchanged("<p>a</p>", "<p>b</p>")


def test_html_empty_vs_content():
    assert not html_unchanged("", "<p>x</p>")


def test_html_both_empty():
    assert html_unchanged("", "")


# --- records ------------------------------------------------------------

def _key(r):
    return r.get("id")


def test_records_same_set_different_order():
    assert records_unchanged(
        [{"id": "1"}, {"id": "2"}],
        [{"id": "2"}, {"id": "1"}],
        _key,
    )


def test_records_different_set():
    assert not records_unchanged(
        [{"id": "1"}, {"id": "2"}],
        [{"id": "1"}, {"id": "3"}],
        _key,
    )


def test_records_empty_vs_populated():
    assert not records_unchanged([{"id": "1"}], [], _key)
    assert not records_unchanged([], [{"id": "1"}], _key)


def test_records_both_empty():
    assert records_unchanged([], [], _key)


def test_records_without_keys_are_ignored():
    # Records missing the identity key contribute nothing to the comparison.
    assert records_unchanged([{"x": 1}], [{"y": 2}], _key)


def test_records_mixed_keys_and_garbage():
    assert records_unchanged(
        [{"id": "1"}, {"junk": True}],
        [{"id": "1"}, {"alsojunk": 5}],
        _key,
    )