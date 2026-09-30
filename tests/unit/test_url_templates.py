"""Unit tests for URL template expansion (spec §14.1)."""
import pytest

from src.navigation.templates import expand, has_template


def test_simple_range():
    assert expand("https://x.com/p?page={1..3}") == [
        "https://x.com/p?page=1",
        "https://x.com/p?page=2",
        "https://x.com/p?page=3",
    ]


def test_range_with_step():
    assert expand("https://x.com/p?page={1..10:2}") == [
        "https://x.com/p?page=1",
        "https://x.com/p?page=3",
        "https://x.com/p?page=5",
        "https://x.com/p?page=7",
        "https://x.com/p?page=9",
    ]


def test_range_single_value():
    assert expand("https://x.com/p?page={5..5}") == ["https://x.com/p?page=5"]


def test_negative_range():
    assert expand("https://x.com/d?n={-2..0}") == [
        "https://x.com/d?n=-2",
        "https://x.com/d?n=-1",
        "https://x.com/d?n=0",
    ]


def test_range_end_before_start_raises():
    with pytest.raises(ValueError):
        expand("https://x.com/p={5..1}")


def test_range_zero_step_raises():
    with pytest.raises(ValueError):
        expand("https://x.com/p={1..5:0}")


def test_choices():
    assert expand("https://x.com/{a,b,c}") == [
        "https://x.com/a",
        "https://x.com/b",
        "https://x.com/c",
    ]


def test_choices_trimmed():
    assert expand("https://x.com/{ a , b }") == [
        "https://x.com/a",
        "https://x.com/b",
    ]


def test_cartesian_product():
    # Order is not guaranteed by the spec — compare as sets of URLs.
    got = sorted(expand("https://x.com/{a,b}?p={1..2}"))
    expected = sorted([
        "https://x.com/a?p=1",
        "https://x.com/a?p=2",
        "https://x.com/b?p=1",
        "https://x.com/b?p=2",
    ])
    assert got == expected


def test_cartesian_product_produces_all_combinations():
    """Independent check: every combination must appear exactly once."""
    got = expand("https://x.com/{shoes,bags}?page={1..3}")
    assert len(got) == 6
    assert len(set(got)) == 6
    for cat in ("shoes", "bags"):
        for page in (1, 2, 3):
            assert f"https://x.com/{cat}?page={page}" in got

def test_no_template_returns_input_as_single_element():
    assert expand("https://x.com/plain") == ["https://x.com/plain"]


def test_empty_returns_empty():
    assert expand("") == []


def test_max_urls_guard():
    with pytest.raises(ValueError):
        expand("https://x.com/p={1..100000}", max_urls=100)


def test_has_template():
    assert has_template("https://x.com/{1..3}")
    assert has_template("https://x.com/{a,b}")
    assert not has_template("https://x.com/plain")
    assert not has_template("{not a template}")  # no .. and no comma