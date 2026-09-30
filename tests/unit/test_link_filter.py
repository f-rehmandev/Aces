"""Unit tests for link filtering (spec §14.1)."""
from src.navigation.link_filter import LinkFilter, LinkFilterPolicy


ORIGIN = "https://shop.example.com/category/shoes"


def make(**overrides) -> LinkFilter:
    defaults = dict(
        include_patterns=[],
        exclude_patterns=[],
        same_domain_only=True,
        max_depth=None,
    )
    defaults.update(overrides)
    return LinkFilter(LinkFilterPolicy(**defaults))


# --- default behaviour --------------------------------------------------

def test_default_allows_same_domain_http():
    lf = make()
    ok, reason = lf.allows("https://shop.example.com/x", from_url=ORIGIN)
    assert ok and reason == "allowed"


def test_rejects_different_domain():
    lf = make()
    ok, reason = lf.allows("https://other.com/x", from_url=ORIGIN)
    assert not ok and reason == "different_domain"


def test_same_domain_only_false_allows_other_domains():
    lf = make(same_domain_only=False)
    ok, _ = lf.allows("https://other.com/x", from_url=ORIGIN)
    assert ok


def test_rejects_non_http_scheme():
    lf = make(same_domain_only=False)
    ok, reason = lf.allows("ftp://x.com/a", from_url=ORIGIN)
    assert not ok and reason == "scheme_not_allowed"


# --- include list -------------------------------------------------------

def test_include_list_requires_match():
    lf = make(include_patterns=[r"/products/"])
    ok, _ = lf.allows("https://shop.example.com/products/a", from_url=ORIGIN)
    assert ok
    ok, reason = lf.allows("https://shop.example.com/about", from_url=ORIGIN)
    assert not ok and reason == "not_in_include_list"


# --- exclude beats include ---------------------------------------------

def test_exclude_beats_include():
    lf = make(include_patterns=[r"/products/"], exclude_patterns=[r"\.pdf$"])
    ok, reason = lf.allows("https://shop.example.com/products/a.pdf", from_url=ORIGIN)
    assert not ok and reason == "excluded_by_pattern"


# --- depth gate ---------------------------------------------------------

def test_max_depth_gate():
    lf = make(max_depth=2)
    ok, _ = lf.allows("https://shop.example.com/a", from_url=ORIGIN, depth=2)
    assert ok
    ok, reason = lf.allows("https://shop.example.com/a", from_url=ORIGIN, depth=3)
    assert not ok and reason == "max_depth_exceeded"


# --- odd inputs ---------------------------------------------------------

def test_unparseable_url_rejected():
    lf = make(same_domain_only=False)
    ok, _ = lf.allows("::::not a url", from_url=ORIGIN)
    assert not ok


def test_from_url_missing_skips_same_domain_check():
    # Without a from_url, the same-domain check has nothing to compare against.
    lf = make()
    ok, _ = lf.allows("https://anything.com/x", from_url=None)
    assert ok