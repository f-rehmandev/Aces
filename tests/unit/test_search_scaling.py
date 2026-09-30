"""Unit tests for URL scoring in product vs listing mode."""
import pytest

from src.discovery.product_search import (
    _score_url, _rank_urls, _LISTING_PATH_HINTS, _BAD_PATH_HINTS,
)


# ---------------------------------------------------------------------------
# Product mode (default) — deep product pages win
# ---------------------------------------------------------------------------

def test_product_mode_deep_link_wins():
    product = _score_url("https://x.com/product/panadol-500mg-200s/")
    category = _score_url("https://x.com/category/medicine/")
    assert product > category


def test_product_mode_penalizes_category():
    assert _score_url("https://x.com/category/medicine/") < 0


def test_product_mode_penalizes_homepage():
    assert _score_url("https://x.com/") < 0


def test_product_mode_prefers_numeric_id():
    with_id = _score_url("https://x.com/medicine/panadol-24329.html")
    without = _score_url("https://x.com/medicine/panadol.html")
    assert with_id > without


# ---------------------------------------------------------------------------
# Listing mode — category/list pages win
# ---------------------------------------------------------------------------

def test_listing_mode_category_wins():
    category = _score_url("https://x.com/category/medicine/", prefer_listings=True)
    product = _score_url("https://x.com/product/panadol-500mg-200s/", prefer_listings=True)
    assert category > product


def test_listing_mode_boosts_collections():
    s = _score_url("https://x.com/collections/medicine/", prefer_listings=True)
    assert s > 0


def test_listing_mode_boosts_brand_pages():
    s = _score_url("https://x.com/brands/panadol/", prefer_listings=True)
    assert s > 0


def test_listing_mode_boosts_list_pages():
    s = _score_url("https://x.com/list/medicines/", prefer_listings=True)
    assert s > 0


def test_listing_mode_homepage_less_bad():
    product_mode = _score_url("https://x.com/", prefer_listings=False)
    listing_mode = _score_url("https://x.com/", prefer_listings=True)
    assert listing_mode > product_mode
    # Still below a good listing page
    assert listing_mode < _score_url("https://x.com/category/x/", prefer_listings=True)


def test_listing_mode_still_rewards_document_extension():
    with_ext = _score_url("https://x.com/list/medicine.html", prefer_listings=True)
    without = _score_url("https://x.com/list/medicine/", prefer_listings=True)
    assert with_ext > without


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_rank_urls_keeps_top_n():
    urls = [
        "https://x.com/",                                    # -10
        "https://x.com/product/panadol-24329.html",          # high
        "https://x.com/category/medicine/",                  # negative
        "https://x.com/product/panadol-500mg-200s/",         # high
    ]
    top = _rank_urls(urls, keep=2)
    assert "https://x.com/product/panadol-24329.html" in top
    assert "https://x.com/" not in top


def test_rank_urls_listing_mode_prefers_categories():
    urls = [
        "https://x.com/product/panadol-500mg-200s/",
        "https://x.com/category/medicine/",
        "https://x.com/brands/panadol/",
    ]
    top = _rank_urls(urls, keep=2, prefer_listings=True)
    assert "https://x.com/category/medicine/" in top
    assert "https://x.com/brands/panadol/" in top


def test_rank_urls_empty():
    assert _rank_urls([], keep=5) == []


def test_rank_urls_fewer_than_keep():
    urls = ["https://x.com/product/a/", "https://x.com/product/b/"]
    assert len(_rank_urls(urls, keep=10)) == 2


def test_rank_urls_stable_on_ties():
    # Two identical-shaped URLs → original order preserved
    urls = ["https://x.com/product/aaa/", "https://x.com/product/bbb/"]
    ranked = _rank_urls(urls, keep=5)
    assert ranked == urls  # same order, same score


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

def test_listing_hints_cover_common_patterns():
    for pattern in ("/category/", "/collections/", "/brand/", "/list/", "/shop/"):
        assert pattern in _LISTING_PATH_HINTS


def test_bad_hints_still_penalize_blogs():
    assert "/blog" in _BAD_PATH_HINTS
    assert "/about" in _BAD_PATH_HINTS