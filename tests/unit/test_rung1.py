"""Unit tests for Rung 1 deterministic extraction (spec §17.3)."""
import pytest

from src.healing.rung1 import (
    try_deterministic_extraction,
    _extract_jsonld_blocks,
    _products_from_jsonld,
    _walk_jsonld,
    _normalize_price_string,
)


# ---------------------------------------------------------------------------
# JSON-LD helpers
# ---------------------------------------------------------------------------

def test_walk_simple_dict():
    assert _walk_jsonld({"name": "A"}, "name") == "A"


def test_walk_nested():
    assert _walk_jsonld({"offers": {"price": "10"}}, "offers.price") == "10"


def test_walk_missing_returns_none():
    assert _walk_jsonld({"name": "A"}, "offers.price") is None


def test_walk_list_by_index():
    assert _walk_jsonld({"offers": [{"price": "5"}]}, "offers.0.price") == "5"


def test_walk_list_prefers_type_match():
    obj = {"items": [
        {"@type": "X", "name": "wrong"},
        {"@type": "Y", "name": "right"},
    ]}
    assert _walk_jsonld(obj, "items.Y.name") == "right"


def test_walk_case_insensitive_key():
    assert _walk_jsonld({"Name": "A"}, "name") == "A"


def test_walk_non_dict_scalar_returns_none():
    assert _walk_jsonld("scalar", "any.path") is None


# ---------------------------------------------------------------------------
# JSON-LD block parsing
# ---------------------------------------------------------------------------

def test_extract_blocks_one():
    html = '<script type="application/ld+json">{"a": 1}</script>'
    blocks = _extract_jsonld_blocks(html)
    assert blocks == [{"a": 1}]


def test_extract_blocks_malformed_skipped():
    html = '<script type="application/ld+json">not json</script>'
    assert _extract_jsonld_blocks(html) == []


def test_products_from_graph():
    block = {"@graph": [{"@type": "Product", "name": "A"}]}
    products = _products_from_jsonld([block])
    assert len(products) == 1


# ---------------------------------------------------------------------------
# Price normalization
# ---------------------------------------------------------------------------

def test_normalize_price_plain():
    assert _normalize_price_string("10.50") == 10.50


def test_normalize_price_with_symbol():
    assert _normalize_price_string("$24.99") == 24.99


def test_normalize_price_with_code():
    assert _normalize_price_string("PKR 1,299.00") == 1299.0


def test_normalize_price_empty():
    assert _normalize_price_string("") is None
    assert _normalize_price_string(None) is None


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------

def test_product_jsonld_full():
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@type": "Product", "name": "Widget", "sku": "W-1",
     "offers": {"price": "12.99", "priceCurrency": "USD",
                "availability": "InStock"},
     "brand": {"name": "Acme"}}
    </script>
    </head></html>
    """
    fields = ["title", "price", "currency", "availability", "brand", "sku"]
    recs = try_deterministic_extraction(html, fields,
                                        url="https://x.example/w")
    assert len(recs) == 1
    r = recs[0]
    assert r["title"] == "Widget"
    assert r["price"] == 12.99
    assert r["currency"] == "USD"
    assert r["availability"] == "InStock"
    assert r["brand"] == "Acme"
    assert r["sku"] == "W-1"
    assert r["source_url"] == "https://x.example/w"
    assert r["_extraction_method"] == "rung1_jsonld"


def test_itemlist_yields_multiple_records():
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@type": "ItemList", "itemListElement": [
      {"item": {"@type": "Product", "name": "A",
                 "offers": {"price": "1.00"}}},
      {"item": {"@type": "Product", "name": "B",
                 "offers": {"price": "2.00"}}}
    ]}
    </script>
    </head></html>
    """
    recs = try_deterministic_extraction(html, ["title", "price"])
    assert [r["title"] for r in recs] == ["A", "B"]
    assert [r["price"] for r in recs] == [1.0, 2.0]


def test_meta_only_fallback():
    html = """
    <html><head>
    <meta property="og:title" content="Panadol 500mg">
    <meta property="product:price:amount" content="250">
    <meta property="product:price:currency" content="PKR">
    </head></html>
    """
    recs = try_deterministic_extraction(html, ["title", "price", "currency"])
    assert len(recs) == 1
    r = recs[0]
    assert r["_extraction_method"] == "rung1_meta"
    # Price coercion must produce a float, not a string
    assert r["price"] == 250.0
    assert isinstance(r["price"], float)
    assert r["title"] == "Panadol 500mg"
    assert r["currency"] == "PKR"


def test_meta_price_with_symbol_normalized():
    html = """
    <html><head>
    <meta property="og:title" content="Widget">
    <meta property="product:price:amount" content="PKR 1,299.00">
    </head></html>
    """
    recs = try_deterministic_extraction(html, ["title", "price"])
    assert recs[0]["price"] == 1299.0


def test_meta_with_only_title_is_skipped():
    html = '<html><head><meta property="og:title" content="X"></head></html>'
    assert try_deterministic_extraction(html, ["title", "price"]) == []


def test_no_structured_data_returns_empty():
    assert try_deterministic_extraction(
        "<html><body>nothing</body></html>", ["title", "price"],
    ) == []


def test_empty_inputs():
    assert try_deterministic_extraction("", ["title"]) == []
    assert try_deterministic_extraction("<html></html>", []) == []


def test_jsonld_wins_over_meta():
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@type": "Product", "name": "FromLD", "offers": {"price": "5.00"}}
    </script>
    <meta property="og:title" content="FromMeta">
    <meta property="product:price:amount" content="999">
    </head></html>
    """
    recs = try_deterministic_extraction(html, ["title", "price"])
    assert recs[0]["title"] == "FromLD"
    assert recs[0]["price"] == 5.0