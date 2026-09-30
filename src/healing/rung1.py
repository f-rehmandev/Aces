"""
Rung 1 — deterministic extraction (spec §17.3).

Before spending an LLM call, look for the data in places the page itself
declares as structured:

    1. JSON-LD (schema.org Product, Offer, ItemList, etc.)
    2. Meta tags (OpenGraph, product:*)
    3. Regex over raw HTML for common fields

Any hit here costs ~0 and is more reliable than an LLM guess because it
comes from the site's own structured markup.

Returns a list of records (may be empty). Never raises — every sub-strategy
is tried in order and any failure moves on to the next.
"""

from __future__ import annotations
import json
import re
from typing import Any, Optional

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Field-name → structured-data mapping
# ---------------------------------------------------------------------------
# Left side is the caller's field name (lowercase). Right side is a tuple of
# candidate paths within a JSON-LD object. The first path that resolves wins.

_JSONLD_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "title":            ("name", "headline"),
    "product_name":     ("name",),
    "name":             ("name",),
    "price":            ("offers.price", "offers.0.price", "price"),
    "cost":             ("offers.price", "offers.0.price"),
    "currency":         ("offers.priceCurrency", "offers.0.priceCurrency", "priceCurrency"),
    "availability":     ("offers.availability", "offers.0.availability", "availability"),
    "product_url":      ("url", "offers.url", "mainEntityOfPage"),
    "url":              ("url",),
    "image":            ("image", "image.url", "offers.image"),
    "image_url":        ("image", "image.url"),
    "description":      ("description",),
    "brand":            ("brand.name", "brand"),
    "manufacturer":     ("manufacturer.name", "manufacturer"),
    "rating":           ("aggregateRating.ratingValue", "aggregateRating"),
    "review_count":     ("aggregateRating.reviewCount",),
    "sku":              ("sku", "productID"),
    "gtin":             ("gtin13", "gtin", "gtin8", "gtin12"),
    "category":         ("category",),
    "quantity":         ("size", "additionalProperty.value"),
}

# Meta tag candidates: field name -> list of (attr, value) tuples
_META_FIELD_PATHS: dict[str, tuple[tuple[str, str], ...]] = {
    "title":        (("property", "og:title"), ("name", "twitter:title"), ("name", "title")),
    "product_name": (("property", "og:title"),),
    "name":         (("property", "og:title"),),
    "price":        (("property", "product:price:amount"), ("property", "og:price:amount")),
    "currency":     (("property", "product:price:currency"), ("property", "og:price:currency")),
    "image":        (("property", "og:image"), ("name", "twitter:image")),
    "image_url":    (("property", "og:image"),),
    "description":  (("property", "og:description"), ("name", "description")),
    "product_url":  (("property", "og:url"),),
    "url":          (("property", "og:url"),),
    "availability": (("property", "product:availability"), ("property", "og:availability")),
}

# Regex fallbacks (only used when JSON-LD + meta both miss)
_PRICE_RE = re.compile(
    r"(?:(?:USD|PKR|EUR|GBP|INR|AED|SAR|CNY|CAD|AUD|Rs\.?|₨|₹|£|€|\$)\s*)?"
    r"(\d{1,3}(?:[,]\d{3})*(?:\.\d{1,2})?)"
    r"\s*(?:USD|PKR|EUR|GBP|INR|AED|SAR|CNY|CAD|AUD|Rs\.?|₨|₹|£|€|\$)?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# JSON-LD walking
# ---------------------------------------------------------------------------

def _walk_jsonld(obj: Any, path: str) -> Any:
    """
    Walk a dotted path. Numeric segments index into lists. `@type` is
    preferred when a list contains mixed-typed objects.
    """
    segments = path.split(".")
    current = obj
    i = 0
    while i < len(segments):
        seg = segments[i]
        if current is None:
            return None
        if isinstance(current, list):
            if seg.isdigit():
                idx = int(seg)
                current = current[idx] if 0 <= idx < len(current) else None
                i += 1
                continue
            match = next(
                (x for x in current
                 if isinstance(x, dict) and x.get("@type") == seg),
                None,
            )
            if match is not None:
                current = match
                i += 1
                continue
            if current:
                current = current[0]
                continue
            return None
        if isinstance(current, dict):
            # Accept both exact and case-insensitive key match
            if seg in current:
                current = current[seg]
                i += 1
                continue
            lower_match = next(
                (k for k in current if k.lower() == seg.lower()), None,
            )
            if lower_match is not None:
                current = current[lower_match]
                i += 1
                continue
            return None
        return None
    return current


def _extract_jsonld_blocks(html: str) -> list[Any]:
    """Return every parseable JSON-LD block from the page."""
    soup = BeautifulSoup(html, "lxml")
    blocks: list[Any] = []
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        raw = script.string
        if not raw:
            continue
        try:
            blocks.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            continue
    return blocks


def _products_from_jsonld(blocks: list[Any]) -> list[dict]:
    """
    Extract product-shaped dicts from a list of JSON-LD blocks.

    Handles three common shapes:
        - A single `Product` object
        - A list containing `Product` objects
        - An `ItemList` with `itemListElement` -> `item` -> `Product`
    """
    out: list[dict] = []

    def _collect(obj: Any) -> None:
        if isinstance(obj, list):
            for item in obj:
                _collect(item)
            return
        if not isinstance(obj, dict):
            return
        t = obj.get("@type")
        if isinstance(t, list):
            t = t[0] if t else None
        if t == "Product":
            out.append(obj)
        elif t == "ItemList":
            for el in obj.get("itemListElement", []) or []:
                item = el.get("item") if isinstance(el, dict) else None
                if isinstance(item, dict):
                    _collect(item)
        elif "@graph" in obj:
            _collect(obj["@graph"])

    for block in blocks:
        _collect(block)
    return out


# ---------------------------------------------------------------------------
# Meta tag reading
# ---------------------------------------------------------------------------

def _meta_value(soup: BeautifulSoup, attr: str, value: str) -> Optional[str]:
    el = soup.find("meta", {attr: value})
    if el is None:
        # Also try case-insensitive on the attribute name
        for meta in soup.find_all("meta"):
            a = meta.get(attr) or meta.get(attr.lower())
            if a and a.lower() == value.lower():
                el = meta
                break
    if el is None:
        return None
    content = el.get("content")
    return str(content).strip() if content else None


# ---------------------------------------------------------------------------
# Price normalization
# ---------------------------------------------------------------------------

def _normalize_price_string(s: str) -> Optional[float]:
    if not s:
        return None
    m = _PRICE_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def try_deterministic_extraction(
    html: str,
    fields: list[str],
    url: str = "",
) -> list[dict]:
    """
    Attempt to extract records from structured data alone.

    Returns a list of records (possibly a single record) or [] if the
    page has no usable structured data. Never raises.
    """
    if not html or not fields:
        return []

    try:
        blocks = _extract_jsonld_blocks(html)
    except Exception:
        blocks = []

    products = _products_from_jsonld(blocks) if blocks else []

    # --- Strategy 1: JSON-LD Product(s) ---
    if products:
        records: list[dict] = []
        for product in products:
            rec = _record_from_jsonld(product, fields, url)
            if rec and any(v not in (None, "", []) for v in rec.values()):
                records.append(rec)
        if records:
            for r in records:
                r["_extraction_method"] = "rung1_jsonld"
            return records

    # --- Strategy 2: meta tags ---
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return []

    meta_rec: dict = {}
    for field in fields:
        candidates = _META_FIELD_PATHS.get(field.lower(), ())
        for attr, value in candidates:
            v = _meta_value(soup, attr, value)
            if v:
                # Route through the same coercion JSON-LD uses so price
                # strings become floats, availability dicts unwrap, etc.
                meta_rec[field] = _coerce_value(field, v)
                break

    if meta_rec:
        # Meta tags are page-level, not per-product — only emit a record if
        # at least one *data* field (not just title) is present.
        data_fields_present = any(
            f.lower() not in ("title", "description", "image", "image_url")
            for f in meta_rec
        )
        if data_fields_present:
            if url:
                meta_rec.setdefault("source_url", url)
            meta_rec["_extraction_method"] = "rung1_meta"
            return [meta_rec]

    return []


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def _record_from_jsonld(product: dict, fields: list[str], url: str) -> dict:
    rec: dict = {}
    for field in fields:
        candidates = _JSONLD_FIELD_PATHS.get(field.lower(), ())
        value = None
        for path in candidates:
            value = _walk_jsonld(product, path)
            if value not in (None, "", []):
                break
        rec[field] = _coerce_value(field, value)
    if url:
        rec.setdefault("source_url", url)
    return rec


def _coerce_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
        if value is None:
            return None
    if isinstance(value, dict):
        # e.g. availability sometimes appears as {"@type": "ItemAvailability", ...}
        if "@id" in value:
            value = value["@id"]
        elif "name" in value:
            value = value["name"]
        else:
            return None

    field_lower = field.lower()
    if "price" in field_lower or "cost" in field_lower:
        return _normalize_price_string(str(value)) if not isinstance(value, (int, float)) else float(value)

    return str(value).strip() if value is not None else None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- JSON-LD product ---
    html_product = """
    <html><head>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": "Panadol 500mg Tablets",
        "sku": "PAN-500-200",
        "image": "https://example.com/panadol.jpg",
        "description": "Pain relief tablets",
        "brand": {"@type": "Brand", "name": "Panadol"},
        "offers": {
          "@type": "Offer",
          "price": "400.00",
          "priceCurrency": "PKR",
          "availability": "https://schema.org/InStock",
          "url": "https://example.com/p/panadol-500"
        },
        "aggregateRating": {"@type": "AggregateRating", "ratingValue": "4.5", "reviewCount": "120"}
      }
      </script>
    </head><body></body></html>
    """
    fields = ["title", "price", "currency", "availability", "brand",
              "rating", "review_count", "sku", "product_url"]
    records = try_deterministic_extraction(html_product, fields,
                                            url="https://example.com/p/panadol")
    assert len(records) == 1
    r = records[0]
    assert r["title"] == "Panadol 500mg Tablets"
    assert r["price"] == 400.0
    assert r["currency"] == "PKR"
    assert r["brand"] == "Panadol"
    assert r["rating"] == "4.5"
    assert r["review_count"] == "120"
    assert r["sku"] == "PAN-500-200"
    assert r["_extraction_method"] == "rung1_jsonld"

    # --- ItemList with items ---
    html_list = """
    <html><head>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "ItemList",
        "itemListElement": [
          {"@type": "ListItem", "position": 1, "item": {
            "@type": "Product", "name": "A",
            "offers": {"@type": "Offer", "price": "10.00", "priceCurrency": "USD"}
          }},
          {"@type": "ListItem", "position": 2, "item": {
            "@type": "Product", "name": "B",
            "offers": {"@type": "Offer", "price": "20.00", "priceCurrency": "USD"}
          }}
        ]
      }
      </script>
    </head></html>
    """
    records = try_deterministic_extraction(html_list, ["title", "price"])
    assert len(records) == 2
    assert records[0]["title"] == "A" and records[0]["price"] == 10.0
    assert records[1]["title"] == "B" and records[1]["price"] == 20.0

    # --- Meta tags only ---
    html_meta = """
    <html><head>
      <meta property="og:title" content="Panadol 500mg">
      <meta property="product:price:amount" content="250.00">
      <meta property="product:price:currency" content="PKR">
      <meta property="product:availability" content="in stock">
    </head></html>
    """
    records = try_deterministic_extraction(html_meta,
                                            ["title", "price", "currency", "availability"])
    assert len(records) == 1
    r = records[0]
    assert r["title"] == "Panadol 500mg"
    assert r["price"] == 250.0
    assert r["currency"] == "PKR"
    assert r["_extraction_method"] == "rung1_meta"

    # --- Meta tags with only title -> not emitted ---
    html_title_only = """
    <html><head><meta property="og:title" content="Something"></head></html>
    """
    assert try_deterministic_extraction(html_title_only, ["title", "price"]) == []

    # --- Nothing structured ---
    assert try_deterministic_extraction("<html><body>plain</body></html>",
                                         ["title", "price"]) == []

    # --- Empty input ---
    assert try_deterministic_extraction("", ["title"]) == []
    assert try_deterministic_extraction("<html></html>", []) == []

    # --- Non-dict availability value ---
    html_bad_avail = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Product", "name": "X",
       "offers": {"price": "5.00",
                  "availability": {"@type": "ItemAvailability", "@id": "https://schema.org/InStock"}}}
      </script>
    </head></html>
    """
    recs = try_deterministic_extraction(html_bad_avail, ["title", "price", "availability"])
    assert recs[0]["availability"] == "https://schema.org/InStock"

    print("Rung 1 deterministic extraction OK.")
