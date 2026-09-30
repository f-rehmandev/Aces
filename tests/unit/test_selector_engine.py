"""Unit tests for SelectorEngine (spec §13, §62).

Convention (scrapy-style):
    ".title"          -> element(s)   (Tag objects)
    ".title::text"    -> text string(s)
    "a::attr(href)"   -> attribute value(s)
"""
import pytest
from bs4 import Tag

from src.selectors.engine import SelectorEngine
from src.selectors.spec import SelectorSpec, SelectorType, MatchMode


FIXTURE_HTML = """
<html><body>
  <div class="product-card">
    <h2 class="title">Wireless Mouse</h2>
    <span class="price" data-price="24.99">$24.99</span>
  </div>
  <div class="product-card">
    <h2 class="title">Mechanical Keyboard</h2>
    <span class="price" data-price="75.00">$75.00</span>
  </div>
  <div class="product-card">
    <h2 class="title">USB-C Hub</h2>
    <span class="price" data-price="39.50">$39.50</span>
  </div>
  <a class="detail-link" href="/products/mouse">View</a>
</body></html>
"""


@pytest.fixture
def engine():
    return SelectorEngine(FIXTURE_HTML)


# --- CSS: raw element mode --------------------------------------------

def test_css_returns_element(engine):
    spec = SelectorSpec(name="title", expression=".product-card .title")
    r = engine.run(spec)
    assert r.success
    assert isinstance(r.value, Tag)
    assert r.value.get_text(strip=True) == "Wireless Mouse"
    assert r.match_count == 3


def test_css_all_returns_list_of_elements(engine):
    spec = SelectorSpec(name="title", expression=".product-card .title", multiple=MatchMode.ALL)
    r = engine.run(spec)
    assert isinstance(r.value, list)
    assert all(isinstance(x, Tag) for x in r.value)
    assert [x.get_text(strip=True) for x in r.value] == [
        "Wireless Mouse", "Mechanical Keyboard", "USB-C Hub",
    ]


def test_css_nth_returns_element(engine):
    spec = SelectorSpec(
        name="title", expression=".product-card .title",
        multiple=MatchMode.NTH, nth=1,
    )
    r = engine.run(spec)
    assert isinstance(r.value, Tag)
    assert r.value.get_text(strip=True) == "Mechanical Keyboard"


# --- CSS: ::text suffix ------------------------------------------------

def test_css_text_suffix_first(engine):
    spec = SelectorSpec(name="title", expression=".product-card .title::text")
    r = engine.run(spec)
    assert r.value == "Wireless Mouse"


def test_css_text_suffix_all(engine):
    spec = SelectorSpec(
        name="title", expression=".product-card .title::text",
        multiple=MatchMode.ALL,
    )
    r = engine.run(spec)
    assert r.value == ["Wireless Mouse", "Mechanical Keyboard", "USB-C Hub"]


def test_css_text_suffix_join(engine):
    spec = SelectorSpec(
        name="title", expression=".product-card .title::text",
        multiple=MatchMode.JOIN, join_separator=" | ",
    )
    r = engine.run(spec)
    assert r.value == "Wireless Mouse | Mechanical Keyboard | USB-C Hub"


# --- CSS: ::attr() suffix ---------------------------------------------

def test_css_attr_suffix(engine):
    spec = SelectorSpec(name="url", expression="a.detail-link::attr(href)")
    r = engine.run(spec)
    assert r.value == "/products/mouse"


def test_css_attr_shorthand(engine):
    spec = SelectorSpec(name="url", expression="a.detail-link @href")
    r = engine.run(spec)
    assert r.value == "/products/mouse"


# --- TEXT --------------------------------------------------------------

def test_text_extracts_visible_text(engine):
    spec = SelectorSpec(name="price", type=SelectorType.TEXT, expression=".price")
    r = engine.run(spec)
    assert r.value == "$24.99"


def test_text_strips_whitespace():
    html = '<html><body><span class="x">   hello   </span></body></html>'
    e = SelectorEngine(html)
    spec = SelectorSpec(name="x", type=SelectorType.TEXT, expression=".x")
    r = e.run(spec)
    assert r.value == "hello"


# --- ATTRIBUTE ---------------------------------------------------------

def test_attribute_reads_named_attribute(engine):
    spec = SelectorSpec(
        name="price", type=SelectorType.ATTRIBUTE,
        expression=".price", attribute="data-price",
    )
    r = engine.run(spec)
    assert r.value == "24.99"


def test_attribute_requires_attribute_field(engine):
    spec = SelectorSpec(
        name="price", type=SelectorType.ATTRIBUTE,
        expression=".price",
    )
    r = engine.run(spec)
    assert not r.success
    assert r.error and "attribute" in r.error.lower()


# --- FALLBACKS ---------------------------------------------------------

def test_fallback_used_when_primary_fails(engine):
    primary = SelectorSpec(name="x", expression=".nope::text")
    primary.fallback_selectors.append(SelectorSpec(name="x", expression=".title::text"))
    r = engine.run(primary)
    assert r.success
    assert r.rung_used == 2
    assert r.value == "Wireless Mouse"


def test_fallback_chain_stops_at_first_success(engine):
    primary = SelectorSpec(name="x", expression=".nope::text")
    primary.fallback_selectors.append(SelectorSpec(name="x", expression=".also-nope::text"))
    primary.fallback_selectors.append(SelectorSpec(name="x", expression=".title::text"))
    primary.fallback_selectors.append(SelectorSpec(name="x", expression=".price::text"))
    r = engine.run(primary)
    assert r.rung_used == 3
    assert r.value == "Wireless Mouse"


def test_total_failure_returns_unsuccessful(engine):
    spec = SelectorSpec(name="x", expression=".does-not-exist")
    r = engine.run(spec)
    assert not r.success
    assert r.value is None
    assert r.match_count == 0

    

# =====================================================================
# Round 3: extended types
# =====================================================================

EXTENDED_HTML = """
<html><head>
  <meta property="og:price:amount" content="24.99">
  <meta name="description" content="Buy the best mouse.">
  <script type="application/ld+json">
  {"@type": "Product", "name": "Wireless Mouse",
   "offers": {"@type": "Offer", "price": "24.99", "priceCurrency": "USD"}}
  </script>
  <script id="__NEXT_DATA__" type="application/json">
  {"props": {"pageProps": {"products": [
    {"sku": "A1", "title": "Mouse"}, {"sku": "B2", "title": "Keyboard"}
  ]}}}
  </script>
</head><body>
  <img class="thumb" src="/img/mouse.jpg">
  <img class="thumb" data-src="/img/lazy.jpg">
  <a class="detail" href="/products/mouse">View</a>
  <a class="detail" href="/products/kb">View</a>
  <a rel="next" href="/page/2">Next</a>
  <table class="specs">
    <thead><tr><th>Key</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Weight</td><td>90g</td></tr>
      <tr><td>Battery</td><td>AA</td></tr>
    </tbody>
  </table>
  <p class="raw">Price: 24.99 and discount: 5.00</p>
</body></html>
"""


@pytest.fixture
def xengine():
    return SelectorEngine(EXTENDED_HTML)


# --- JSON-LD ----------------------------------------------------------

def test_jsonld_simple_path(xengine):
    spec = SelectorSpec(name="price", type=SelectorType.JSONLD, expression="offers.price")
    r = xengine.run(spec)
    assert r.value == "24.99"


def test_jsonld_top_level_field(xengine):
    spec = SelectorSpec(name="name", type=SelectorType.JSONLD, expression="name")
    r = xengine.run(spec)
    assert r.value == "Wireless Mouse"


def test_jsonld_missing_path_returns_failure(xengine):
    spec = SelectorSpec(name="x", type=SelectorType.JSONLD, expression="offers.nope")
    r = xengine.run(spec)
    assert not r.success


# --- Embedded JSON ----------------------------------------------------

def test_embedded_json_next_data_path(xengine):
    spec = SelectorSpec(
        name="sku", type=SelectorType.EMBEDDED_JSON,
        expression="props.pageProps.products.0.sku",
    )
    r = xengine.run(spec)
    assert r.value == "A1"


def test_embedded_json_missing_returns_failure(xengine):
    spec = SelectorSpec(
        name="x", type=SelectorType.EMBEDDED_JSON,
        expression="props.nope.deep",
    )
    r = xengine.run(spec)
    assert not r.success


# --- Meta -------------------------------------------------------------

def test_meta_by_property(xengine):
    spec = SelectorSpec(name="price", type=SelectorType.META, expression="og:price:amount")
    r = xengine.run(spec)
    assert r.value == "24.99"


def test_meta_by_name(xengine):
    spec = SelectorSpec(name="desc", type=SelectorType.META, expression="description")
    r = xengine.run(spec)
    assert r.value == "Buy the best mouse."


def test_meta_missing(xengine):
    spec = SelectorSpec(name="x", type=SelectorType.META, expression="og:nope")
    r = xengine.run(spec)
    assert not r.success


# --- Regex ------------------------------------------------------------

def test_regex_capture_group(xengine):
    spec = SelectorSpec(name="num", type=SelectorType.REGEX, expression=r"Price: (\d+\.\d{2})")
    r = xengine.run(spec)
    assert r.value == "24.99"


def test_regex_all_matches(xengine):
    spec = SelectorSpec(
        name="num", type=SelectorType.REGEX,
        expression=r"(\d+\.\d{2})",
        multiple=MatchMode.ALL,
    )
    r = xengine.run(spec)
    assert "24.99" in r.value and "5.00" in r.value


def test_regex_no_match(xengine):
    spec = SelectorSpec(name="x", type=SelectorType.REGEX, expression=r"NEVER_MATCHES_XYZ")
    r = xengine.run(spec)
    assert not r.success


# --- Image ------------------------------------------------------------

def test_image_returns_src(xengine):
    spec = SelectorSpec(
        name="img", type=SelectorType.IMAGE, expression="img.thumb",
        multiple=MatchMode.ALL,
    )
    r = xengine.run(spec)
    assert r.value == ["/img/mouse.jpg", "/img/lazy.jpg"]


def test_image_falls_back_to_data_src(xengine):
    spec = SelectorSpec(name="img", type=SelectorType.IMAGE, expression="img.thumb:nth-of-type(2)")
    r = xengine.run(spec)
    assert r.value == "/img/lazy.jpg"


# --- Link -------------------------------------------------------------

def test_link_returns_href(xengine):
    spec = SelectorSpec(name="url", type=SelectorType.LINK, expression="a.detail")
    r = xengine.run(spec)
    assert r.value == "/products/mouse"


def test_link_all(xengine):
    spec = SelectorSpec(
        name="url", type=SelectorType.LINK, expression="a.detail",
        multiple=MatchMode.ALL,
    )
    r = xengine.run(spec)
    assert r.value == ["/products/mouse", "/products/kb"]


# --- Pagination -------------------------------------------------------

def test_pagination_explicit_selector(xengine):
    spec = SelectorSpec(name="next", type=SelectorType.PAGINATION, expression="a[rel='next']")
    r = xengine.run(spec)
    assert r.value == "/page/2"


def test_pagination_default_selector(xengine):
    spec = SelectorSpec(name="next", type=SelectorType.PAGINATION)
    r = xengine.run(spec)
    assert r.value == "/page/2"


# --- Table ------------------------------------------------------------

def test_table_returns_row_dicts(xengine):
    spec = SelectorSpec(
        name="rows", type=SelectorType.TABLE, expression="table.specs",
        multiple=MatchMode.ALL,
    )
    r = xengine.run(spec)
    assert r.value == [
        {"Key": "Weight", "Value": "90g"},
        {"Key": "Battery", "Value": "AA"},
    ]


# --- Deferred types raise clearly -------------------------------------

def test_url_type_raises_unsupported(xengine):
    spec = SelectorSpec(name="x", type=SelectorType.URL, expression="path[2]")
    r = xengine.run(spec)
    assert not r.success
    assert "url" in (r.error or "").lower()


def test_interaction_type_raises_unsupported(xengine):
    spec = SelectorSpec(name="x", type=SelectorType.INTERACTION, expression="click .load-more")
    r = xengine.run(spec)
    assert not r.success
    assert "interaction" in (r.error or "").lower()



# =====================================================================
# Round 4: SelectorTree
# =====================================================================

from src.selectors.tree import SelectorTree


LIST_HTML = """
<html><body>
  <div class="card">
    <h2 class="title">Wireless Mouse</h2>
    <span class="price">$24.99</span>
    <a class="detail" href="/p/mouse">View</a>
  </div>
  <div class="card">
    <h2 class="title">Mechanical Keyboard</h2>
    <span class="price">$75.00</span>
    <a class="detail" href="/p/keyboard">View</a>
  </div>
  <div class="card">
    <h2 class="title">USB-C Hub</h2>
    <span class="price">$39.50</span>
    <a class="detail" href="/p/hub">View</a>
  </div>
</body></html>
"""


SINGLE_HTML = """
<html><body>
  <div class="page">
    <h1 class="name">Tony's Pizza</h1>
    <span class="phone">0300-1234567</span>
    <a class="site" href="https://tonys.example">Website</a>
  </div>
</body></html>
"""


NESTED_HTML = """
<html><body>
  <div class="product">
    <h2 class="title">Laptop</h2>
    <ul class="reviews">
      <li class="review"><span class="who">Alice</span><span class="stars">5</span></li>
      <li class="review"><span class="who">Bob</span><span class="stars">4</span></li>
    </ul>
  </div>
  <div class="product">
    <h2 class="title">Tablet</h2>
    <ul class="reviews">
      <li class="review"><span class="who">Carol</span><span class="stars">3</span></li>
    </ul>
  </div>
</body></html>
"""


def _list_tree() -> SelectorTree:
    t = SelectorTree()
    t.add(SelectorSpec(name="product", expression=".card", multiple=MatchMode.ALL))
    t.add(SelectorSpec(name="title", expression=".title::text"), parent="product")
    t.add(SelectorSpec(name="price", expression=".price::text"), parent="product")
    t.add(SelectorSpec(
        name="url", type=SelectorType.LINK, expression="a.detail",
    ), parent="product")
    return t


def test_tree_list_mode_produces_one_record_per_card():
    records = _list_tree().extract(LIST_HTML)
    assert len(records) == 3
    assert records[0] == {"title": "Wireless Mouse", "price": "$24.99", "url": "/p/mouse"}
    assert records[1]["title"] == "Mechanical Keyboard"
    assert records[2]["url"] == "/p/hub"


def test_tree_single_record_mode():
    t = SelectorTree()
    t.add(SelectorSpec(name="name", expression=".name::text"))
    t.add(SelectorSpec(name="phone", expression=".phone::text"))
    t.add(SelectorSpec(name="url", type=SelectorType.LINK, expression="a.site"))
    records = t.extract(SINGLE_HTML)
    assert len(records) == 1
    assert records[0] == {
        "name": "Tony's Pizza",
        "phone": "0300-1234567",
        "url": "https://tonys.example",
    }


def test_tree_nested_list_inside_record():
    t = SelectorTree()
    t.add(SelectorSpec(name="product", expression=".product", multiple=MatchMode.ALL))
    t.add(SelectorSpec(name="title", expression=".title::text"), parent="product")
    t.add(SelectorSpec(name="review", expression=".review", multiple=MatchMode.ALL), parent="product")
    t.add(SelectorSpec(name="who", expression=".who::text"), parent="review")
    t.add(SelectorSpec(name="stars", expression=".stars::text"), parent="review")

    records = t.extract(NESTED_HTML)
    assert len(records) == 2
    assert records[0]["title"] == "Laptop"
    assert records[0]["review"] == [
        {"who": "Alice", "stars": "5"},
        {"who": "Bob", "stars": "4"},
    ]
    assert records[1]["review"] == [{"who": "Carol", "stars": "3"}]


def test_tree_missing_field_is_none_not_crash():
    html = """
    <html><body>
      <div class="card"><h2 class="title">Only Title</h2></div>
    </body></html>
    """
    records = _list_tree().extract(html)
    assert records == [{"title": "Only Title", "price": None, "url": None}]


def test_tree_no_matches_returns_empty_list():
    records = _list_tree().extract("<html><body></body></html>")
    assert records == []


def test_tree_field_fallback_inside_card():
    html = """
    <html><body>
      <div class="card">
        <h2 class="title">X</h2>
        <span data-price="10">£10</span>
      </div>
    </body></html>
    """
    t = SelectorTree()
    t.add(SelectorSpec(name="product", expression=".card", multiple=MatchMode.ALL))

    price_spec = SelectorSpec(name="price", expression=".price::text")
    price_spec.fallback_selectors.append(
        SelectorSpec(name="price", expression="[data-price]::attr(data-price)")
    )
    t.add(price_spec, parent="product")

    records = t.extract(html)
    assert records == [{"price": "10"}]


def test_tree_duplicate_name_raises():
    t = SelectorTree()
    t.add(SelectorSpec(name="x", expression=".a"))
    with pytest.raises(ValueError):
        t.add(SelectorSpec(name="x", expression=".b"))


def test_tree_missing_parent_raises():
    t = SelectorTree()
    with pytest.raises(KeyError):
        t.add(SelectorSpec(name="child", expression=".c"), parent="nonexistent")


def test_tree_dict_round_trip():
    original = _list_tree()
    rebuilt = SelectorTree.from_dict(original.to_dict())
    assert rebuilt.extract(LIST_HTML) == original.extract(LIST_HTML)


def test_tree_from_dict_preserves_fallback_chain():
    t = SelectorTree()
    t.add(SelectorSpec(name="product", expression=".card", multiple=MatchMode.ALL))
    price_spec = SelectorSpec(name="price", expression=".price::text")
    price_spec.fallback_selectors.append(
        SelectorSpec(name="price", expression=".alt-price::text")
    )
    t.add(price_spec, parent="product")

    rebuilt = SelectorTree.from_dict(t.to_dict())
    price_node = rebuilt.get("price")
    assert len(price_node.spec.fallback_selectors) == 1
    assert price_node.spec.fallback_selectors[0].expression == ".alt-price::text"