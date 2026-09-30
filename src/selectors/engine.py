"""
Selector engine — spec §13. Executes a SelectorSpec against real HTML.

Supports a `scope` argument so a child selector can run inside the DOM
returned by its parent (§13.3 — parent-child context).

Supported types:
    css, text, attribute, xpath, html,
    jsonld, embedded_json, meta, regex,
    image, link, table, pagination

Deferred (handled elsewhere):
    url          -> derived from the record's own URL, done by the record engine
    interaction  -> needs a live browser, done by the navigation engine
"""

from __future__ import annotations
import json
import re
from typing import Optional, Any
from bs4 import BeautifulSoup, Tag
from lxml import etree

from src.selectors.spec import (
    SelectorSpec, SelectorType, MatchMode, SelectorResult,
)


class UnsupportedSelectorType(NotImplementedError):
    """Raised when the engine encounters a type handled by another layer."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_walk(container: Any, segments: list[str]) -> Any:
    """
    Walk a dotted JSON path.

    List handling:
        - "N" (digit)  -> index into the list
        - otherwise    -> prefer element whose @type == segment
        - else         -> descend into first element WITHOUT consuming the
                          segment (it may still address a key inside that item)
    """
    current = container
    i = 0
    while i < len(segments):
        seg = segments[i]
        if current is None:
            return None

        # --- list ---
        if isinstance(current, list):
            if seg.isdigit():
                idx = int(seg)
                if 0 <= idx < len(current):
                    current = current[idx]
                    i += 1
                    continue
                return None

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

        # --- dict ---
        if isinstance(current, dict):
            if seg in current:
                current = current[seg]
                i += 1
                continue
            return None

        # --- scalar ---
        return None

    return current


class SelectorEngine:
    """Executes SelectorSpec objects against parsed HTML."""

    def __init__(self, html: str):
        self.html = html
        self._soup: Optional[BeautifulSoup] = None
        self._lxml_root = None

    # ------------------------------------------------------------------
    # Lazy parsing
    # ------------------------------------------------------------------
    @property
    def soup(self) -> BeautifulSoup:
        if self._soup is None:
            self._soup = BeautifulSoup(self.html, "lxml")
        return self._soup

    @property
    def lxml_root(self):
        if self._lxml_root is None:
            self._lxml_root = etree.HTML(self.html)
        return self._lxml_root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, spec: SelectorSpec, scope: Optional[Tag] = None) -> SelectorResult:
        attempts: list[tuple[int, SelectorSpec]] = [(1, spec)]
        for i, fb in enumerate(spec.fallback_selectors, start=2):
            attempts.append((i, fb))

        last_error: Optional[str] = None
        for rung, current in attempts:
            try:
                matches = self._execute(current, scope=scope)
            except UnsupportedSelectorType as e:
                last_error = f"unsupported type: {e}"
                continue
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                continue

            if matches:
                value = self._apply_match_mode(current, matches)
                return SelectorResult(
                    selector_id=current.id,
                    field_name=current.name,
                    success=True,
                    value=value,
                    match_count=len(matches),
                    rung_used=rung,
                    selector_used=current.expression,
                )

        return SelectorResult(
            selector_id=spec.id,
            field_name=spec.name,
            success=False,
            value=None,
            match_count=0,
            rung_used=0,
            error=last_error or "no matches",
        )

    # ------------------------------------------------------------------
    # Dispatch by type
    # ------------------------------------------------------------------
    def _execute(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        handler = getattr(self, f"_run_{spec.type.value}", None)
        if handler is None:
            raise UnsupportedSelectorType(spec.type.value)
        return handler(spec, scope)

    # ------------------------------------------------------------------
    # CSS family
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_css_suffix(expr: str) -> tuple[str, Optional[str], Optional[str]]:
        expr = expr.strip()
        if " @" in expr:
            css, _, attr = expr.rpartition(" @")
            return css.strip(), "attr", attr.strip()
        if "::attr(" in expr:
            css, _, tail = expr.partition("::attr(")
            attr = tail.rstrip(")").strip()
            return css.strip(), "attr", attr
        if expr.endswith("::text"):
            return expr[: -len("::text")].strip(), "text", None
        return expr, None, None

    def _run_css(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        css, mode, extra = self._parse_css_suffix(spec.expression)
        root = scope if scope is not None else self.soup
        elements = root.select(css)

        if mode is None:
            return elements
        if mode == "text":
            texts = [el.get_text(strip=True) for el in elements]
            return [t for t in texts if t]
        if mode == "attr":
            values = []
            for el in elements:
                v = el.get(extra)
                if v:
                    values.append(v if isinstance(v, str) else " ".join(v))
            return values
        return elements

    def _run_text(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        expr = spec.expression
        if expr.endswith("::text"):
            expr = expr[: -len("::text")]
        root = scope if scope is not None else self.soup
        elements = root.select(expr)
        texts = [el.get_text(strip=True) for el in elements]
        return [t for t in texts if t]

    def _run_attribute(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        if not spec.attribute:
            raise ValueError("ATTRIBUTE selector requires `attribute` to be set")
        expr = spec.expression
        attr = spec.attribute
        if "@" in expr and not expr.strip().startswith("["):
            expr, _, shorthand = expr.rpartition("@")
            attr = shorthand or attr
        root = scope if scope is not None else self.soup
        elements = root.select(expr)
        values = []
        for el in elements:
            v = el.get(attr)
            if v:
                values.append(v if isinstance(v, str) else " ".join(v))
        return values

    def _run_xpath(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        if scope is not None:
            node = etree.fromstring(str(scope))
        else:
            node = self.lxml_root
        raw = node.xpath(spec.expression)
        results = []
        for r in raw:
            if isinstance(r, etree._ElementUnicodeResult):
                results.append(str(r))
            elif isinstance(r, etree._Element):
                text = (r.text or "").strip()
                if text:
                    results.append(text)
            else:
                results.append(str(r))
        return results

    def _run_html(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        root = scope if scope is not None else self.soup
        elements = root.select(spec.expression)
        return [str(el) for el in elements]

    # ------------------------------------------------------------------
    # JSON-LD
    # ------------------------------------------------------------------
    def _run_jsonld(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        root = scope if scope is not None else self.soup
        scripts = root.find_all("script", {"type": "application/ld+json"})
        blocks = []
        for s in scripts:
            raw = s.string
            if not raw:
                continue
            try:
                blocks.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                continue
        if not blocks:
            return []
        container = blocks[0] if len(blocks) == 1 else blocks
        if not spec.expression:
            return [json.dumps(container)]
        value = _json_walk(container, spec.expression.split("."))
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) if not isinstance(v, str) else v for v in value]
        return [str(value)]

    # ------------------------------------------------------------------
    # Embedded JSON
    # ------------------------------------------------------------------
    def _run_embedded_json(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        root = scope if scope is not None else self.soup
        container = None

        script = root.find("script", {"id": "__NEXT_DATA__"})
        if script and script.string:
            try:
                container = json.loads(script.string)
            except json.JSONDecodeError:
                pass

        if container is None:
            for var in ("__INITIAL_STATE__", "__NUXT__", "__PRELOADED_STATE__"):
                pattern = re.compile(
                    rf"window\.{var}\s*=\s*(\{{.*?\}});",
                    re.DOTALL,
                )
                m = pattern.search(self.html)
                if m:
                    try:
                        container = json.loads(m.group(1))
                        break
                    except json.JSONDecodeError:
                        continue

        if container is None:
            return []

        if not spec.expression:
            return [json.dumps(container)]
        value = _json_walk(container, spec.expression.split("."))
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) if not isinstance(v, str) else v for v in value]
        return [str(value)]

    # ------------------------------------------------------------------
    # Meta tags
    # ------------------------------------------------------------------
    def _run_meta(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        root = scope if scope is not None else self.soup
        expr = spec.expression.strip()
        if expr.startswith("meta"):
            elements = root.select(expr)
        else:
            elements = (
                root.select(f'meta[property="{expr}"]')
                or root.select(f'meta[name="{expr}"]')
            )
        values = []
        for el in elements:
            content = el.get("content")
            if content:
                values.append(content)
        return values

    # ------------------------------------------------------------------
    # Regex
    # ------------------------------------------------------------------
    def _run_regex(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        source = str(scope) if scope is not None else self.html
        pattern = re.compile(spec.expression)
        results = []
        for m in pattern.findall(source):
            if isinstance(m, tuple):
                first = next((g for g in m if g), None)
                if first is not None:
                    results.append(first)
            elif m:
                results.append(m)
        return results

    # ------------------------------------------------------------------
    # Image URL
    # ------------------------------------------------------------------
    def _run_image(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        root = scope if scope is not None else self.soup
        expr = spec.expression or "img"
        elements = root.select(expr)
        urls = []
        for el in elements:
            for attr in ("src", "data-src", "data-original", "data-lazy-src"):
                v = el.get(attr)
                if v:
                    urls.append(v)
                    break
        return urls

    # ------------------------------------------------------------------
    # Link URL
    # ------------------------------------------------------------------
    def _run_link(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        root = scope if scope is not None else self.soup
        elements = root.select(spec.expression)
        hrefs = []
        for el in elements:
            v = el.get("href")
            if v:
                hrefs.append(v)
        return hrefs

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------
    def _run_pagination(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        root = scope if scope is not None else self.soup
        expr = spec.expression or (
            "a[rel='next'], li.next a, a.next, "
            "a[aria-label='Next'], a.pagination-next"
        )
        elements = root.select(expr)
        hrefs = []
        for el in elements:
            v = el.get("href")
            if v:
                hrefs.append(v)
        return hrefs

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    def _run_table(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        root = scope if scope is not None else self.soup
        tables = root.select(spec.expression)
        rows_out: list[dict] = []
        for table in tables:
            headers = [
                th.get_text(strip=True)
                for th in table.select("thead th")
            ]
            body_rows = table.select("tbody tr") or table.select("tr")
            for row in body_rows:
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                values = [c.get_text(strip=True) for c in cells]
                if headers and len(headers) == len(values):
                    rows_out.append(dict(zip(headers, values)))
                else:
                    rows_out.append({f"col_{i}": v for i, v in enumerate(values)})
        return rows_out

    # ------------------------------------------------------------------
    # Deferred types
    # ------------------------------------------------------------------
    def _run_url(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        raise UnsupportedSelectorType(
            "url selectors are resolved by the record engine, not the HTML engine"
        )

    def _run_interaction(self, spec: SelectorSpec, scope: Optional[Tag]) -> list:
        raise UnsupportedSelectorType(
            "interaction selectors (click/scroll) require a live browser context"
        )

    # ------------------------------------------------------------------
    # Match-mode application (§13.4)
    # ------------------------------------------------------------------
    def _apply_match_mode(self, spec: SelectorSpec, matches: list):
        mode = spec.multiple
        if mode == MatchMode.FIRST:
            return matches[0]
        if mode == MatchMode.ALL:
            return matches
        if mode == MatchMode.NTH:
            idx = spec.nth or 0
            if idx < 0 or idx >= len(matches):
                raise IndexError(f"nth({idx}) out of range (got {len(matches)} matches)")
            return matches[idx]
        if mode == MatchMode.JOIN:
            return spec.join_separator.join(str(m) for m in matches)
        raise ValueError(f"Unknown match mode: {mode}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    <html><head>
      <meta property="og:price:amount" content="24.99">
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
      <a class="detail" href="/products/mouse">View</a>
      <table class="specs">
        <thead><tr><th>Key</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Weight</td><td>90g</td></tr>
          <tr><td>Battery</td><td>AA</td></tr>
        </tbody>
      </table>
    </body></html>
    """

    engine = SelectorEngine(sample)

    r = engine.run(SelectorSpec(name="price", type=SelectorType.JSONLD, expression="offers.price"))
    print(f"jsonld:        {r.value!r}")

    r = engine.run(SelectorSpec(
        name="sku", type=SelectorType.EMBEDDED_JSON,
        expression="props.pageProps.products.0.sku",
    ))
    print(f"embedded_json: {r.value!r}")

    r = engine.run(SelectorSpec(name="price", type=SelectorType.META, expression="og:price:amount"))
    print(f"meta:          {r.value!r}")

    r = engine.run(SelectorSpec(name="img", type=SelectorType.IMAGE, expression="img.thumb"))
    print(f"image:         {r.value!r}")

    r = engine.run(SelectorSpec(name="url", type=SelectorType.LINK, expression="a.detail"))
    print(f"link:          {r.value!r}")

    r = engine.run(SelectorSpec(
        name="row", type=SelectorType.TABLE, expression="table.specs",
        multiple=MatchMode.ALL,
    ))
    print(f"table rows:    {r.value!r}")

    r = engine.run(SelectorSpec(
        name="num", type=SelectorType.REGEX,
        expression=r'price": "(\d+\.\d+)"',
    ))
    print(f"regex (hit):   {r.value!r}")

    print("\nSelectorEngine OK.")