"""
Bot-shell detection — spec §16.

Not every blocked page is short. Modern retailers (Amazon, Walmart,
CVS) serve large (10K–500K char) pages that look like real pages to a
naive size check, but contain zero product data — the actual content is
fetched client-side by JS we can't run.

`looks_like_bot_shell(html, query_terms)` catches these by looking for
*absence of product-shaped content* rather than length alone.
"""

from __future__ import annotations
import re
from bs4 import BeautifulSoup


# Phrases that appear on real page shells served to bots
_BLOCK_PHRASES = (
    "enable javascript",
    "please enable javascript",
    "javascript is disabled",
    "checking your browser",
    "just a moment",
    "attention required",
    "access denied",
    "request blocked",
    "are you a robot",
    "verify you are human",
    "unusual traffic",
    "automated queries",
)


# Minimum size that counts as "large enough to be suspicious"
_MEDIUM_MIN = 500
_MEDIUM_MAX = 2_000_000


def _title_text(soup) -> str:
    t = soup.find("title")
    if t is None or not t.string:
        return ""
    return t.string.strip().lower()


def _heading_texts(soup) -> list[str]:
    out = []
    for tag in ("h1", "h2", "h3"):
        for el in soup.find_all(tag):
            txt = el.get_text(" ", strip=True)
            if txt:
                out.append(txt.lower())
    return out


def _visible_text(soup) -> str:
    """Strip script/style/nav, return lowercased visible text."""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(" ", strip=True).lower().split())


def looks_like_bot_shell(
    html: str,
    query_terms: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Returns (is_bot_shell, reason). Never raises.

    Decision tree:
      1. Explicit block phrase  → shell
      2. Empty / tiny page       → not our job (looks_blocked handles it)
      3. No meaningful <title>   → shell (real content pages have titles)
      4. No headings AND no JSON-LD → shell (no structure at all)
      5. Query terms given AND none appear in visible text → shell
      6. Otherwise               → real page
    """
    if not html:
        return True, "empty html"

    n = len(html)
    lower = html.lower()

    # 1. Explicit block markers
    for phrase in _BLOCK_PHRASES:
        if phrase in lower:
            return True, f"block phrase: {phrase!r}"

    # 2. Tiny pages — defer to size-based detector
    if n < _MEDIUM_MIN:
        return False, ""

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:
        return True, f"unparseable html: {type(e).__name__}"

    # 3. Title check
    title = _title_text(soup)
    has_title = bool(title) and len(title) > 3

    # 4. Structural content check
    headings = _heading_texts(soup)
    has_headings = len(headings) > 0
    has_jsonld = bool(soup.find("script", {"type": "application/ld+json"}))

    if not has_title:
        return True, "no meaningful <title>"

    if not has_headings and not has_jsonld:
        return True, "no headings and no JSON-LD"

    # 5. Query-term check (only when the caller provided terms)
    query_terms = [t.lower() for t in (query_terms or []) if t]
    if query_terms:
        visible = _visible_text(BeautifulSoup(html, "lxml"))
        term_hits = sum(1 for t in query_terms if t in visible)
        if term_hits == 0:
            return True, "query terms not present in visible text"

    # 6. Passed all checks
    return False, ""

    # Explicit block markers win immediately
    for phrase in _BLOCK_PHRASES:
        if phrase in lower:
            return True, f"block phrase: {phrase!r}"

    # Tiny pages are handled by looks_blocked; not our job
    if n < _MEDIUM_MIN:
        return False, ""

    # Very large pages that contain no meaningful content — classic shell
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:
        return True, f"unparseable html: {type(e).__name__}"

    title = _title_text(soup)
    headings = _heading_texts(soup)
    has_jsonld = bool(soup.find("script", {"type": "application/ld+json"}))

    # If title exists but is a known block phrase — already caught above.
    # If we have real headings + jsonld, this is a real page.
    if headings and has_jsonld:
        return False, ""

    visible = _visible_text(BeautifulSoup(html, "lxml"))

    # Query terms (from the user prompt) should appear somewhere in
    # visible text on a real page.
    query_terms = [t.lower() for t in (query_terms or []) if t]
    term_hits = sum(1 for t in query_terms if t in visible)

    # Heuristic assembly
    has_title = bool(title) and len(title) > 3
    has_headings = len(headings) > 0

    # A real product page has: a real title, at least one heading,
    # SOME query term match, and JSON-LD (in the vast majority of cases).
    if has_title and has_headings and term_hits > 0:
        # Even without JSON-LD, if the query term appears in text this
        # looks like a real page. Trust the extractor.
        return False, ""

    reasons = []
    if not has_title:
        reasons.append("no meaningful <title>")
    if not has_headings:
        reasons.append("no <h1>/<h2> headings")
    if query_terms and term_hits == 0:
        reasons.append("no query terms in visible text")
    if not has_jsonld:
        reasons.append("no JSON-LD")

    return True, "; ".join(reasons) if reasons else "no product-shaped content"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Realistic-looking Amazon shell (no title text, no headings, JS-heavy)
    amazon_shell = (
        "<html><head>"
        '<script src="/opensearch.xml"></script>'
        "<title></title>"
        "</head><body>"
        + "<script>var x = 1;</script>" * 500
        + "</body></html>"
    )
    blocked, reason = looks_like_bot_shell(amazon_shell, ["panadol"])
    assert blocked, reason

    # Real product page — has title, headings, term match
    real_page = """
    <html><head><title>Panadol 500mg Tablets | Pharmacy</title>
    <script type="application/ld+json">{"@type":"Product","name":"Panadol"}</script>
    </head><body>
    <h1>Panadol 500mg Tablets</h1>
    <h2>Description</h2>
    <p>Panadol is a pain reliever.</p>
    </body></html>
    """
    blocked, reason = looks_like_bot_shell(real_page, ["panadol"])
    assert not blocked, reason

    # Real page without JSON-LD — but has real text
    real_page_no_jsonld = """
    <html><head><title>Panadol 500mg</title></head>
    <body><h1>Panadol 500mg Tablets</h1>
    <p>Buy Panadol online.</p>
    </body></html>
    """
    blocked, reason = looks_like_bot_shell(real_page_no_jsonld, ["panadol"])
    assert not blocked, reason

    # Explicit block phrase
    blocked, reason = looks_like_bot_shell(
        "<html><body>Please enable JavaScript</body></html>"
    )
    assert blocked
    assert "javascript" in reason.lower()

    # Empty
    blocked, reason = looks_like_bot_shell("")
    assert blocked

    # Tiny page — not our job, defer to looks_blocked
    blocked, reason = looks_like_bot_shell("hi")
    assert not blocked

    print("Bot-shell detection OK.")