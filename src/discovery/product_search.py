"""
Product/source search — spec §14.1.

Searches DuckDuckGo's HTML endpoint for a plain-text query and returns
real destination URLs. No API key needed.

After fetching results, we rank the URLs by a relevance score so that
deep links (product/detail/item pages) come out ahead of homepages and
category pages. Homepages rarely contain a specific product and are the
single biggest source of "0 records returned" on real queries.
"""

import asyncio
import logging
import re as _re
import sys
import os
from urllib.parse import quote_plus, unquote, urlparse, parse_qs
from bs4 import BeautifulSoup

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.scraper.engine import ScraperEngine

logger = logging.getLogger("product_search")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# DuckDuckGo redirect unwrapping
# ---------------------------------------------------------------------------

def _extract_real_url(ddg_href: str) -> str | None:
    """DuckDuckGo wraps result links in a redirect — pull the real URL out."""
    ddg_href = ddg_href.strip().strip("'\"")
    if ddg_href.startswith("//duckduckgo.com/l/") or "uddg=" in ddg_href:
        parsed = urlparse(ddg_href if ddg_href.startswith("http") else "https:" + ddg_href)
        qs = parse_qs(parsed.query)
        real = qs.get("uddg", [None])[0]
        return unquote(real).strip("'\"") if real else None
    if ddg_href.startswith("http"):
        return ddg_href
    return None


# ---------------------------------------------------------------------------
# URL relevance scoring
# ---------------------------------------------------------------------------
# Signal table:
#   Deep path (>1 slash)   : +depth (capped at 4)
#   Product keyword in path: +4
#   Document extension     : +4  (.html/.php/.aspx)
#   3+ digit ID in path    : +5
#   Query string present   : -2  (usually a search/listing page)
#   Category/blog keyword  : -6
#   Bare homepage          : -10 (and short-circuit)
#
# Higher score = better. We sort descending, stable on ties so we keep
# DuckDuckGo's own ordering for equal scores.

# ---------------------------------------------------------------------------
# URL relevance scoring
# ---------------------------------------------------------------------------
# Two modes:
#   product mode (default)   → prefer deep links to specific items
#   listing mode             → prefer category/list/brand pages that
#                              contain many items at once
#
# Higher score = better. We sort descending, stable on ties so we keep
# DuckDuckGo's own ordering for equal scores.

_DEEP_PATH_HINTS = (
    "product", "item", "medicine", "drug", "detail", "p/",
    "buy", "shop", "listing",
)

_BAD_PATH_HINTS = (
    "/category/", "/categories/", "/search", "/blog",
    "/tag/", "/author/", "/page/", "/about", "/contact",
    "/privacy", "/terms",
)

# When we want many records, category/list/brand pages are gold.
# These are boosted instead of penalized in listing mode.
_LISTING_PATH_HINTS = (
    "/category/", "/categories/", "/collections/", "/collection/",
    "/brand/", "/brands/", "/list/", "/lists/", "/catalog/",
    "/products/", "/store/", "/shop/", "/c/", "/search",
)

_PRODUCT_ID_RE = _re.compile(r"\d{3,}")
_DOC_EXT_RE = _re.compile(r"\.(html?|php|aspx?)(\?|$)", _re.IGNORECASE)


def _score_url(url: str, prefer_listings: bool = False) -> int:
    """
    Score a URL for extraction value.

    product mode (prefer_listings=False): deep product pages win
    listing mode (prefer_listings=True):  category/list/brand pages win
    """
    try:
        parts = urlparse(url)
    except Exception:
        return -100

    path = (parts.path or "").strip()
    query = parts.query or ""
    low = path.lower()

    # Bare homepage — never great, but a *bit* less bad in listing mode
    if path in ("", "/"):
        return 2 if prefer_listings else -10

    score = 0

    # Depth boosts specificity (both modes)
    depth = path.strip("/").count("/")
    score += min(depth, 4)

    if prefer_listings:
        # Listing-oriented paths get a boost
        for hint in _LISTING_PATH_HINTS:
            if hint in low:
                score += 6
                break
    else:
        # Product-page keywords boost, category keywords hurt
        for hint in _DEEP_PATH_HINTS:
            if hint in low:
                score += 4
                break
        for hint in _BAD_PATH_HINTS:
            if hint in low:
                score -= 6
                break

    # Document extension — real page, always good
    if _DOC_EXT_RE.search(path):
        score += 4

    # Numeric ID in path — almost always a detail page (product mode)
    if _PRODUCT_ID_RE.search(path):
        score += 3 if prefer_listings else 5

    # Long query strings usually mean search/listing pages
    if query:
        score += 3 if prefer_listings else -2

    return score


def _rank_urls(
    urls: list[str],
    keep: int,
    prefer_listings: bool = False,
) -> list[str]:
    """Score, sort descending, keep the top `keep`. Stable on ties."""
    if not urls:
        return []
    scored = [(u, _score_url(u, prefer_listings=prefer_listings)) for u in urls]
    scored.sort(key=lambda t: -t[1])
    return [u for u, _ in scored[:keep]]



def _rank_urls(
    urls: list[str],
    keep: int,
    prefer_listings: bool = False,
) -> list[str]:
    """Score, sort descending, keep the top `keep`. Stable on ties."""
    if not urls:
        return []
    scored = [(u, _score_url(u, prefer_listings=prefer_listings)) for u in urls]
    scored.sort(key=lambda t: -t[1])
    return [u for u, _ in scored[:keep]]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def search_products(
    query: str,
    max_results: int = 5,
    prefer_listings: bool = False,
) -> list[str]:
    """
    Search DuckDuckGo for a plain-text query, return a ranked list of
    destination URLs.

    `max_results` — how many ranked URLs to keep. Caps at 30 because the
                    DDG HTML endpoint rarely surfaces more than that for
                    a single query.
    `prefer_listings` — when True, category/list pages score higher than
                        individual product pages.
    """
    scraper = ScraperEngine()
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    logger.info(f"Searching: {query}  (max={max_results}, "
                f"listing_mode={prefer_listings})")
    html = await scraper.fetch_html(search_url)

    soup = BeautifulSoup(html, "lxml")
    links = soup.select("a.result__a")

    # Collect every DDG result we see (usually 10-30)
    urls: list[str] = []
    seen = set()
    for link in links:
        href = link.get("href", "")
        real_url = _extract_real_url(href)
        if real_url and real_url not in seen:
            seen.add(real_url)
            urls.append(real_url)

    logger.info(f"Found {len(urls)} raw results for '{query}'")

    ranked = _rank_urls(urls, keep=max_results, prefer_listings=prefer_listings)

    dropped = [u for u in urls if u not in ranked]
    if dropped:
        logger.info(f"URL scoring kept top {len(ranked)} of {len(urls)}; "
                    f"dropped {len(dropped)}")

    logger.info(f"Returning {len(ranked)} ranked results for '{query}'")
    return ranked

if __name__ == "__main__":
    async def main():
        # Product mode — deep link wins
        print("Product mode, 'Panadol price':")
        results = await search_products("Panadol price", max_results=5)
        for i, url in enumerate(results, 1):
            print(f"  {i}. {url}  (score: {_score_url(url)})")

        # Listing mode — category pages win
        print("\nListing mode, 'Panadol price':")
        results = await search_products(
            "Panadol price", max_results=5, prefer_listings=True,
        )
        for i, url in enumerate(results, 1):
            print(f"  {i}. {url}  (score: {_score_url(url, prefer_listings=True)})")

    asyncio.run(main())