import asyncio
import logging
import sys
import os
from urllib.parse import quote_plus, unquote, urlparse, parse_qs
from bs4 import BeautifulSoup

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scraper.engine import ScraperEngine

logger = logging.getLogger("product_search")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _extract_real_url(ddg_href: str) -> str | None:
    """DuckDuckGo wraps result links in a redirect — pull the real destination URL out."""
    ddg_href = ddg_href.strip().strip("'\"")  # strip stray quotes/whitespace

    if ddg_href.startswith("//duckduckgo.com/l/") or "uddg=" in ddg_href:
        parsed = urlparse(ddg_href if ddg_href.startswith("http") else "https:" + ddg_href)
        qs = parse_qs(parsed.query)
        real = qs.get("uddg", [None])[0]
        return unquote(real).strip("'\"") if real else None
    if ddg_href.startswith("http"):
        return ddg_href
    return None


async def search_products(query: str, max_results: int = 5) -> list[str]:
    """
    Searches DuckDuckGo for a plain-text query and returns a list of real
    result URLs. No API key needed.
    """
    scraper = ScraperEngine()
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    logger.info(f"Searching: {query}")
    html = await scraper.fetch_html(search_url)

    soup = BeautifulSoup(html, "lxml")
    links = soup.select("a.result__a")

    urls = []
    for link in links:
        href = link.get("href", "")
        real_url = _extract_real_url(href)
        if real_url and real_url not in urls:
            urls.append(real_url)
        if len(urls) >= max_results:
            break

    logger.info(f"Found {len(urls)} results for '{query}'")
    return urls


if __name__ == "__main__":
    async def main():
        results = await search_products("best wireless mouse price", max_results=5)
        for i, url in enumerate(results, 1):
            print(f"{i}. {url}")

    asyncio.run(main())