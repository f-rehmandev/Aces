import asyncio
import re
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scraper.engine import ScraperEngine
from extractor.schema_extractor import DataExtractor

logger = logging.getLogger("triangulator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _parse_price(price_str) -> float | None:
    """Pulls the first numeric value out of a price string like '£51.77' or '$24.99'."""
    if not price_str:
        return None
    match = re.search(r"[\d,]+\.?\d*", str(price_str))
    if not match:
        return None
    return float(match.group().replace(",", ""))


class Triangulator:
    """
    Scrapes the same product across multiple independent source URLs and
    computes a per-source confidence score based on how closely prices agree.
    """

    def __init__(self):
        self.scraper = ScraperEngine()
        self.extractor = DataExtractor()

    async def _scrape_one(self, url: str, instruction: str) -> dict:
        try:
            html = await self.scraper.fetch_html(url)
            data = self.extractor.extract(html=html, instruction=instruction)
            data["source_url"] = url
            data["status"] = "ok"
            return data
        except Exception as e:
            logger.warning(f"Failed on {url}: {e}")
            return {"source_url": url, "status": "failed", "error": str(e)}

    async def triangulate(self, urls: list[str], instruction: str, price_field: str = "price") -> list[dict]:
        tasks = [self._scrape_one(url, instruction) for url in urls]
        results = await asyncio.gather(*tasks)

        ok_results = [r for r in results if r.get("status") == "ok"]
        prices = [p for r in ok_results if (p := _parse_price(r.get(price_field))) is not None]

        if prices:
            avg_price = sum(prices) / len(prices)
            for r in ok_results:
                p = _parse_price(r.get(price_field))
                if p is not None and avg_price > 0:
                    deviation = abs(p - avg_price) / avg_price
                    r["confidence_score"] = round(max(0, 1 - deviation), 2)
                else:
                    r["confidence_score"] = None
            consensus = f"{len(prices)}/{len(urls)} sources returned usable price data"
        else:
            consensus = "No sources returned usable price data"

        for r in results:
            r["source_consensus"] = consensus

        return results


if __name__ == "__main__":
    # Demo: 3 different book pages, treated as if they were 3 "sources" for the same product.
    # Real cross-retailer matching (e.g. same product on Daraz + Amazon) comes once
    # the search/URL-discovery step is built.
    urls = [
        "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
        "https://books.toscrape.com/catalogue/soumission_998/index.html",
    ]
    instruction = "Extract the book title and price as JSON with keys: title, price"

    async def main():
        t = Triangulator()
        results = await t.triangulate(urls, instruction)
        for r in results:
            print(r)

    asyncio.run(main())