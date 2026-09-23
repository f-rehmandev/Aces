import asyncio
import sys
sys.path.append("src")
from scraper.engine import ScraperEngine

async def main():
    scraper = ScraperEngine()
    pages = await scraper.fetch_paginated(
        "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
        max_pages=3,
        next_link_selector="li.next a",
    )
    print(f"\nFetched {len(pages)} pages, sizes: {[len(p) for p in pages]}")

asyncio.run(main())