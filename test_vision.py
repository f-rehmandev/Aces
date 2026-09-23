import asyncio
import sys, os
sys.path.append("src")
from scraper.engine import ScraperEngine
from extractor.schema_extractor import DataExtractor

async def main():
    scraper = ScraperEngine()
    extractor = DataExtractor()

    screenshot = await scraper.fetch_screenshot("https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html")
    items = extractor.extract_from_image(
        screenshot,
        instruction="Find the book title and price shown on this page. Return as JSON with keys: title, price."
    )
    print(items)

asyncio.run(main())