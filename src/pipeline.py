import asyncio
import sys
import os

sys.path.append(os.path.dirname(__file__))
from scraper.engine import ScraperEngine
from extractor.schema_extractor import DataExtractor
from diff.excel_writer import write_excel


async def run_list(url: str, instruction: str) -> list[dict]:
    scraper = ScraperEngine()
    extractor = DataExtractor()

    html = await scraper.fetch_html(url)
    data = extractor.extract_list(html=html, instruction=instruction)
    return data


if __name__ == "__main__":
    url = "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html"
    instruction = (
        "Extract every book listed on this page. "
        "For each book, give: title, price, availability. "
        "Return a JSON array with keys: title, price, availability"
    )

    results = asyncio.run(run_list(url, instruction))
    write_excel(results, "output.xlsx")
    print(f"\nDone. Saved {len(results)} records to output.xlsx")