import asyncio
import os
import re
import sys

from storage.db import get_tracked_sources, save_tracked_sources

sys.path.append(os.path.dirname(__file__))
from diff.diff_engine import compute_diff, get_previous_run
from diff.excel_writer import write_excel
from discovery.product_search import search_products
from extractor.schema_extractor import DataExtractor
from scraper.engine import ScraperEngine
from storage.db import save_run_results


def _parse_price(price_str) -> float | None:
    if not price_str:
        return None
    match = re.search(r"[\d,]+\.?\d*", str(price_str))
    if not match:
        return None
    return float(match.group().replace(",", ""))


async def _scrape_source(scraper, extractor, url: str, search_query: str, fields: list[str] = None, notes: str = "") -> list[dict]:
    """Scrapes one URL and returns a list of matching results found there (may be empty)."""
    fields = fields or ["title", "price", "description"]
    field_list = ", ".join(fields)

    instruction = (
        f"This page may or may not be relevant to: '{search_query}'. "
        f"Additional context: {notes} " if notes else
        f"This page may or may not be relevant to: '{search_query}'. "
    )
    instruction += (
        f"Find every distinct item on this page genuinely matching this request — "
        f"include an item even if you can only fill in SOME of the fields for it; "
        f"partial data is expected and fine, do not skip an item just because some fields are missing. "
        f"For each match, extract exactly these fields: {field_list}. "
        f"Use null ONLY for a field that this specific page doesn't mention — "
        f"null means 'not shown here', not 'confirmed absent'. Never invent a value. "
        f"For any yes/no field (e.g. 'has_website'), answer true only with direct positive evidence "
        f"on this page, otherwise null (not false). "
        f"Return ONLY a JSON array of objects, each with exactly these keys: {field_list}. "
        f"Only return an empty array if there are truly NO items on this page matching the request at all."
    )

    try:
        html = await scraper.fetch_html(url)
        items = extractor.extract_list(html=html, instruction=instruction)

        if not items:
            print(f"  ~ Text extraction found nothing on {url}, trying vision fallback...")
            screenshot = await scraper.fetch_screenshot(url)
            items = extractor.extract_from_image(screenshot, instruction)

        for item in items:
            item["source_url"] = url
        return items
    except Exception as e:
        print(f"  ! Failed on {url}: {e}")
        return []
    try:
        html = await scraper.fetch_html(url)
        items = extractor.extract_list(html=html, instruction=instruction)
        for item in items:
            item["source_url"] = url
        return items
    except Exception as e:
        print(f"  ! Failed on {url}: {e}")
        return []


async def find_best_prices(product_query: str, max_sources: int = 3) -> list[dict]:
    """
    Returns ALL matching items found, ranked by price (cheapest first).
    Callers decide how many to actually display/save as "top N" —
    but diffing/history should always use the full list, not a trimmed one,
    or items look falsely "Removed" just for dropping out of a ranking cutoff.
    """
    print(f"Searching for: {product_query}...")

    urls = get_tracked_sources(product_query)
    if urls:
        print(f"Reusing {len(urls)} previously-tracked source(s) for consistent comparison.")
    else:
        urls = await search_products(product_query, max_results=max_sources)
        if urls:
            save_tracked_sources(product_query, urls)

    if not urls:
        print("No sources found.")
        return []

    print(f"Found {len(urls)} sources. Scraping each for matching products...\n")

    scraper = ScraperEngine()
    extractor = DataExtractor()

    tasks = [_scrape_source(scraper, extractor, url, product_query) for url in urls]
    results_per_source = await asyncio.gather(*tasks)

    all_items = [item for sublist in results_per_source for item in sublist]

    for item in all_items:
        item["_numeric_price"] = _parse_price(item.get("price"))
    priced_items = [i for i in all_items if i["_numeric_price"] is not None]
    priced_items.sort(key=lambda i: i["_numeric_price"])
    for item in priced_items:
        del item["_numeric_price"]

    print(f"Found {len(all_items)} total matching items, {len(priced_items)} with usable prices.")
    return priced_items


if __name__ == "__main__":
    query = "web scraping freelance jobs"
    all_results = asyncio.run(find_best_prices(query, max_sources=3))

    if all_results:
        previous = get_previous_run(query)
        diffed_all = compute_diff(all_results, previous)

        # Save the FULL diffed pool to history (so nothing looks falsely "Removed" later)
        save_run_results(query, diffed_all)

        # Only the top 3 go to Excel/display
        top_results = diffed_all[:3]

        print("\nTop matches (with diff status):")
        for r in top_results:
            print(r)

        write_excel(top_results, "output.xlsx")
        print(f"\nSaved top {len(top_results)} results to output.xlsx")
    else:
        print("\nNothing to save.")