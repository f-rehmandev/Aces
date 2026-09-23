import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from assistant_core.command_parser import parse_command
from discovery.product_search import search_products
from scraper.engine import ScraperEngine
from extractor.schema_extractor import DataExtractor
from storage.db import get_tracked_sources, save_tracked_sources, save_run_results
from diff.diff_engine import get_previous_run, compute_diff
from diff.excel_writer import write_excel
from assistant import _scrape_source, _parse_price
from storage.audit_logger import Timer, log_event


async def run_task(user_request: str, client_id: str = "default") -> list[dict]:
    """
    The real chatbot entry point: takes a plain-English request,
    plans it, executes it, and returns structured results.
    Everything is scoped to client_id, so different clients asking about
    the same topic get fully independent tracking/history.
    """
    with Timer(user_request):
        return await _run_task_inner(user_request, client_id)


async def _run_task_inner(user_request: str, client_id: str = "default") -> list[dict]:
    spec = parse_command(user_request)
    spec.client_id = client_id
    search_query = spec.source_hint
    target_count = spec.min_records
    fields = spec.fields
    notes = spec.filters

    print(f"Plan: search '{search_query}', want ~{target_count} results, fields: {fields}")
    if getattr(spec, "ambiguity_note", ""):
        print(f"⚠ Note: {spec.ambiguity_note}")

    max_sources = min(max(3, target_count // 5), 10)

    urls = get_tracked_sources(user_request, client_id=client_id)
    if urls:
        print(f"Reusing {len(urls)} previously-tracked source(s) for this client.")
    else:
        urls = await search_products(search_query, max_results=max_sources)
        if urls:
            save_tracked_sources(user_request, urls, client_id=client_id)

    if not urls:
        print("No sources found.")
        return []

    print(f"Scraping {len(urls)} source(s)...\n")

    scraper = ScraperEngine()
    extractor = DataExtractor()

    tasks = [_scrape_source(scraper, extractor, url, search_query, fields, notes) for url in urls]
    results_per_source = await asyncio.gather(*tasks)
    all_items = [item for sublist in results_per_source for item in sublist]

    if "price" in fields:
        for item in all_items:
            item["_numeric_price"] = _parse_price(item.get("price"))
        priced = [i for i in all_items if i["_numeric_price"] is not None]
        unpriced = [i for i in all_items if i["_numeric_price"] is None]
        priced.sort(key=lambda i: i["_numeric_price"])
        for item in priced:
            del item["_numeric_price"]
        all_items = priced + unpriced

    print(f"Found {len(all_items)} total matching items.")

    if all_items:
        previous = get_previous_run(search_query, client_id=client_id)
        all_items = compute_diff(all_items, previous)
        save_run_results(search_query, all_items, client_id=client_id)

    log_event("task_completed", query=search_query, details={"result_count": len(all_items), "client_id": client_id})
    return all_items[:target_count]


if __name__ == "__main__":
    request = "generate 15 leads for a website maker, pizza shops niche, need phone, address, email and whether they have a website or not"
    results = asyncio.run(run_task(request, client_id="test_client_B"))

    print(f"\nFinal results ({len(results)}):")
    print("(Note: null fields mean 'not found on the scraped page', not confirmed absent — verify before outreach)")
    for r in results:
        print(r)

    if results:
        write_excel(results, "output_leads.xlsx")
        print(f"\nSaved to output_leads.xlsx")