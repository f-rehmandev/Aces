import asyncio
import re

from src.storage.db import get_tracked_sources, save_tracked_sources, save_run_results
from src.diff.diff_engine import compute_diff, get_previous_run
from src.diff.excel_writer import write_excel
from src.discovery.product_search import search_products
from src.extractor.schema_extractor import DataExtractor
from src.scraper.engine import ScraperEngine
from src.healing.rung1 import try_deterministic_extraction

# --- security primitives (Part XI) ---
from src.security.dom_sanitizer import DomSanitizer
from src.security.honeypot import HoneypotDetector
from src.security.injection_guard import InjectionGuard
from src.security.ssrf import SSRFGuard
from src.security.secrets import scrub
from src.security.trace import SecurityTrace

import logging

logger = logging.getLogger("assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Module-level guards (stateless — safe to share across calls)
# ---------------------------------------------------------------------------

_dom_sanitizer = DomSanitizer()
_honeypot_detector = HoneypotDetector()
_injection_guard = InjectionGuard()
_ssrf_guard = SSRFGuard()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_price(price_str) -> float | None:
    if not price_str:
        return None
    match = re.search(r"[\d,]+\.?\d*", str(price_str))
    if not match:
        return None
    return float(match.group().replace(",", ""))


def _filter_injections(records: list[dict], report) -> tuple[list[dict], list[str]]:
    """
    Null out field values that the injection guard flagged. Cross-record
    anomalies (record_index < 0) don't null anything — they're logged only.

    Returns (cleaned_records, list_of_flagged_descriptions).
    """
    flagged_pairs = {
        (f.record_index, f.field_name)
        for f in report.findings
        if f.record_index >= 0
    }
    flagged_desc = [
        f"{f.kind}@{f.field_name}[{f.record_index}]"
        for f in report.findings
        if f.record_index >= 0
    ]

    out = []
    for i, rec in enumerate(records):
        cleaned = dict(rec)
        for fname in list(cleaned.keys()):
            if (i, fname) in flagged_pairs:
                cleaned[fname] = None
        out.append(cleaned)
    return out, flagged_desc


# ---------------------------------------------------------------------------
# Scrape + security pipeline
# ---------------------------------------------------------------------------

async def _scrape_source(
    scraper,
    extractor,
    url: str,
    search_query: str,
    fields: list[str] = None,
    notes: str = "",
    trace: SecurityTrace = None,
    network_manager=None,
    return_html: bool = False,
):
    """
    Scrapes one URL with the full security pipeline:

        1. SSRF gate    — reject private / metadata / non-HTTP URLs
        2. Honeypot     — flag hidden links, form traps, duplicate bursts
        3. DOM sanitize — strip hidden / injected content before the LLM sees it
        4. Extract      — LLM structured extraction (vision fallback if empty)
        5. Injection    — validate output, null any field flagged as hostile
        6. Secret scrub — used on every logged/returned error string

    Signature is backward-compatible: `trace` is optional.
    """
    fields = fields or ["title", "price", "description"]
    field_list = ", ".join(fields)

    # If the caller asked for the sanitized HTML back, all early-exit
    # paths must return the (items, html) shape too. This local helper
    # keeps every `return _maybe_pair([])` consistent.
    def _maybe_pair(items_out, html_out=""):
        return (items_out, html_out) if return_html else items_out
    # --- 1. SSRF gate ---
    ssrf_result = _ssrf_guard.validate_url(url)
    if not ssrf_result.allowed:
        logger.warning(f"SSRF guard rejected {url}: {ssrf_result.reason}")
        if trace is not None:
            trace.url = url
            trace.ssrf_allowed = False
            trace.ssrf_reason = ssrf_result.reason
        return _maybe_pair([])

    if trace is not None:
        trace.url = url
        trace.ssrf_allowed = True

    # --- build the instruction (unchanged) ---
    instruction = f"This page may or may not be relevant to: '{search_query}'. "
    if notes:
        instruction += f"Additional context: {notes} "
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

    # --- fetch (Playwright + optional ScraperAPI fallback) ---
    # NOTE: we do NOT auto-create a real ScraperAPIProvider here.
    # Tests and dev callers should get pure-Playwright behaviour unless
    # they explicitly hand us a manager with a provider. The production
    # path wires the real provider in src/pipeline_runner.py.
    if network_manager is None:
        from src.network.manager import NetworkManager
        network_manager = NetworkManager(scraper, None)

    try:
        fetch_result = await network_manager.fetch(
            url, query_terms=fields,
        )
    except Exception as e:
        safe_err = scrub(f"Fetch layer raised on {url}: {e}").scrubbed
        logger.warning(safe_err)
        return []

    if trace is not None:
        trace.provider_used = fetch_result.provider
        trace.fallback_used = fetch_result.provider != "playwright"
        trace.block_reason = fetch_result.block_reason
    html = fetch_result.html or ""
    if not html:
        # Prefer the scrubbed error the NetworkManager captured, if any.
        upstream = fetch_result.metadata.get("playwright_error") \
            or fetch_result.metadata.get("fallback_error") or ""
        if upstream:
            safe_err = scrub(
                f"Fetch returned no HTML for {url} ({upstream})"
            ).scrubbed
        else:
            safe_err = scrub(f"Fetch returned no HTML for {url}").scrubbed
        logger.warning(safe_err)
        return _maybe_pair([])
    
    if fetch_result.blocked:
        logger.info(
            f"Proceeding with extraction on {url} even though the page "
            f"looks blocked ({fetch_result.block_reason})"
        )

    # --- 2. honeypot detection (records findings, does not block) ---
    try:
        honeypot_report = _honeypot_detector.detect_in_html(html)
        if honeypot_report.is_suspicious:
            logger.warning(f"Honeypot signals on {url}: {honeypot_report.summary()}")
        if trace is not None:
            trace.honeypot_findings = [f.kind for f in honeypot_report.findings]
    except Exception as e:
        safe = scrub(f"Honeypot detector failed on {url}: "
                      f"{type(e).__name__}: {e}").scrubbed
        logger.warning(safe)

    # --- 3. DOM sanitize before the LLM sees it ---
    try:
        san = _dom_sanitizer.sanitize(html)
    except Exception as e:
        safe = scrub(f"DOM sanitizer failed on {url}: "
                      f"{type(e).__name__}: {e}").scrubbed
        logger.warning(safe)
        return _maybe_pair([])
    
    if trace is not None:
        trace.sanitizer_removed = san.removed_count
        trace.sanitizer_reasons = dict(san.reasons)
    if san.removed_count:
        logger.info(f"Sanitized {url}: {san.summary()}")

    # --- 4a. Rung 1 — deterministic (JSON-LD, meta tags) ---
    # Cheap, no LLM. Structured markup is ground truth from the page.
    deterministic_items: list[dict] = []
    try:
        deterministic_items = try_deterministic_extraction(
            san.sanitized_html, fields, url=url,
        )
    except Exception as e:
        safe = scrub(f"Rung 1 failed on {url}: "
                      f"{type(e).__name__}: {e}").scrubbed
        logger.warning(safe)


    if deterministic_items:
        logger.info(
            f"Rung 1 succeeded on {url}: "
            f"{len(deterministic_items)} record(s) via structured data"
        )
        items = deterministic_items
    else:
        # --- 4b. Rung 2 — text-only LLM on cleaned DOM ---
        try:
            items = extractor.extract_list(
                html=san.sanitized_html, instruction=instruction,
            )
        except Exception as e:
            safe_err = scrub(f"Rung 2 extraction failed on {url}: {e}").scrubbed
            logger.warning(safe_err)
            items = []

        # --- 4c. Rung 3 — vision fallback ---
        if not items:
            logger.info(f"Rung 2 empty on {url}, trying Rung 3 (vision fallback)...")
            try:
                screenshot = await scraper.fetch_screenshot(url)
                items = extractor.extract_from_image(screenshot, instruction)
                if items:
                    for item in items:
                        item.setdefault("_extraction_method", "rung3_vision")
            except Exception as e:
                safe_err = scrub(f"Rung 3 extraction failed on {url}: {e}").scrubbed
                logger.warning(safe_err)
                items = []

    # Record which rung produced the data (for the trace / UI)
    if items and trace is not None:
        trace.extraction_rung = items[0].get("_extraction_method", "")

    if trace is not None:
        trace.records_extracted = len(items)

    if not items:
      return _maybe_pair([], san.sanitized_html)
    
    # --- 5. injection guard on output ---
    inj_report = _injection_guard.check_records(items)
    if inj_report.is_suspicious:
        logger.warning(
            f"Injection guard on {url}: {inj_report.summary()}"
        )
        items, flagged = _filter_injections(items, inj_report)
        if trace is not None:
            trace.injection_findings = flagged

    if trace is not None:
        trace.records_after_guard = len(items)

    for item in items:
        item["source_url"] = url

    if return_html:
        # `san` was created above; on early-exit paths it may not exist,
        # but by the time we reach here we've gone through sanitization.
        sanitized = locals().get("san")
        html_out = sanitized.sanitized_html if sanitized is not None else ""
        return items, html_out
    return items


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def find_best_prices(product_query: str, max_sources: int = 3) -> list[dict]:
    """
    Returns ALL matching items found, ranked by price (cheapest first).
    Every source is scraped through the full security pipeline.
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
        save_run_results(query, diffed_all)
        top_results = diffed_all[:3]

        print("\nTop matches (with diff status):")
        for r in top_results:
            print(r)

        write_excel(top_results, "output.xlsx")
        print(f"\nSaved top {len(top_results)} results to output.xlsx")
    else:
        print("\nNothing to save.")