"""
Grid search for Google Maps leads — expands one query into many.

Google Maps returns ~20 listings max per keyword. To get 100+ leads from
a city, we split the request into neighborhoods ("pizza shops in DHA
Lahore", "pizza shops in Gulberg Lahore", ...), run each separately, then
merge and deduplicate by `place_id`.

Discovery vs monitoring mode:
    - discovery_mode=True  → only NEW leads are returned. Existing leads
      (already seen by this client) are filtered out. Quantity will be
      lower than a normal run because the strict filter removes repeats.
    - discovery_mode=False → every lead is returned, each tagged with
      `diff_status` = "NEW" or "EXISTING" (or "NEW_UNVERIFIED" when the
      source didn't supply a place_id).

Lead memory is per-client: the same lead seen by two clients is tracked
independently for each.

Public API:
    expand_grid_search(query, areas=None, llm_router=None, max_areas=10)
    run_grid_leads_query(base_query, ..., discovery_mode=False,
                         client_id="default", lead_memory=None)
"""

from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.discovery.gmaps_client import (
    GmapsClient, GmapsError, GmapsScraperUnavailable,
)
from src.discovery.gmaps_leads import LeadsResult, _normalize_row
from src.discovery.lead_memory import (
    LeadMemoryStore,
    STATUS_NEW,
    STATUS_EXISTING,
)


logger = logging.getLogger("grid_search")


# ---------------------------------------------------------------------------
# Expansion result
# ---------------------------------------------------------------------------

@dataclass
class GridExpansion:
    original_query: str
    entity: str = ""
    city: str = ""
    areas: list[str] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    source: str = ""            # "llm" | "explicit" | "passthrough"
    error: str = ""


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_GRID_PROMPT = """You are a geographic research assistant.

Given a Google Maps search query, identify the entity being searched for
and its location, then list up to {max_areas} well-known neighborhoods,
towns, or sub-districts within that location.

User query: "{query}"

Respond with ONLY valid JSON. No explanation, no code fences:
{{
  "entity": "the thing being searched for, e.g. 'pizza shops'",
  "city": "the city or region, e.g. 'Lahore'",
  "areas": ["area 1", "area 2", "area 3"]
}}

If the query has no identifiable city or region, return an empty areas list.
"""


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

def expand_grid_search(
    query: str,
    areas: Optional[list[str]] = None,
    llm_router=None,
    max_areas: int = 30,
) -> GridExpansion:
    """
    Turn a single query into N per-area queries.

    Priority:
        1. If `areas` is provided, use it directly (no LLM call).
        2. Else, ask the LLM for neighborhoods.
        3. If the LLM fails or returns nothing, pass the original query
           through unchanged.
    """
    expansion = GridExpansion(original_query=query)

    # --- 1. Explicit areas ---
    if areas:
        expansion.areas = [a.strip() for a in areas if a.strip()]
        expansion.source = "explicit"
        for area in expansion.areas[:max_areas]:
            expansion.expanded_queries.append(f"{query} {area}")
        if not expansion.expanded_queries:
            expansion.expanded_queries = [query]
            expansion.source = "passthrough"
        return expansion

    # --- 2. LLM-driven expansion ---
    if llm_router is None:
        try:
            from src.llm.router import LLMRouter
            llm_router = LLMRouter()
        except Exception as e:
            expansion.error = f"no LLM router available: {e}"
            expansion.expanded_queries = [query]
            expansion.source = "passthrough"
            return expansion

    prompt = _GRID_PROMPT.format(query=query, max_areas=max_areas)
    try:
        result = llm_router.call(prompt)
        raw = (result.get("text") or "").strip()
    except Exception as e:
        expansion.error = f"LLM call failed: {e}"
        expansion.expanded_queries = [query]
        expansion.source = "passthrough"
        return expansion

    # Strip code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        expansion.error = f"LLM returned non-JSON: {e}"
        expansion.expanded_queries = [query]
        expansion.source = "passthrough"
        return expansion

    entity = str(parsed.get("entity") or "").strip()
    city = str(parsed.get("city") or "").strip()
    raw_areas = parsed.get("areas") or []

    expansion.entity = entity
    expansion.city = city

    if not isinstance(raw_areas, list) or not raw_areas:
        expansion.expanded_queries = [query]
        expansion.source = "passthrough"
        return expansion

    # Clean and cap areas
    cleaned: list[str] = []
    for a in raw_areas:
        a_s = str(a).strip()
        if a_s and a_s not in cleaned:
            cleaned.append(a_s)
    expansion.areas = cleaned[:max_areas]

    # Rebuild per-area queries
    for area in expansion.areas:
        if city and city.lower() not in area.lower():
            expansion.expanded_queries.append(f"{entity} in {area} {city}")
        else:
            expansion.expanded_queries.append(f"{entity} in {area}")

    if not expansion.expanded_queries:
        expansion.expanded_queries = [query]
        expansion.source = "passthrough"
    else:
        expansion.source = "llm"

    return expansion


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedupe_key(row: dict) -> str:
    """Stable identifier for one business, preferring Google's place_id."""
    for key in ("place_id", "data_id", "cid", "placeId"):
        v = str(row.get(key) or "").strip()
        if v:
            return f"id:{v}"
    title = str(row.get("title") or "").strip().lower()
    addr = str(row.get("address") or "").strip().lower()
    if title or addr:
        return f"ta:{title}|{addr}"
    canonical = json.dumps(row, sort_keys=True, default=str)
    return "h:" + hashlib.md5(canonical.encode("utf-8")).hexdigest()


def dedupe_by_identity(rows: list[dict]) -> tuple[list[dict], int]:
    """Return (unique_rows, duplicates_dropped)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        k = _dedupe_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out, len(rows) - len(out)


# ---------------------------------------------------------------------------
# Lead memory: filter (discovery) or annotate (monitoring)
# ---------------------------------------------------------------------------

def _apply_lead_memory(
    records: list[dict],
    client_id: str,
    query_key: str,
    discovery_mode: bool,
    lead_memory: Optional[LeadMemoryStore],
) -> tuple[list[dict], int, int, int, str]:
    """
    Filter or annotate `records` using persistent lead memory.

    Returns (kept_records, new_count, existing_count, unverified_count, warning).

    In both modes, every scraped lead is remembered afterward so the next
    run's memory is up to date. Gracefully no-ops when `lead_memory` is
    None — callers that don't care about memory get the old behavior.
    """
    if lead_memory is None or not records:
        return records, 0, 0, 0, ""

    place_ids = [str(r.get("place_id") or "") for r in records]
    nonempty = [p for p in place_ids if p]
    known = lead_memory.known_place_ids(client_id, nonempty)

    for rec, pid in zip(records, place_ids):
        if not pid:
            rec["diff_status"] = "NEW_UNVERIFIED"
        elif pid in known:
            rec["diff_status"] = STATUS_EXISTING
        else:
            rec["diff_status"] = STATUS_NEW

    unverified_count = sum(
        1 for r in records if r.get("diff_status") == "NEW_UNVERIFIED"
    )

    if discovery_mode:
        kept = [r for r in records if r.get("diff_status") == STATUS_NEW]
        warning = ""
        if unverified_count:
            warning = (
                f"Discovery mode excluded {unverified_count} record(s) "
                f"without a place_id — they cannot be verified as new."
            )
    else:
        kept = records
        warning = ""

    # Remember everything we scraped, even items filtered out this run.
    lead_memory.remember_batch(client_id, nonempty, query_key=query_key)

    # Counters describe the FULL scrape, not just what was kept.
    # Otherwise, discovery mode would always report existing_count = 0
    # (since existing leads are filtered out of `kept`) and the user
    # would have no way to see how many were excluded.
    new_count = sum(
        1 for r in records if r.get("diff_status") == STATUS_NEW
    )
    existing_count = sum(
        1 for r in records if r.get("diff_status") == STATUS_EXISTING
    )

    return kept, new_count, existing_count, unverified_count, warning


# ---------------------------------------------------------------------------
# Full grid run
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, str], None]


def run_grid_leads_query(
    base_query: str,
    areas: Optional[list[str]] = None,
    no_website_only: bool = False,
    max_areas: int = 30,
    output_path: Optional[str | Path] = None,
    client: Optional[GmapsClient] = None,
    llm_router=None,
    timeout_per_area: float = 300,
    collect_emails: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    discovery_mode: bool = False,
    client_id: str = "default",
    lead_memory: Optional[LeadMemoryStore] = None,
) -> LeadsResult:
    """
    Run a full grid search.

    Args:
        discovery_mode:
            True  → return only NEW leads (per-client, persistent memory).
                    A warning is written to the result when the strict
                    filter drops records — quantities will be lower than
                    an open run.
            False → return every lead, each tagged with `diff_status`
                    ("NEW", "EXISTING", or "NEW_UNVERIFIED").
        client_id:
            Tenant scope for lead memory. Different clients track
            independently.
        lead_memory:
            Injected store. When None, memory is disabled and behavior is
            identical to the previous version.
    """
    client = client or GmapsClient()
    result = LeadsResult(keywords=[base_query])
    result.discovery_mode = discovery_mode

    expansion = expand_grid_search(
        base_query, areas=areas, llm_router=llm_router, max_areas=max_areas,
    )

    # -----------------------------------------------------------------
    # Passthrough — LLM couldn't expand, so we run a single-query search.
    # We still apply lead memory so the toggle behaves identically.
    # -----------------------------------------------------------------
    if expansion.source == "passthrough":
        from src.discovery.gmaps_leads import run_leads_query
        single = run_leads_query(
            keywords=[base_query],
            no_website_only=no_website_only,
            output_path=output_path,
            client=client,
            timeout_seconds=timeout_per_area * max_areas,
            collect_emails=collect_emails,
        )
        if lead_memory is not None and single.records:
            (single.records, single.new_count, single.existing_count,
             single.unverified_count, single.discovery_warning) = _apply_lead_memory(
                single.records,
                client_id=client_id,
                query_key=base_query,
                discovery_mode=discovery_mode,
                lead_memory=lead_memory,
            )
        single.discovery_mode = discovery_mode
        single.areas_searched = 1
        return single

    queries = expansion.expanded_queries
    result.areas_searched = len(queries)

    all_raw: list[dict] = []

    for idx, q in enumerate(queries, start=1):
        if progress_callback:
            try:
                progress_callback(idx, len(queries), q)
            except Exception:
                pass

        try:
            scrape_result = client.scrape(
                keywords=[q],
                name=f"aces-grid-{idx}",
                email=collect_emails,
                timeout_seconds=timeout_per_area,
            )
        except GmapsScraperUnavailable as e:
            result.per_area_errors.append(f"[{q}] unavailable: {e}")
            continue
        except GmapsError as e:
            result.per_area_errors.append(f"[{q}] error: {e}")
            continue
        except Exception as e:
            result.per_area_errors.append(
                f"[{q}] unexpected {type(e).__name__}: {e}"
            )
            continue

        if not scrape_result.succeeded:
            result.per_area_errors.append(
                f"[{q}] {scrape_result.error or scrape_result.status}"
            )
            continue

        all_raw.extend(scrape_result.records)

    result.total_scraped = len(all_raw)

    if not all_raw:
        result.error = "no results from any area"
        if result.per_area_errors:
            result.error += f" ({len(result.per_area_errors)} errors)"
        return result

    # Dedup before normalize so we drop on the raw fields the scraper gives us
    deduped, dropped = dedupe_by_identity(all_raw)
    result.duplicates_removed = dropped

    normalized = [_normalize_row(r) for r in deduped]

    # --- Lead memory: filter (discovery) or annotate (monitoring) ---
    (memory_kept, new_count, existing_count, unverified_count, warning) = \
        _apply_lead_memory(
            normalized,
            client_id=client_id,
            query_key=base_query,
            discovery_mode=discovery_mode,
            lead_memory=lead_memory,
        )
    result.new_count = new_count
    result.existing_count = existing_count
    result.unverified_count = unverified_count
    result.discovery_warning = warning

    # --- Optional no-website filter (applies after memory) ---
    if no_website_only:
        filtered = [r for r in memory_kept if not r["has_website"]]
        result.filtered_no_website = len(memory_kept) - len(filtered)
    else:
        filtered = memory_kept

    result.records = filtered
    result.total_returned = len(filtered)

    if output_path is not None and filtered:
        from src.output.formats import write_records
        written = write_records(filtered, output_path)
        result.output_path = str(written.path)

    return result


# ---------------------------------------------------------------------------
# Preview API — used by the UI during Plan Review
# ---------------------------------------------------------------------------

def preview_grid_areas(
    query: str,
    llm_router=None,
    max_areas: int = 15,
) -> tuple[list[str], str]:
    """
    Light wrapper: return (areas, error) for a query.

    - `areas` empty + `error` empty  → query has no location; caller should
      skip grid mode
    - `areas` populated              → caller should show them for editing
    - `error` non-empty              → LLM/parse issue; caller should show
      a warning but can still proceed single-query
    """
    exp = expand_grid_search(query, llm_router=llm_router, max_areas=max_areas)
    if exp.source == "passthrough":
        return [], exp.error or ""
    return exp.areas, ""
