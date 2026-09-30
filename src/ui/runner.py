"""
UI ↔ pipeline bridge.

Streamlit cannot call `PipelineRunner` directly for three reasons:
    1. It's async.
    2. It returns a rich object the UI can't render verbatim.
    3. It needs scraper/extractor dependencies that may not be set up
       on every machine.

This module absorbs all three. One entry point — `run_ui_task()` — resolves
a TaskSpec via the real dispatcher, runs the full pipeline, falls back to
a demo run on any error, and returns a `UiRunResult` with a fixed shape
that the Streamlit layer can render without knowing anything about the
internals.
"""

from __future__ import annotations
import asyncio
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from src.auth.context import ClientContext, anonymous_context
# Windows: Streamlit launches in a worker thread, but Playwright needs a
# Proactor event loop for its subprocess pipe. Set the policy before any
# asyncio.run() call this module makes.
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass



from src.core.task_spec import TaskSpec
from src.pipeline_runner import PipelineRunner, PipelineResult
from src.network.manager import NetworkManager
from src.network.scraperapi import ScraperAPIProvider
from src.core.task_spec import TaskSpec
from src.pipeline_runner import PipelineRunner, PipelineResult

from src.discovery.gmaps_client import GmapsScraperUnavailable
from src.discovery.grid_search import preview_grid_areas, run_grid_leads_query
from src.discovery.gmaps_leads import LeadsResult

# ---------------------------------------------------------------------------
# Result shape — the UI's contract
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    time: str
    kind: str
    message: str
    level: str = "info"       # info | success | warning | error

    def to_dict(self) -> dict:
        return {
            "time": self.time, "kind": self.kind,
            "message": self.message, "level": self.level,
        }


@dataclass
class UiRunResult:
    mode: str                          # "real" | "demo"
    records: list[dict] = field(default_factory=list)

    # top-of-screen metrics
    record_count: int = 0
    quality_score: Optional[float] = None
    quality_passed: Optional[bool] = None
    confidence_mean: Optional[float] = None
    source_count: int = 0

    # change badges
    new_count: int = 0
    changed_count: int = 0
    removed_count: int = 0
    existing_count: int = 0            # monitoring mode: leads already seen
    discovery_mode: bool = False       # True when strict new-only filter ran
    discovery_warning: str = ""        # set when discovery dropped records

    # trace + diagnostics
    trace: list[TraceEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    # extras
    spec: Optional[TaskSpec] = None
    workbook_path: Optional[str] = None
    receipt_signature: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _evt(kind: str, message: str, level: str = "info") -> TraceEvent:
    return TraceEvent(time=_now(), kind=kind, message=message, level=level)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ui_task(
    prompt: str,
    url: str = "",
    client_id: str = "default",
    output_format: str = "XLSX",
    min_sources: int = 1,
    scraper=None,
    extractor=None,
    output_path=None,
    context: Optional[ClientContext] = None,
) -> UiRunResult:
    """
    Full end-to-end UI call. Falls back to a demo run on any error, so
    the caller always gets a renderable result.
    """
    trace: list[TraceEvent] = []
    trace.append(_evt("PARSE", f'Task received: "{prompt[:80]}"', "info"))
    trace.append(_evt("CLIENT", f"client_id = {client_id}", "info"))
    if url:
        trace.append(_evt("TARGET", f"Explicit start URL: {url}", "info"))

    # --- 1. Build the TaskSpec via the real dispatcher (NL path) ---
    try:
        spec = _build_spec(prompt, url, client_id, output_format, min_sources)
    except Exception as e:
        trace.append(_evt("ERROR", f"Failed to build TaskSpec: {e}", "error"))
        return _demo_result(prompt, url, client_id, trace,
                            reason=f"spec build failed: {type(e).__name__}")

    trace.append(_evt(
        "SPEC",
        f"TaskSpec built: objective={spec.objective}, "
        f"fields={spec.field_names}",
        "info",
    ))

    # --- 2. Compliance pre-screen (§11.1 step 9) ---
    if spec.compliance.refusal_reason:
        trace.append(_evt(
            "COMPLY",
            f"Request refused at pre-screen: {spec.compliance.refusal_reason}",
            "warning",
        ))
        return _demo_result(prompt, url, client_id, trace,
                            reason="compliance refusal")
    trace.append(_evt("COMPLY", "Compliance pre-screen passed.", "success"))

    # --- 3. Run the real pipeline (with discovery if needed) ---
    try:
        scraper, extractor, injected = _resolve_runtime(scraper, extractor)
        pipeline_result, trace = asyncio.run(
            _run_pipeline_async(spec, client_id, scraper, extractor,
                                 output_path, trace, context=context,
                                 injected=injected)
        )
    except Exception as e:
        trace.append(_evt(
            "ERROR",
            f"Pipeline run failed: {type(e).__name__}: {e}",
            "error",
        ))
        trace.append(_evt("FALLBACK", "Falling back to demo simulation.", "warning"))
        return _demo_result(prompt, url, client_id, trace,
                            reason=f"pipeline error: {type(e).__name__}")

    # --- 4. Translate the pipeline result into the UI shape ---
    trace.extend(_trace_from_pipeline(pipeline_result))
    result = _real_result(pipeline_result, trace)
    result.spec = spec
    return result


# ---------------------------------------------------------------------------
# Spec building
# ---------------------------------------------------------------------------

def _build_spec(
    prompt: str, url: str, client_id: str,
    output_format: str, min_sources: int,
) -> TaskSpec:
    """
    Use the real dispatcher (natural-language path) to produce the
    TaskSpec, then apply UI-level overrides (URL, output format, min sources).
    """
    from src.intake.dispatcher import resolve_input, InputKind

    resolution = resolve_input(
        prompt, kind=InputKind.NATURAL_LANGUAGE, client_id=client_id,
    )
    spec = resolution.spec

    if url:
        # If the user gave a URL, it becomes the ONLY target.
        spec.target.start_urls = [url]
    if output_format:
        spec.output.format = output_format.lower()
    spec.source_requirements.min_independent_sources = max(1, int(min_sources))
    return spec


def _resolve_runtime(scraper, extractor):
    """
    Build real scraper/extractor if none was injected.

    Returns (scraper, extractor, injected) where `injected` is True when
    the caller supplied a scraper — in that case we must NOT create a real
    network provider, because the caller is testing or in dev mode.
    """
    injected = scraper is not None
    if scraper is None:
        from src.scraper.engine import ScraperEngine
        scraper = ScraperEngine()
    if extractor is None:
        from src.extractor.schema_extractor import DataExtractor
        extractor = DataExtractor()
    return scraper, extractor, injected


def _build_network_manager(scraper, injected: bool) -> NetworkManager:
    """
    Fetch layer.

    Production (scraper built by us): the full tier chain — Byparr,
    ScraperAPI, ScrapingAnt, WebScrapingAPI, ZenRows, Zenscrape, Apify
    — whichever .env keys are present.

    Test/dev (scraper injected by caller): Playwright-only. Never
    reaches the network in tests.
    """
    if injected:
        return NetworkManager(scraper, None)

    from src.network.manager import build_production_manager
    return build_production_manager(scraper)

# ---------------------------------------------------------------------------
# Trace translation
# ---------------------------------------------------------------------------

def _trace_from_pipeline(result: PipelineResult) -> list[TraceEvent]:
    events: list[TraceEvent] = []

    for t in result.security_traces:
        if not t.ssrf_allowed:
            events.append(_evt(
                "SSRF", f"Blocked {t.url}: {t.ssrf_reason}", "warning",
            ))
            continue
        events.append(_evt("FETCH", f"Fetched {t.url}", "info"))
        if t.sanitizer_removed:
            events.append(_evt(
                "SANITIZE",
                f"Stripped {t.sanitizer_removed} hidden element(s) before extraction",
                "info",
            ))
        if t.honeypot_findings:
            events.append(_evt(
                "HONEYPOT",
                f"Signals on {t.url}: {', '.join(t.honeypot_findings)}",
                "warning",
            ))
        if t.injection_findings:
            events.append(_evt(
                "INJECT",
                f"Nulled {len(t.injection_findings)} hostile field(s)",
                "warning",
            ))

    events.append(_evt(
        "EXTRACT",
        f"Extracted {len(result.records)} record(s).",
        "success" if result.records else "warning",
    ))

    events.append(_evt(
        "QUALITY",
        f"Quality score {result.quality_score:.2f} — "
        + ("passed" if result.quality_passed else "failed"),
        "success" if result.quality_passed else "warning",
    ))

    if result.publication_decision is not None:
        events.append(_evt(
            "PUBLISH",
            f"Publication gate: {result.publication_decision.state.value}",
            "success" if result.publication_decision.allowed else "warning",
        ))

    cs = result.change_set_summary or {}
    events.append(_evt(
        "DIFF",
        f"Changes: {cs.get('new', 0)} new, "
        f"{cs.get('modified', 0)} modified, "
        f"{cs.get('removed', 0)} removed",
        "info",
    ))

    if result.confidence_mean:
        events.append(_evt(
            "CONFIDENCE",
            f"Mean confidence {result.confidence_mean:.2f}",
            "info",
        ))

    if result.receipt_signature:
        events.append(_evt(
            "RECEIPT",
            "Signed execution receipt generated.",
            "success",
        ))

    # Surface pipeline warnings (which include silent per-stage failures)
    for w in (result.warnings or []):
        events.append(_evt("WARN", str(w), "warning"))

    events.append(_evt("DONE", "Run complete.", "success"))
    return events


# ---------------------------------------------------------------------------
# Result translation
# ---------------------------------------------------------------------------

def _real_result(result: PipelineResult, trace: list[TraceEvent]) -> UiRunResult:
    cs = result.change_set_summary or {}
    source_count = len({t.url for t in result.security_traces})

    workbook_path = None
    if result.workbook is not None:
        workbook_path = str(result.workbook.path)

    return UiRunResult(
        mode="real",
        records=result.records,
        record_count=len(result.records),
        quality_score=result.quality_score,
        quality_passed=result.quality_passed,
        confidence_mean=result.confidence_mean,
        source_count=source_count,
        new_count=cs.get("new", 0),
        changed_count=cs.get("modified", 0),
        removed_count=cs.get("removed", 0),
        trace=trace,
        warnings=list(result.warnings),
        workbook_path=workbook_path,
        receipt_signature=result.receipt_signature,
    )


def _demo_result(
    prompt: str, url: str, client_id: str,
    trace: list[TraceEvent], reason: str = "",
) -> UiRunResult:
    """
    Plausible-but-honest demo run. The trace explicitly says "demo" so
    nobody mistakes it for a real extraction.
    """
    trace.append(_evt(
        "DEMO",
        f"Demo mode: {reason or 'backend unavailable'}",
        "warning",
    ))

    demo_records = [
        {
            "record": f"Item {i + 1}",
            "price": round(random.uniform(15, 900), 2),
            "confidence": round(random.uniform(0.7, 0.99), 2),
            "source": random.choice(
                ["retailer-a.com", "retailer-b.com", "retailer-c.com"]
            ),
        }
        for i in range(6)
    ]

    for kind, msg, level in [
        ("PARSE", "Parsing natural-language intent into a TaskSpec.", "info"),
        ("SCHEMA", "Schema discovery: record boundary + fields identified.", "info"),
        ("FETCH", "Layer 2 (Playwright) fetch simulated.", "info"),
        ("EXTRACT", "Schema-free extraction returned demo records.", "success"),
        ("QUALITY", "Quality gate evaluated (demo).", "success"),
        ("DIFF", "Compared against last trusted run (demo).", "info"),
    ]:
        trace.append(_evt(kind, msg, level))

    return UiRunResult(
        mode="demo",
        records=demo_records,
        record_count=random.randint(80, 400),
        quality_score=round(random.uniform(0.88, 0.98), 2),
        quality_passed=True,
        confidence_mean=round(random.uniform(0.85, 0.97), 2),
        source_count=random.randint(3, 12),
        new_count=random.randint(2, 20),
        changed_count=random.randint(5, 60),
        removed_count=random.randint(0, 8),
        trace=trace,
        error=reason,
    )



# ---------------------------------------------------------------------------
# Plan Review entry points (used by the two-step UI flow)
# ---------------------------------------------------------------------------

def build_spec_for_preview(
    prompt: str,
    url: str = "",
    client_id: str = "default",
    output_format: str = "XLSX",
    min_sources: int = 1,
) -> tuple[Optional[TaskSpec], str]:
    """
    Public wrapper around `_build_spec` for the Plan Review step.

    Returns (spec, error_message). On success, `error_message` is empty.
    Never raises — the UI shows the error inline and lets the user retry.
    """
    try:
        spec = _build_spec(prompt, url, client_id, output_format, min_sources)
        return spec, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def run_ui_task_from_spec(
    spec: TaskSpec,
    client_id: str = "default",
    scraper=None,
    extractor=None,
    output_path=None,
    context: Optional[ClientContext] = None,
    grid_areas: Optional[list[str]] = None,
    discovery_mode: bool = False,
) -> UiRunResult:
    
    """
    Run the pipeline against a spec the user has already reviewed (and
    possibly edited) in the Plan Review screen.
    """
    trace: list[TraceEvent] = [
        _evt("RUN", f"Executing approved plan (task_id={spec.task_id[:8]}…)", "info"),
    ]

    if spec.compliance.refusal_reason:
        trace.append(_evt(
            "COMPLY",
            f"Refused at pre-screen: {spec.compliance.refusal_reason}",
            "warning",
        ))
        return _demo_result("", "", client_id, trace,
                            reason="compliance refusal")

    trace.append(_evt("COMPLY", "Compliance pre-screen passed.", "success"))

    # --- Grid-search path (Google Maps lead-gen) ---
    if grid_areas:
        return _run_grid_path(
            spec=spec, areas=grid_areas, trace=trace,
            output_path=output_path, client_id=client_id,
            discovery_mode=discovery_mode,
        )

    trace.append(_evt(
        "SPEC",
        f"objective={spec.objective}, fields={spec.field_names}, "
        f"urls={len(spec.target.start_urls)}",
        "info",
    ))
    try:
        scraper, extractor, injected = _resolve_runtime(scraper, extractor)
        pipeline_result, trace = asyncio.run(
            _run_pipeline_async(spec, client_id, scraper, extractor,
                                 output_path, trace, context=context,
                                 injected=injected)
        )

    except Exception as e:
        trace.append(_evt(
            "ERROR", f"Pipeline run failed: {type(e).__name__}: {e}", "error",
        ))
        trace.append(_evt("FALLBACK", "Falling back to demo simulation.", "warning"))
        return _demo_result("", "", client_id, trace,
                            reason=f"pipeline error: {type(e).__name__}")

    trace.extend(_trace_from_pipeline(pipeline_result))
    result = _real_result(pipeline_result, trace)
    result.spec = spec
    return result



# ---------------------------------------------------------------------------
# Async pipeline orchestration with source discovery
# ---------------------------------------------------------------------------

async def _run_pipeline_async(
    spec: TaskSpec,
    client_id: str,
    scraper,
    extractor,
    output_path,
    trace: list[TraceEvent],
    context: Optional[ClientContext] = None,
    injected: bool = False,
) -> tuple[PipelineResult, list[TraceEvent]]:
    """
    Discovery + execution. If the spec has no start_urls but does have a
    source_hint (the natural-language search query the LLM produced),
    run source discovery to find candidate URLs before fetching.

    Mutates `spec.target.start_urls` in place and appends trace events
    describing what discovery did.
    """
    trace = list(trace)
    if not spec.target.start_urls and spec.target.source_hint:
        hint = spec.target.source_hint

        # ----------------------------------------------------------------
        # Source pinning: once a query has run, reuse the same URLs for
        # subsequent runs. Without this, DuckDuckGo returns a slightly
        # different top-5 each time and results drift run-to-run.
        # Keyed on the natural-language prompt (stable), scoped by client.
        # ----------------------------------------------------------------
        pin_key = spec.natural_language_prompt.strip() or hint
        pinned: list[str] = []
        try:
            from src.storage.db import get_tracked_sources
            pinned = get_tracked_sources(pin_key, client_id=client_id) or []
        except Exception as e:
            trace.append(_evt(
                "DISCOVER",
                f"Could not read pinned sources: {type(e).__name__}: {e}",
                "warning",
            ))

                # Scale source count to the user's request. Asking for 50 records
        # needs many more sources than asking for 5.
        min_records = max(1, int(getattr(spec.quality, "min_records", 10) or 10))
        max_sources = max(5, min(20, min_records))
        prefer_listings = min_records >= 10

        # If we pinned fewer sources than we now need, re-discover to top up.
        need_more = pinned and len(pinned) < max_sources

        if pinned and not need_more:
            spec.target.start_urls = list(pinned)
            trace.append(_evt(
                "DISCOVER",
                f"Reusing {len(pinned)} pinned source(s) from the previous run.",
                "info",
            ))
        else:
            if pinned:
                trace.append(_evt(
                    "DISCOVER",
                    f"Have {len(pinned)} pinned source(s) but need "
                    f"{max_sources} — running fresh discovery.",
                    "info",
                ))

            trace.append(_evt(
                "DISCOVER",
                f"Searching for up to {max_sources} source(s) for: \"{hint}\" "
                f"(target: {min_records} records, listing_mode={prefer_listings})",
                "info",
            ))
            try:
                from src.discovery.product_search import search_products
                fresh = await search_products(
                    hint,
                    max_results=max_sources,
                    prefer_listings=prefer_listings,
                )
                # Merge pinned + fresh, dedup, preserve order
                merged = list(pinned)
                seen = set(merged)
                for u in fresh:
                    if u not in seen:
                        seen.add(u)
                        merged.append(u)
                urls = merged[:max_sources]

                # Re-rank the merged URL list by domain reputation.
                # Domains that have produced records before get promoted;
                # consistently-empty domains get demoted.
                try:
                    from src.discovery.domain_reputation import (
                        build_supabase_store, apply_reputation,
                    )
                    from src.discovery.product_search import _score_url
                    rep_store = build_supabase_store()
                    base_scores = [
                        _score_url(u, prefer_listings=prefer_listings)
                        for u in urls
                    ]
                    ranked = apply_reputation(urls, base_scores, rep_store)
                    urls = [u for u, _ in ranked]
                    trace.append(_evt(
                        "DISCOVER",
                        f"Re-ranked {len(urls)} sources by domain reputation.",
                        "info",
                    ))
                except Exception as e:
                    # Any failure here just means we use the original order.
                    trace.append(_evt(
                        "DISCOVER",
                        f"Domain re-ranking skipped: {type(e).__name__}: {e}",
                        "warning",
                    ))

            except Exception as e:
                urls = list(pinned)
                trace.append(_evt(
                    "DISCOVER",
                    f"Discovery failed: {type(e).__name__}: {e}",
                    "warning",
                ))

            if urls:
                spec.target.start_urls = list(urls)
                trace.append(_evt(
                    "DISCOVER",
                    f"Found {len(urls)} candidate source(s) — pinning for "
                    f"future runs.",
                    "success",
                ))
                # Pin them for consistency
                try:
                    from src.storage.db import save_tracked_sources
                    save_tracked_sources(pin_key, urls, client_id=client_id)
                except Exception as e:
                    trace.append(_evt(
                        "DISCOVER",
                        f"Could not pin sources: {type(e).__name__}: {e}",
                        "warning",
                    ))
            else:
                trace.append(_evt(
                    "DISCOVER",
                    "Discovery returned no usable URLs — pipeline will have "
                    "nothing to fetch.",
                    "warning",
                ))

    elif not spec.target.start_urls and not spec.target.source_hint:
        trace.append(_evt(
            "DISCOVER",
            "No start URLs and no search hint — nothing to fetch. "
            "Provide a URL in Advanced options or make the prompt more specific.",
            "warning",
        ))

    # Translate the TaskSpec's quality thresholds into runner rules.
    # Without this, the runner always uses min_records=1 and the gate
    # passes trivially no matter what the user requested.
    from src.quality.rules import QualityRules

    user_quality = QualityRules(
        min_records=max(1, int(getattr(spec.quality, "min_records", 1) or 1)),
        min_populated_field_pct=getattr(
            spec.quality, "min_populated_field_pct", 0.5,
        ),
        max_failed_page_pct=getattr(
            spec.quality, "max_failed_page_pct", 0.3,
        ),
        max_empty_page_pct=getattr(
            spec.quality, "max_empty_page_pct", 0.5,
        ),
                max_source_disagreement_rate=getattr(
            spec.quality,
            "max_source_disagreement_rate",
            0.25,
        ),
        freshness_window_seconds=int(
            getattr(
                spec.quality,
                "freshness_window_seconds",
                0,
            ) or 0
        ),
        freshness_fields=list(
            getattr(
                spec.quality,
                "freshness_fields",
                [],
            ) or []
        ),
        min_freshness_rate=float(
            getattr(
                spec.quality,
                "min_freshness_rate",
                1.0,
            )
        ),
    )

    runner = PipelineRunner(
        scraper, extractor,
        client_id=client_id,
        context=context,
        network_manager=_build_network_manager(scraper, injected),
        quality_rules=user_quality,
    )
    result = await runner.run(spec, output_path=output_path)

    # --- Record per-domain outcomes for future runs ---
    try:
        from src.discovery.domain_reputation import build_supabase_store
        from collections import defaultdict

        rep_store = build_supabase_store()

        # Count records per source_url in the final dataset
        counts: dict[str, int] = defaultdict(int)
        for rec in result.records:
            src = rec.get("source_url")
            if src:
                counts[src] += 1

        # Also mark every URL we fetched but got zero records from
        fetched_urls: set[str] = set()
        for st in result.security_traces:
            if st.url:
                fetched_urls.add(st.url)

        url_to_count: dict[str, int] = {}
        for u in fetched_urls:
            url_to_count[u] = counts.get(u, 0)

        rep_store.record_run(url_to_count)
        trace.append(_evt(
            "LEARN",
            f"Recorded outcomes for {len(url_to_count)} domain(s).",
            "info",
        ))
    except Exception as e:
        trace.append(_evt(
            "LEARN",
            f"Domain-outcome recording skipped: {type(e).__name__}: {e}",
            "warning",
        ))

    return result, trace




# ---------------------------------------------------------------------------
# Grid / Google Maps lead generation
# ---------------------------------------------------------------------------

def build_grid_preview(
    entity: str,
    city: str,
    llm_router=None,
    max_areas: int = 15,
) -> tuple[list[str], str]:
    """
    Called by Plan Review to populate the neighborhoods textarea.
    Returns (areas, error_message).
    """
    if not entity or not city:
        return [], ""
    query = f"{entity} in {city}"
    return preview_grid_areas(query, llm_router=llm_router, max_areas=max_areas)


def _leads_to_ui_result(
    leads: LeadsResult,
    prompt: str,
    trace: list[TraceEvent],
) -> UiRunResult:
    """Convert a LeadsResult into the UI's UiRunResult shape."""
    trace = list(trace)

    if leads.areas_searched:
        trace.append(_evt(
            "DISCOVER",
            f"Searched {leads.areas_searched} neighborhood(s).",
            "info",
        ))

    if leads.duplicates_removed:
        trace.append(_evt(
            "DEDUPE",
            f"Merged + removed {leads.duplicates_removed} duplicate lead(s).",
            "info",
        ))

    trace.append(_evt(
        "EXTRACT",
        f"Scraped {leads.total_scraped} raw lead(s).",
        "info",
    ))

    if leads.filtered_no_website:
        trace.append(_evt(
            "FILTER",
            f"Kept {leads.total_returned} with no website "
            f"({leads.filtered_no_website} had a site, filtered out).",
            "info",
        ))
    elif leads.total_scraped > 0 and leads.total_scraped == leads.total_returned:
        trace.append(_evt(
            "FILTER",
            "No website filter applied — returned every scraped lead.",
            "info",
        ))

    for err in leads.per_area_errors[:5]:
        trace.append(_evt("WARN", err, "warning"))

    if leads.error:
        trace.append(_evt("ERROR", leads.error, "error"))

    trace.append(_evt(
        "DONE",
        f"Run complete — {leads.total_returned} lead(s).",
        "success" if leads.succeeded else "warning",
    ))

    # new_count semantics:
    #   - With lead memory ON: use the actual new count from memory.
    #   - With lead memory OFF: fall back to total_returned, so legacy
    #     runs still show "N new" in the badge row.
    effective_new = (
        leads.new_count if (leads.discovery_mode or leads.existing_count)
        else leads.total_returned
    )

    discovery_warning = getattr(leads, "discovery_warning", "") or ""
    if discovery_warning:
        trace.append(_evt("DISCOVERY", discovery_warning, "warning"))

    return UiRunResult(
        mode="real" if leads.succeeded else "demo",
        records=leads.records,
        record_count=leads.total_returned,
        quality_score=1.0 if leads.succeeded else 0.0,
        quality_passed=leads.succeeded,
        confidence_mean=None,
        source_count=leads.areas_searched or 1,
        new_count=effective_new,
        changed_count=0,
        removed_count=0,
        existing_count=leads.existing_count,
        discovery_mode=leads.discovery_mode,
        discovery_warning=discovery_warning,
        trace=trace,
        warnings=list(leads.per_area_errors),
        error=leads.error,
        spec=None,
        workbook_path=leads.output_path,
        receipt_signature=None,
    )

_NO_WEBSITE_PATTERNS = (
    "without website", "without a website", "no website",
    "no websites", "without sites", "no site",
    "doesn't have a website", "dont have a website",
    "don't have a website", "without web",
)



def _wants_no_website(spec: TaskSpec) -> bool:
    """
    Detect the "I only want businesses with no website" intent from
    either the natural-language prompt or the constraints filters.
    """
    haystack = " ".join([
        (spec.natural_language_prompt or ""),
        (spec.constraints.filters or ""),
    ]).lower()
    return any(p in haystack for p in _NO_WEBSITE_PATTERNS)


def _run_grid_path(
    spec: TaskSpec,
    areas: list[str],
    trace: list[TraceEvent],
    output_path=None,
    client_id: str = "default",
    discovery_mode: bool = False,
) -> UiRunResult:
    """Execute the grid-search branch of a lead-gen task."""
    trace = list(trace)
    no_website_only = _wants_no_website(spec)

    trace.append(_evt(
        "GRID",
        f"Running grid search across {len(areas)} neighborhood(s)."
        + ("  Filter: no-website only." if no_website_only else ""),
        "info",
    ))

    if discovery_mode:
        trace.append(_evt(
            "DISCOVERY",
            "Discovery mode ON — only NEW leads will be returned. "
            "Quantity may be lower than a normal run.",
            "warning",
        ))
    else:
        trace.append(_evt(
            "DISCOVERY",
            "Monitoring mode — every lead is returned and tagged "
            "NEW or EXISTING relative to previous runs for this client.",
            "info",
        ))

    base_query = spec.natural_language_prompt or ""
    entity = spec.entities[0].entity_name if spec.entities else ""
    city = spec.constraints.geography or ""

    # Build the lead memory store on the fly. Any failure degrades to
    # None — the run proceeds without memory, same as before this feature.
    lead_memory = None
    try:
        from src.discovery.lead_memory import build_supabase_store
        lead_memory = build_supabase_store()
    except Exception as e:
        trace.append(_evt(
            "DISCOVERY",
            f"Lead memory unavailable ({type(e).__name__}: {e}); "
            f"continuing without it.",
            "warning",
        ))

    try:
        leads = run_grid_leads_query(
            base_query=base_query,
            areas=areas,
            no_website_only=no_website_only,
            max_areas=max(1, len(areas)),
            output_path=output_path,
            discovery_mode=discovery_mode,
            client_id=client_id,
            lead_memory=lead_memory,
        )
        
    except GmapsScraperUnavailable as e:
        trace.append(_evt(
            "ERROR",
            f"Google Maps scraper unavailable: {e} "
            "(is Docker running? try: docker start aces-gmaps)",
            "error",
        ))
        return _demo_result(base_query, "", client_id, trace,
                            reason="gmaps scraper unavailable")
    except Exception as e:
        trace.append(_evt(
            "ERROR",
            f"Grid run failed: {type(e).__name__}: {e}",
            "error",
        ))
        return _demo_result(base_query, "", client_id, trace,
                            reason=f"grid error: {type(e).__name__}")

    return _leads_to_ui_result(leads, prompt=base_query, trace=trace)