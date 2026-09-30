"""
Google Maps leads pipeline — spec §10, §14.

Runs a lead-gen query end to end against the local gosom scraper, then
normalizes rows into ACES lead fields, optionally filters to businesses
without a website, and optionally writes the result to disk.

Kept separate from GmapsClient (the HTTP wrapper) so the pipeline logic
is unit-testable without a live container.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from src.discovery.website_classifier import classify_website, has_real_website
from src.discovery.gmaps_client import (
    GmapsClient, GmapsError, GmapsScraperUnavailable,
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class LeadsResult:
    keywords: list[str]
    total_scraped: int = 0
    total_returned: int = 0
    filtered_no_website: int = 0
    records: list[dict] = field(default_factory=list)
    output_path: Optional[str] = None
    error: str = ""

    # Grid-search fields (default 0/empty for single-query runs)
    areas_searched: int = 0
    duplicates_removed: int = 0
    per_area_errors: list[str] = field(default_factory=list)

    # Lead-memory fields (populated when lead_memory is passed in)
    discovery_mode: bool = False
    new_count: int = 0
    existing_count: int = 0        # count from the FULL scrape, not just kept
    unverified_count: int = 0
    discovery_warning: str = ""

    @property
    def succeeded(self) -> bool:
        """A run succeeded if it completed without error.

        Note: discovery mode can legitimately return zero records (no new
        leads for this client). That is a successful run, not a failure.
        Callers who care about "got records" should check `total_returned`.
        """
        return not self.error


# ---------------------------------------------------------------------------
# Column mapping — gosom CSV header → ACES lead field
# ---------------------------------------------------------------------------

_COLUMN_MAP = {
    "title":         "business_name",
    "address":       "address",
    "phone":         "phone",
    "website":       "website",
    "email":         "email",
    "category":      "category",
    "review_rating": "rating",
    "review_count":  "review_count",
    "latitude":      "latitude",
    "longitude":     "longitude",
    "place_id":      "place_id",
    "link":          "source_url",
}


def _normalize_row(row: dict) -> dict:
    """
    Map a raw gosom CSV row into ACES lead fields.

    `has_website` reflects whether the business has a REAL website —
    social pages (facebook.com, wa.me, linktr.ee, foodpanda, ...) do not
    count. The raw URL is still preserved in the `website` column, and
    `website_type` records which bucket it fell into.
    """
    out: dict = {}
    for src_key, dst_key in _COLUMN_MAP.items():
        v = row.get(src_key)
        out[dst_key] = v if v is not None else ""

    raw_website = str(row.get("website") or "").strip()
    out["has_website"] = has_real_website(raw_website)
    out["website_type"] = classify_website(raw_website)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_leads_query(
    keywords: list[str],
    no_website_only: bool = False,
    output_path: Optional[str | Path] = None,
    client: Optional[GmapsClient] = None,
    timeout_seconds: Optional[float] = None,
    collect_emails: bool = False,
) -> LeadsResult:
    """
    Full lead-gen cycle:

        1. Scrape Google Maps for the given keywords.
        2. Normalize rows into ACES lead fields.
        3. Optionally filter to businesses with no listed website.
        4. Optionally write the result to disk.
    """
    client = client or GmapsClient()
    result = LeadsResult(keywords=list(keywords))

    try:
        scrape_result = client.scrape(
            keywords=keywords,
            name="aces-leads",
            email=collect_emails,
            timeout_seconds=timeout_seconds,
        )
    except GmapsScraperUnavailable as e:
        result.error = f"scraper unavailable: {e}"
        return result
    except GmapsError as e:
        result.error = f"scrape failed: {e}"
        return result

    if not scrape_result.succeeded:
        result.error = (
            scrape_result.error
            or f"job ended with status {scrape_result.status!r}"
        )
        return result

    raw = scrape_result.records
    result.total_scraped = len(raw)

    normalized = [_normalize_row(r) for r in raw]

    if no_website_only:
        filtered = [r for r in normalized if not r["has_website"]]
        result.filtered_no_website = len(normalized) - len(filtered)
    else:
        filtered = normalized

    result.records = filtered
    result.total_returned = len(filtered)

    if output_path is not None and filtered:
        from src.output.formats import write_records
        written = write_records(filtered, output_path)
        result.output_path = str(written.path)

    return result