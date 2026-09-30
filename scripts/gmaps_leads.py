"""
CLI for Google Maps lead queries.

Single-query:
    python -m scripts.gmaps_leads "pizza shops in Lahore" --no-website --out leads.xlsx

Grid search (splits Lahore into neighborhoods, merges, dedupes):
    python -m scripts.gmaps_leads "pizza shops in Lahore" --grid --no-website --out leads.xlsx

Discovery mode (only new leads for this client):
    python -m scripts.gmaps_leads "pizza shops in Lahore" --grid --discovery-mode --client-id acme

Monitoring mode (mark every lead as NEW or EXISTING):
    python -m scripts.gmaps_leads "pizza shops in Lahore" --grid --monitoring --client-id acme

Explicit areas:
    python -m scripts.gmaps_leads "pizza shops in Lahore" --grid --areas "DHA,Gulberg,Johar Town" --out leads.xlsx
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.discovery.gmaps_leads import run_leads_query
from src.discovery.grid_search import run_grid_leads_query
from src.discovery.lead_memory import build_supabase_store


def _progress(idx: int, total: int, query: str) -> None:
    print(f"  [{idx}/{total}] {query}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Google Maps lead query.")
    parser.add_argument("keyword", help='e.g. "pizza shops in Lahore"')
    parser.add_argument(
        "--grid", action="store_true",
        help="Split the city into neighborhoods and merge results",
    )
    parser.add_argument(
        "--areas", default=None,
        help="Comma-separated area names (overrides LLM expansion, implies --grid)",
    )
    parser.add_argument(
        "--max-areas", type=int, default=10,
        help="Maximum neighborhoods to search in grid mode (default 10)",
    )
    parser.add_argument(
        "--no-website", action="store_true",
        help="Return only businesses with no listed website",
    )
    parser.add_argument(
        "--emails", action="store_true",
        help="Also try to collect emails (slower)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output path (.xlsx / .csv / .json). Optional.",
    )
    parser.add_argument(
        "--timeout", type=float, default=900,
        help="Max seconds to wait per area (default 900)",
    )
    parser.add_argument(
        "--client-id", default="default",
        help="Tenant scope for lead memory (default: 'default')",
    )
    parser.add_argument(
        "--discovery-mode", action="store_true",
        help=(
            "Only return leads this client has NEVER seen before. "
            "Quantity will be lower than a normal run — the strict filter "
            "removes anything already in lead memory."
        ),
    )
    parser.add_argument(
        "--monitoring", action="store_true",
        help=(
            "Return every lead, each tagged with diff_status = NEW or "
            "EXISTING relative to previous runs for this client."
        ),
    )
    args = parser.parse_args()

    if args.discovery_mode and args.monitoring:
        print("--discovery-mode and --monitoring are mutually exclusive.")
        return 2

    areas = None
    if args.areas:
        areas = [a.strip() for a in args.areas.split(",") if a.strip()]

    grid = args.grid or bool(areas)

    # Lead memory is opt-in: build it only when one of the two flags is set.
    lead_memory = None
    if args.discovery_mode or args.monitoring:
        try:
            lead_memory = build_supabase_store()
        except Exception as e:
            print(f"! Lead memory unavailable: {type(e).__name__}: {e}")
            print("  Continuing without memory (every lead treated as NEW).")
            lead_memory = None

    print(f"Querying Google Maps for: {args.keyword}")
    if grid:
        print(f"Grid mode: up to {args.max_areas} neighborhoods")
    if args.no_website:
        print("Filter: only businesses WITHOUT a website")
    if args.discovery_mode:
        print(
            f"DISCOVERY MODE ON (client={args.client_id}) — "
            f"only NEW leads will be returned; quantity may be lower."
        )
    elif args.monitoring:
        print(
            f"MONITORING MODE (client={args.client_id}) — "
            f"every lead is tagged NEW or EXISTING."
        )
    print("This can take several minutes...\n")

    if grid:
        result = run_grid_leads_query(
            base_query=args.keyword,
            areas=areas,
            no_website_only=args.no_website,
            max_areas=args.max_areas,
            output_path=args.out,
            timeout_per_area=args.timeout,
            collect_emails=args.emails,
            progress_callback=_progress,
            discovery_mode=args.discovery_mode,
            client_id=args.client_id,
            lead_memory=lead_memory,
        )
    else:
        result = run_leads_query(
            keywords=[args.keyword],
            no_website_only=args.no_website,
            output_path=args.out,
            timeout_seconds=args.timeout,
            collect_emails=args.emails,
        )

    print()
    if result.error:
        print(f"! {result.error}")

    if grid:
        print(f"Areas searched:      {result.areas_searched}")
    print(f"Scraped rows:        {result.total_scraped}")
    if result.duplicates_removed:
        print(f"Duplicates removed:  {result.duplicates_removed}")
    if result.filtered_no_website:
        print(f"No-website filtered: {result.filtered_no_website}")
    if result.discovery_mode:
        print(f"New leads:           {result.new_count}")
        print(f"Existing leads:      {result.existing_count}")
        if result.unverified_count:
            print(f"Unverified (no ID):  {result.unverified_count}")
        if result.discovery_warning:
            print(f"! {result.discovery_warning}")
    elif result.new_count or result.existing_count:
        print(f"NEW leads:           {result.new_count}")
        print(f"EXISTING leads:      {result.existing_count}")
    print(f"Returned:            {result.total_returned}")
    if result.output_path:
        print(f"Written to:          {result.output_path}")

    # -------- Helpful hint when discovery mode returns nothing --------
    # 0 new leads is a legitimate, successful outcome — it means this
    # client has already seen everything this search currently surfaces.
    # Say so explicitly, otherwise it looks like a failure.
    if (
        result.discovery_mode
        and result.total_returned == 0
        and result.total_scraped > 0
    ):
        print()
        print("This is expected — every lead from this search has already")
        print("been seen by this client. To find more, try one of:")
        print("  • adding more neighborhoods via --areas")
        print("  • using a broader --max-areas value (grid expansion)")
        print("  • searching a different keyword or city")
        print("  • switching to --monitoring to see the full list with")
        print("    each lead tagged NEW or EXISTING")

    if result.per_area_errors:
        print()
        print(f"Errors ({len(result.per_area_errors)}):")
        for e in result.per_area_errors[:5]:
            print(f"  - {e}")

    print()
    for r in result.records[:10]:
        addr = (r.get("address") or "")[:48]
        status = r.get("diff_status", "")
        tag = f"  [{status}]" if status else ""
        print(f"  • {r['business_name']}  |  {r['phone']}  |  {addr}{tag}")
    if len(result.records) > 10:
        print(f"  ... and {len(result.records) - 10} more")

    return 0 if result.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
