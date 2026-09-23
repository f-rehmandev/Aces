import logging
import re
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from storage.db import get_client

logger = logging.getLogger("diff_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _parse_price(price_str) -> float | None:
    if not price_str:
        return None
    match = re.search(r"[\d,]+\.?\d*", str(price_str))
    return float(match.group().replace(",", "")) if match else None


def get_previous_run(query: str, client_id: str = "default", before_run_at=None) -> list[dict]:
    """Fetches the most recent prior batch of results for this query+client from Supabase."""
    client = get_client()
    q = (
        client.table("run_history")
        .select("*")
        .eq("query", query)
        .eq("client_id", client_id)
        .order("created_at", desc=True)
    )
    response = q.limit(50).execute()
    rows = response.data

    if not rows:
        return []

    if before_run_at:
        rows = [r for r in rows if r["created_at"] < before_run_at]

    if not rows:
        return []

    latest_batch_time = rows[0]["created_at"]
    previous_batch = [r for r in rows if r["created_at"] == latest_batch_time]
    return [r["data"] for r in previous_batch]


def _identity_key(item: dict) -> str:
    """Items can be products, leads, jobs, etc. — use whichever identifying field is present."""
    return item.get("title") or item.get("business_name") or item.get("job_title") or item.get("name")


def compute_diff(current: list[dict], previous: list[dict]) -> list[dict]:
    """
    Compares current results against previous results (matched by a stable identity key).
    Adds a 'diff_status' field to each current item: New, Price Changed, or Unchanged.
    Also returns any Removed items (present before, missing now).
    """
    prev_by_title = {_identity_key(item): item for item in previous if _identity_key(item)}
    curr_titles = {_identity_key(item) for item in current if _identity_key(item)}

    annotated = []
    for item in current:
        title = _identity_key(item)
        prev_item = prev_by_title.get(title)

        if prev_item is None:
            item["diff_status"] = "New"
        else:
            old_price = _parse_price(prev_item.get("price"))
            new_price = _parse_price(item.get("price"))
            if old_price is not None and new_price is not None and old_price != new_price:
                item["diff_status"] = f"Price Changed (was {prev_item.get('price')})"
            else:
                item["diff_status"] = "Unchanged"
        annotated.append(item)

    removed = [
        {**item, "diff_status": "Removed"}
        for title, item in prev_by_title.items()
        if title not in curr_titles
    ]

    return annotated + removed


if __name__ == "__main__":
    # Quick manual test with fake data
    previous = [
        {"title": "Wireless Mouse A", "price": "$20.00"},
        {"title": "Wireless Mouse B", "price": "$30.00"},
    ]
    current = [
        {"title": "Wireless Mouse A", "price": "$25.00"},  # price changed
        {"title": "Wireless Mouse C", "price": "$15.00"},  # new
        # Mouse B is missing = removed
    ]
    result = compute_diff(current, previous)
    for r in result:
        print(r)