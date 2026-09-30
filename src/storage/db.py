import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
logger = logging.getLogger("db")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_client() -> Client:
    """
    Service-role Supabase client for BACKEND writes.

    Uses SUPABASE_SERVICE_KEY, which bypasses RLS. This client must never
    be exposed to a browser or to user-facing code paths. User-scoped
    reads go through src.auth.supabase_backend.SupabaseAuthBackend, which
    uses the publishable key + the user's own JWT so RLS applies.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env "
            "(backend writes need the service key; the publishable key "
            "is only for user-facing auth)."
        )
    return create_client(url, key)


def save_run_results(query: str, results: list[dict], client_id: str = "default"):
    """Saves each extracted item as a row in run_history, scoped to a client."""
    client = get_client()
    rows = [
        {"query": query, "source_url": item.get("source_url"), "data": item, "client_id": client_id}
        for item in results
    ]
    response = client.table("run_history").insert(rows).execute()
    logger.info(f"Saved {len(rows)} rows to run_history (client: {client_id})")
    return response


def get_tracked_sources(query: str, client_id: str = "default") -> list[str]:
    """Returns previously-used source URLs for this query, scoped to a client."""
    client = get_client()
    response = (
        client.table("tracked_sources")
        .select("url")
        .eq("query", query)
        .eq("client_id", client_id)
        .execute()
    )
    return [row["url"] for row in response.data]


def save_tracked_sources(query: str, urls: list[str], client_id: str = "default"):
    """Remembers which URLs were used for this query, scoped to a client."""
    client = get_client()
    rows = [{"query": query, "url": url, "client_id": client_id} for url in urls]
    client.table("tracked_sources").upsert(rows, on_conflict="query,url,client_id").execute()
    logger.info(f"Tracked {len(urls)} source(s) for '{query}' (client: {client_id})")


if __name__ == "__main__":
    test_results = [
        {"title": "Test Product", "price": "$9.99", "source_url": "https://example.com"}
    ]
    save_run_results("connectivity test", test_results)
    print("If no error appeared above, Supabase connection works.")