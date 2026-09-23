import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
import logging
from urllib.parse import urlparse
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from storage.db import get_client

logger = logging.getLogger("strategy_memory")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_STRATEGY = {"timeout": config.DEFAULT_TIMEOUT_MS}


def _domain_of(url: str) -> str:
    return urlparse(url).netloc


def get_strategy(url: str) -> dict:
    """Fetches the known-good strategy for this domain, or returns sensible defaults."""
    domain = _domain_of(url)
    client = get_client()
    response = client.table("strategies").select("*").eq("domain", domain).execute()

    if response.data:
        logger.info(f"Using saved strategy for {domain}: {response.data[0]['strategy']}")
        return response.data[0]["strategy"]

    return DEFAULT_STRATEGY.copy()


def save_strategy(url: str, strategy: dict):
    """Saves/updates the working strategy for this domain."""
    domain = _domain_of(url)
    client = get_client()
    client.table("strategies").upsert({"domain": domain, "strategy": strategy}).execute()
    logger.info(f"Saved strategy for {domain}: {strategy}")


def record_failure_and_adapt(url: str, current_strategy: dict) -> dict:
    """
    Called after a failure. Increases the timeout as the simplest adaptive
    strategy, saves it, and returns the new strategy to retry with.
    """
    domain = _domain_of(url)
    new_timeout = min(
        current_strategy.get("timeout", config.DEFAULT_TIMEOUT_MS) + config.STRATEGY_TIMEOUT_INCREMENT_MS,
        config.STRATEGY_MAX_TIMEOUT_MS,
    )
    new_strategy = {**current_strategy, "timeout": new_timeout}
    save_strategy(url, new_strategy)
    logger.info(f"Adapted strategy for {domain}: timeout now {new_timeout}ms")
    return new_strategy


if __name__ == "__main__":
    test_url = "https://www.daraz.pk/mouse/"
    strategy = get_strategy(test_url)
    print("Current strategy:", strategy)

    adapted = record_failure_and_adapt(test_url, strategy)
    print("Adapted strategy after simulated failure:", adapted)

    refetched = get_strategy(test_url)
    print("Re-fetched from Supabase:", refetched)


_last_request_time: dict[str, float] = {}  # in-process cache, domain -> timestamp


async def wait_for_rate_limit(url: str):
    """
    Enforces the domain's current pacing interval before a request goes out.
    Per project knowledge Section 15.6: starts at 1000ms, decays on success,
    doubles on repeated failure (handled via the strategy's own interval field).
    """
    import asyncio
    domain = _domain_of(url)
    strategy = get_strategy(url)
    interval_ms = strategy.get("pacing_interval_ms", 1000)

    last_time = _last_request_time.get(domain, 0)
    elapsed = (time.time() - last_time) * 1000
    if elapsed < interval_ms:
        wait_time = (interval_ms - elapsed) / 1000
        logger.info(f"Rate-limit pacing: waiting {wait_time:.1f}s before hitting {domain} again")
        await asyncio.sleep(wait_time)

    _last_request_time[domain] = time.time()


def record_rate_limit_hit(url: str):
    """Call this when a 429 or rate-limit signal is detected — doubles the pacing interval."""
    strategy = get_strategy(url)
    current_interval = strategy.get("pacing_interval_ms", 1000)
    new_interval = min(current_interval * 2, 30000)
    strategy["pacing_interval_ms"] = new_interval
    save_strategy(url, strategy)
    logger.warning(f"Rate limit hit on {_domain_of(url)} — pacing interval now {new_interval}ms")