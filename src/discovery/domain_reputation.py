"""
Domain outcome memory — spec §20 (cross-domain learning).

Records which domains actually produce records. On future runs, sources
that have worked before get a higher URL score, and sources that
consistently return nothing get deprioritized.

Design:
    - Backend-agnostic: `fetch_fn` and `write_fn` are injected. Production
      uses Supabase; tests use in-memory fakes.
    - Score is a single 0..1 number derived from rolling success rate +
      average records. Domains with no history get a neutral 0.5.
    - Never raises on read failure — a broken DB must not break discovery.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlparse


logger = logging.getLogger("domain_reputation")


NEUTRAL_SCORE = 0.5


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class DomainStats:
    domain: str
    attempts: int = 0
    successful_runs: int = 0
    total_records: int = 0

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return NEUTRAL_SCORE
        return self.successful_runs / self.attempts


    @property
    def score(self) -> float:
        """
        A single 0..1 reputation score.

        - Zero history                 → NEUTRAL_SCORE (0.5)
        - Proven productive domain     → approaches 1.0
        - Proven empty / bot-shell     → approaches 0.0

        We use Bayesian smoothing on the success rate so one lucky hit
        (1/1) does not equal ten solid hits (10/10). The prior is 0.5
        and the smoothing constant is 2 — the same shape used for Wilson
        / Laplace smoothing in ranking systems.
        """
        if self.attempts == 0:
            return NEUTRAL_SCORE

        # --- smoothed success rate (Laplace-style) ---
        prior, k = 0.5, 2.0
        smoothed_success = (self.successful_runs + k * prior) / (self.attempts + k)

        # --- record-volume adjustment ---
        # A domain that returns lots of records per successful fetch is
        # worth more than one that barely ekes out a single row.
        if self.successful_runs > 0:
            avg = self.total_records / max(1, self.successful_runs)
            # avg 1 → ~0.0;  avg 5 → ~0.10;  avg 20+ → caps at 0.15
            import math
            bonus = min(0.15, math.log1p(avg) / math.log1p(20) * 0.15)
        else:
            bonus = -0.2   # never produced anything → active penalty

        return max(0.0, min(1.0, smoothed_success + bonus))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class DomainReputationStore:
    def __init__(
        self,
        fetch_fn: Callable[[str], Optional[dict]],
        write_fn: Callable[[str, dict], None],
    ):
        """
        `fetch_fn(domain) -> {"attempts": int, "successful_runs": int,
                              "total_records": int} | None`
        `write_fn(domain, stats_dict) -> None`
        """
        self._fetch = fetch_fn
        self._write = write_fn

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_stats(self, domain: str) -> DomainStats:
        try:
            row = self._fetch(domain)
        except Exception as e:
            logger.warning(f"domain reputation read failed for {domain}: {e}")
            return DomainStats(domain=domain)

        if not row:
            return DomainStats(domain=domain)

        return DomainStats(
            domain=domain,
            attempts=int(row.get("attempts", 0)),
            successful_runs=int(row.get("successful_runs", 0)),
            total_records=int(row.get("total_records", 0)),
        )

    def score(self, domain: str) -> float:
        return self.get_stats(domain).score

    def score_url(self, url: str) -> float:
        try:
            domain = (urlparse(url).netloc or "").lower()
        except Exception:
            return NEUTRAL_SCORE
        if not domain:
            return NEUTRAL_SCORE
        return self.score(domain)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def record_outcome(
        self,
        domain: str,
        records_produced: int,
    ) -> None:
        """Update the running stats for this domain after one fetch."""
        current = self.get_stats(domain)
        updated = {
            "attempts": current.attempts + 1,
            "successful_runs": current.successful_runs
                                + (1 if records_produced > 0 else 0),
            "total_records": current.total_records + max(0, records_produced),
        }
        try:
            self._write(domain, updated)
        except Exception as e:
            logger.warning(f"domain reputation write failed for {domain}: {e}")

    def record_run(self, url_to_record_count: dict[str, int]) -> None:
        """Batch: update every domain from one pipeline run."""
        per_domain: dict[str, int] = {}
        for url, count in url_to_record_count.items():
            try:
                domain = (urlparse(url).netloc or "").lower()
            except Exception:
                continue
            if not domain:
                continue
            per_domain[domain] = per_domain.get(domain, 0) + count
        for domain, total in per_domain.items():
            self.record_outcome(domain, total)


# ---------------------------------------------------------------------------
# URL re-ranking
# ---------------------------------------------------------------------------

def apply_reputation(
    urls: list[str],
    base_scores: list[float],
    reputation: DomainReputationStore,
    weight: float = 8.0,
) -> list[tuple[str, float]]:
    """
    Combine the URL-level score with the domain reputation score.

    reputation is in [0, 1]. We center it around 0.5 so a neutral domain
    contributes 0 to the total; well-performing domains get up to +weight/2,
    consistently-failing domains get down to -weight/2.

    Returns a list of (url, combined_score) pairs, sorted descending.
    """
    combined: list[tuple[str, float]] = []
    for url, base in zip(urls, base_scores):
        try:
            rep = reputation.score_url(url)
        except Exception:
            rep = NEUTRAL_SCORE
        adjustment = (rep - 0.5) * weight
        combined.append((url, base + adjustment))
    combined.sort(key=lambda t: -t[1])
    return combined


# ---------------------------------------------------------------------------
# Supabase-backed factory
# ---------------------------------------------------------------------------

def build_supabase_store() -> DomainReputationStore:
    """
    Build a store backed by the `domain_outcomes` table (service key).
    Falls back to a no-op store if Supabase is unreachable.
    """
    from src.storage.db import get_client

    def fetch(domain: str) -> Optional[dict]:
        client = get_client()
        resp = (
            client.table("domain_outcomes")
            .select("*")
            .eq("domain", domain)
            .limit(1)
            .execute()
        )
        data = getattr(resp, "data", None) or []
        return data[0] if data else None

    def write(domain: str, stats: dict) -> None:
        client = get_client()
        row = {"domain": domain, **stats}
        client.table("domain_outcomes").upsert(row, on_conflict="domain").execute()

    return DomainReputationStore(fetch_fn=fetch, write_fn=write)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- In-memory store ---
    rows: dict[str, dict] = {}

    def fetch(d): return rows.get(d)
    def write(d, stats): rows[d] = stats

    store = DomainReputationStore(fetch_fn=fetch, write_fn=write)

    # Neutral on unknown
    assert store.score("brand-new.com") == NEUTRAL_SCORE
    assert store.score_url("https://brand-new.com/x") == NEUTRAL_SCORE

    # Record successful run
    store.record_outcome("good.com", records_produced=5)
    assert store.score("good.com") > NEUTRAL_SCORE

    # Record failing run
    for _ in range(3):
        store.record_outcome("bad.com", records_produced=0)
    assert store.score("bad.com") < NEUTRAL_SCORE

    # Batch
    store.record_run({
        "https://good.com/a": 3,
        "https://good.com/b": 2,
        "https://new.com/c": 0,
    })
    assert store.get_stats("good.com").attempts == 2
    assert store.get_stats("new.com").attempts == 1

    # apply_reputation
    urls = [
        "https://good.com/p/x",
        "https://bad.com/p/y",
        "https://brand-new.com/p/z",
    ]
    base = [10.0, 10.0, 10.0]
    ranked = apply_reputation(urls, base, store)
    # good.com should outrank brand-new.com, which outranks bad.com
    order = [u for u, _ in ranked]
    assert order.index("https://good.com/p/x") < order.index("https://brand-new.com/p/z")
    assert order.index("https://brand-new.com/p/z") < order.index("https://bad.com/p/y")

    # Broken fetcher must not raise
    def broken_fetch(d): raise RuntimeError("db down")
    store2 = DomainReputationStore(fetch_fn=broken_fetch, write_fn=write)
    assert store2.score("x.com") == NEUTRAL_SCORE   # graceful

    print("Domain reputation OK.")