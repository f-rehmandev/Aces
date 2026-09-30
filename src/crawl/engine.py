"""
Crawl engine — spec §14.3A, §14.3B.
    
    Ties a CrawlFrontier, a CheckpointStore, and a fetch/extract pipeline
    into one resumable crawl loop.
    
    Relationship to PipelineRunner._crawl_with_frontier:

    This engine is a standalone utility. It is NOT currently wired
    into the production pipeline — PipelineRunner drives its own
    CrawlFrontier loop because every page must pass through
    `_scrape_source`, which fuses fetch, sanitisation, extraction,
    and security checks into one call. See that method's docstring
    for the full reasoning.

    This engine remains useful directly for callers who want a plain
    HTML crawler with resumable checkpoints and no extraction step.
    

Decoupling:
    The engine knows how to orchestrate a crawl but not how to fetch or
    extract. It receives:

        fetch_html(url) -> HTML str        (async callable)
        page_processor(url, html) -> (records, discovered_links)

    Production wires these to ScraperEngine and the LLM extractor via a
    small adapter. Tests inject fakes.

Resumption (§14.3B):
    If a checkpoint store is provided and a checkpoint exists for
    (client_id, task_id), the engine loads it and continues from where
    the previous run stopped. The provided seeds are ignored in that case.

Checkpointing:
    Every `checkpoint_interval` processed pages, the engine writes a new
    checkpoint. On successful completion, all checkpoints for the task
    are cleared (nothing to resume from).

Budget:
    Respects policy.max_pages, policy.max_bytes, and
    policy.max_wall_clock_seconds. When any is hit, the crawl stops
    cleanly, keeps its checkpoint, and returns with
    status="stopped_by_budget".
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Union

from src.crawl.checkpoint import CheckpointStore
from src.crawl.frontier import CrawlFrontier
from src.crawl.types import (
    CrawlCheckpoint, CrawlPolicy, CrawlState,
)


logger = logging.getLogger("crawl.engine")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CrawlResult:
    status: str                                    # completed | stopped_by_budget
    records: list[dict] = field(default_factory=list)
    targets_processed: int = 0
    targets_failed: int = 0
    targets_skipped: int = 0
    targets_policy_refused: int = 0
    budget_pages_used: int = 0
    budget_bytes_used: int = 0
    budget_wall_clock_used: float = 0.0
    resumed_from_checkpoint: bool = False
    last_checkpoint_id: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "records_count": len(self.records),
            "targets_processed": self.targets_processed,
            "targets_failed": self.targets_failed,
            "targets_skipped": self.targets_skipped,
            "targets_policy_refused": self.targets_policy_refused,
            "budget_pages_used": self.budget_pages_used,
            "budget_bytes_used": self.budget_bytes_used,
            "budget_wall_clock_used": self.budget_wall_clock_used,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "last_checkpoint_id": self.last_checkpoint_id,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

FetchFn = Callable[[str], Awaitable[str]]

# The page processor may be sync OR async. The engine awaits its
# return value only when it's awaitable, so callers can plug in
# either style — including an async extraction pipeline that runs
# the LLM in the same call.
PageProcessor = Callable[
    [str, str],
    Union[
        tuple[list[dict], list[str]],
        Awaitable[tuple[list[dict], list[str]]],
    ],
]


class CrawlEngine:
    """
    Orchestrates a crawl using an injected fetch callable and page processor.

    `fetch_html(url)` must be async and return an HTML string. Returning
    an empty string is treated as a fetch failure and retried per policy.

    `page_processor(url, html)` may be sync OR async, and must return
    (records, discovered_links). The engine awaits its return value
    only when it's awaitable. Raising any exception fails the target.
    """

    def __init__(
        self,
        fetch_html: FetchFn,
        page_processor: PageProcessor,
        policy: Optional[CrawlPolicy] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        checkpoint_interval: int = 20,
        clock: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ):
        self._fetch = fetch_html
        self._process = page_processor
        self.policy = policy or CrawlPolicy()
        self.checkpoints = checkpoint_store
        self.interval = max(1, int(checkpoint_interval))
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    async def crawl(
        self,
        client_id: str,
        task_id: str = "",
        seeds: Optional[list[str]] = None,
    ) -> CrawlResult:
        frontier = CrawlFrontier(policy=self.policy, clock=self._clock)
        result = CrawlResult(status="completed")

        pages_used = 0
        bytes_used = 0
        wall_clock_start = self._clock()

        # --- Resume from checkpoint if one exists for this task ---
        if self.checkpoints and task_id:
            cp = self.checkpoints.load_latest(client_id, task_id)
            if cp and cp.frontier:
                frontier.load_from_checkpoint(cp.frontier)
                pages_used = cp.budget_pages_used
                bytes_used = cp.budget_bytes_used
                wall_clock_start = self._clock() - cp.budget_wall_clock_used
                result.resumed_from_checkpoint = True
                logger.info(
                    f"Resumed crawl for task {task_id!r}: "
                    f"{len(cp.frontier)} target(s) restored"
                )

        # --- Otherwise seed fresh from the provided URLs ---
        if not result.resumed_from_checkpoint:
            for url in seeds or []:
                frontier.add(url, depth=0)

        if len(frontier) == 0:
            return result

        # --- Main loop ---
        pages_since_checkpoint = 0

        while frontier.has_pending():
            # Budget gate
            if self._budget_exceeded(pages_used, bytes_used, wall_clock_start):
                result.status = "stopped_by_budget"
                break

            target = frontier.next_target()

            if target is None:
                # Nothing ready: only RETRY_WAIT targets remain.
                delay = frontier.min_retry_delay()
                if delay is None:
                    break  # safety net; shouldn't be reachable
                await self._sleep(min(delay, 5.0))
                continue

            # Fetch
            try:
                html = await self._fetch(target.url)
            except Exception as e:
                logger.warning(f"Crawl fetch failed on {target.url}: {e}")
                self._handle_fetch_failure(frontier, target, str(e))
                continue

            if not html:
                self._handle_fetch_failure(frontier, target, "empty response")
                continue

            pages_used += 1
            bytes_used += len(html)

            # Extract
            try:
                outcome = self._process(target.url, html)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                page_records, links = outcome
            except Exception as e:
                logger.warning(f"Crawl processor failed on {target.url}: {e}")
                frontier.mark_failed(target, error=f"processor: {e}")
                continue

            result.records.extend(page_records)
            frontier.mark_processed(target, records_extracted=len(page_records))

            # Add discovered links
            if self.policy.discover_links and links:
                child_depth = target.depth + 1
                for link in links:
                    frontier.add(link, parent_url=target.url, depth=child_depth)

            # Periodic checkpoint
            pages_since_checkpoint += 1
            if (
                self.checkpoints
                and task_id
                and pages_since_checkpoint >= self.interval
            ):
                cid = self._write_checkpoint(
                    client_id, task_id, frontier,
                    pages_used, bytes_used, wall_clock_start,
                )
                if cid:
                    result.last_checkpoint_id = cid
                pages_since_checkpoint = 0

        # --- Wrap up ---
        stats = frontier.stats()
        result.targets_processed = stats.get("PROCESSED", 0)
        result.targets_failed = stats.get("FAILED", 0)
        result.targets_skipped = stats.get("SKIPPED", 0)
        result.targets_policy_refused = stats.get("POLICY_REFUSED", 0)
        result.budget_pages_used = pages_used
        result.budget_bytes_used = bytes_used
        result.budget_wall_clock_used = round(
            self._clock() - wall_clock_start, 3,
        )

        # Clean completion clears checkpoints — nothing to resume from.
        if (
            result.status == "completed"
            and self.checkpoints
            and task_id
        ):
            self.checkpoints.clear_for_task(client_id, task_id)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _budget_exceeded(
        self,
        pages_used: int,
        bytes_used: int,
        wall_clock_start: float,
    ) -> bool:
        if self.policy.max_pages and pages_used >= self.policy.max_pages:
            return True
        if self.policy.max_bytes and bytes_used >= self.policy.max_bytes:
            return True
        if (
            self.policy.max_wall_clock_seconds
            and (self._clock() - wall_clock_start)
                >= self.policy.max_wall_clock_seconds
        ):
            return True
        return False

    def _handle_fetch_failure(
        self,
        frontier: CrawlFrontier,
        target,
        error: str,
    ) -> None:
        if target.attempts < self.policy.max_attempts:
            # Exponential backoff, capped at 30 seconds.
            delay = min(2 ** (target.attempts - 1), 30)
            frontier.mark_retry(
                target, error=error, retry_after_seconds=delay,
            )
        else:
            frontier.mark_failed(target, error=error)

    def _write_checkpoint(
        self,
        client_id: str,
        task_id: str,
        frontier: CrawlFrontier,
        pages_used: int,
        bytes_used: int,
        wall_clock_start: float,
    ) -> str:
        stats = frontier.stats()
        cp = CrawlCheckpoint(
            client_id=client_id,
            task_id=task_id,
            frontier=frontier.snapshot(),
            budget_pages_used=pages_used,
            budget_bytes_used=bytes_used,
            budget_wall_clock_used=round(
                self._clock() - wall_clock_start, 3,
            ),
            stats_processed=stats.get("PROCESSED", 0),
            stats_failed=stats.get("FAILED", 0),
            stats_skipped=stats.get("SKIPPED", 0),
            stats_policy_refused=stats.get("POLICY_REFUSED", 0),
            policy=self.policy,
        )
        ok = self.checkpoints.save(cp)
        return cp.checkpoint_id if ok else ""


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio as _asyncio

    async def _run():
        # Fake fetch: canned HTML keyed on URL
        html_map = {
            "https://x.com/a": "<html>A <a href='/b'>b</a></html>",
            "https://x.com/b": "<html>B</html>",
        }
        fetched: list[str] = []

        async def fetch(url):
            fetched.append(url)
            return html_map.get(url, "")

        def process(url, html):
            records = [{"url": url, "size": len(html)}]
            links = []
            if "/a" in url:
                links.append("https://x.com/b")
            return records, links

        # Simple in-memory checkpoint store
        from src.crawl.checkpoint import CheckpointStore
        rows: dict = {}
        store = CheckpointStore(
            fetch_fn=lambda c, t: rows.get(f"{c}::{t}"),
            write_fn=lambda cp: rows.__setitem__(
                f"{cp['client_id']}::{cp['task_id']}", cp,
            ),
            clear_fn=lambda c, t: int(bool(rows.pop(f"{c}::{t}", None))),
        )

        engine = CrawlEngine(
            fetch_html=fetch,
            page_processor=process,
            policy=CrawlPolicy(max_pages=10, discover_links=True),
            checkpoint_store=store,
            checkpoint_interval=100,   # don't checkpoint in this small test
        )
        result = await engine.crawl("acme", "t-1", seeds=["https://x.com/a"])

        assert result.status == "completed"
        assert len(result.records) == 2
        assert result.targets_processed == 2
        assert set(fetched) == {"https://x.com/a", "https://x.com/b"}
        # Clean completion cleared the (empty) checkpoint
        assert store.load_latest("acme", "t-1") is None

        # ---- Budget stop keeps checkpoint ----
        rows.clear()
        engine2 = CrawlEngine(
            fetch_html=fetch,
            page_processor=process,
            policy=CrawlPolicy(max_pages=1, discover_links=True),
            checkpoint_store=store,
            checkpoint_interval=1,
        )
        r2 = await engine2.crawl("acme", "t-2", seeds=["https://x.com/a"])
        assert r2.status == "stopped_by_budget"
        assert r2.budget_pages_used == 1
        # A checkpoint should still be on disk
        assert store.load_latest("acme", "t-2") is not None
        assert r2.last_checkpoint_id != ""

        # ---- Resume with a fresh engine and larger budget ----
        # A resumed crawl uses a new engine whose policy has headroom;
        # budget usage is cumulative across resumes, so the previous run
        # consuming 1 of 1 pages is correct.
        engine3 = CrawlEngine(
            fetch_html=fetch,
            page_processor=process,
            policy=CrawlPolicy(max_pages=10, discover_links=True),
            checkpoint_store=store,
            checkpoint_interval=100,
        )
        r3 = await engine3.crawl("acme", "t-2", seeds=[])
        assert r3.resumed_from_checkpoint is True
        assert r3.status == "completed"
        assert any(
            rec.get("url") == "https://x.com/b" for rec in r3.records
        ), f"expected /b in resumed records, got {r3.records}"

        print("Crawl engine OK.")

    _asyncio.run(_run())