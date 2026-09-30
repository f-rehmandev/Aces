"""
End-to-end pipeline runner — the capstone.

Ties together everything built so far:
    1. resolve input (TaskSpec)
    2. fetch + security pipeline (§49–§52)
    3. clean + normalize (§27)
    4. triangulate across sources (§23), with reputation weighting
    5. compute per-record confidence (§25)
    6. record provenance for every published value (§26)
    7. update source reputation from consensus agreement (§24)
    8. evaluate quality (§22)
    9. detect changes against prior version (§29)
   10. write outputs (§32, §33)
   11. sign execution receipt (§34)
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from src.trust.quantity import attach_unit_price
from src.storage.db import get_tracked_sources, save_tracked_sources


from src.network.manager import NetworkManager
from src.network.scraperapi import ScraperAPIProvider

from src.assistant import _scrape_source
from src.core.task_spec import TaskSpec
from src.auth.context import ClientContext, anonymous_context
from src.trust.cleaning import clean_records
from src.trust.pii_guard import PIIGuard
from src.trust.triangulation import Triangulator, observations_from_records
from src.trust.reputation import ReputationStore
from src.trust.confidence import ConfidenceScorer, ConfidenceInputs, record_confidence
from src.trust.provenance import ProvenanceStore, make_record_id
from src.quality.rules import QualityRules
from src.quality.evaluator import QualityEvaluator
from src.quality.publication import PublicationGate, PublicationDecision
from src.history.dataset import VersionedDataset
from src.history.change import ChangeClassifier
from src.history.report import ChangeReportBuilder
from src.output.workbook import WorkbookBuilder, WorkbookResult
from src.output.receipt import ReceiptBuilder, ReceiptSigner
from src.security.trace import SecurityTrace
from src.alerts.engine import AlertEngine
from src.alerts.rules import AlertEvent
from src.observability.tracker import IncidentTracker
from src.usage.store import UsageStore
from src.usage.types import ResourceType, UsageEvent
from src.usage.entitlements import EntitlementEngine
from src.jobs.budget import BudgetTracker
from src.alerts.rules import FiredAlert
from src.integrations.registry import ConnectorRegistry


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    task_id: str
    records: list[dict]
    quality_passed: bool
    quality_score: float
    publication_decision: Optional[PublicationDecision]
    change_set_summary: dict
    workbook: Optional[WorkbookResult]
    receipt_signature: Optional[str]
    security_traces: list[SecurityTrace] = field(default_factory=list)
    confidence_mean: float = 0.0
    provenance: list[dict] = field(default_factory=list)
    reputation_updates: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    client_id: str = ""
    is_authenticated: bool = False

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "records_count": len(self.records),
            "quality_passed": self.quality_passed,
            "quality_score": self.quality_score,
            "confidence_mean": self.confidence_mean,
            "publication_decision": (
                self.publication_decision.to_dict()
                if self.publication_decision else None
            ),
            "change_set_summary": self.change_set_summary,
            "workbook": self.workbook.to_dict() if self.workbook else None,
            "receipt_signature": self.receipt_signature,
            "reputation_updates": dict(self.reputation_updates),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    def __init__(
        self,
        scraper,
        extractor,
        client_id: str = "default",
        quality_rules: Optional[QualityRules] = None,
        reputation_store: Optional[ReputationStore] = None,
        receipt_signer: Optional[ReceiptSigner] = None,
        record_id_fn: Optional[Callable[[dict], str]] = None,
        context: Optional[ClientContext] = None,
        network_manager=None,
        use_production_network: bool = False,
        alert_engine: Optional[AlertEngine] = None,
        incident_tracker: Optional[IncidentTracker] = None,
        connector_registry: Optional[ConnectorRegistry] = None,
        usage_store: Optional[UsageStore] = None,
        job_id: str = "",
        entitlement_engine: Optional[EntitlementEngine] = None,
        budget_tracker: Optional[BudgetTracker] = None,
        checkpoint_store=None,
    ):
        self.scraper = scraper
        self.extractor = extractor

        # When a ClientContext is provided, it is the source of truth for
        # tenant scoping. `client_id` is kept as a fallback for anonymous
        # / local-dev runs and backward compatibility.
        if context is not None:
            self.context = context
            self.client_id = context.client_id or client_id
        else:
            self.context = anonymous_context()
            self.client_id = client_id

        # Track whether the caller explicitly injected rules. If they
        # didn't, `run()` derives them from `task.quality` so §11.2's
        # per-task thresholds (min_records, min_populated_field_pct, ...)
        # are actually enforced on the run.
        self._quality_rules_injected = quality_rules is not None
        self.quality_rules = quality_rules or QualityRules(min_records=1)
        self.reputation_store = reputation_store or ReputationStore()
        self.receipt_signer = receipt_signer
        self.record_id_fn = record_id_fn or make_record_id
        self.confidence_scorer = ConfidenceScorer()

        # Alert → incident bridge (§30A + §40.6). Both must be set for
        # the bridge to activate — supplying only one is a no-op, which
        # keeps every existing caller working unchanged.
        self.alert_engine = alert_engine
        self.incident_tracker = incident_tracker
        self.connector_registry = connector_registry

        # --- usage metering (§41.4) ---
        # The store is optional; when None, no usage is recorded and
        # the pipeline behaves exactly as before. `job_id` is used to
        # attribute usage events to a specific job (empty string means
        # "attribute to the task, not a job").
        self.usage_store = usage_store
        self.job_id = job_id

        # --- entitlement + budget gates (§47B, §38) ---
        # Both optional. When None, no gating occurs and the pipeline
        # behaves exactly as before.
        self.entitlement_engine = entitlement_engine
        self.budget_tracker = budget_tracker
                # --- persistent crawl checkpointing (§14.3B) ---
        # Optional so existing callers/tests remain unchanged.
        self.checkpoint_store = checkpoint_store

        # Network layer:
        #   - explicit manager  → use it (production or injected fake)
        #   - use_production_network=True → full tier chain
        #   - otherwise → Playwright-only (default; keeps tests isolated)
        if network_manager is not None:
            self.network_manager = network_manager
        elif use_production_network:
            from src.network.manager import build_production_manager
            self.network_manager = build_production_manager(scraper)
        else:
            self.network_manager = NetworkManager(scraper, None)

    # ------------------------------------------------------------------
    # Per-run budget (§38)
    # ------------------------------------------------------------------
    def _resolve_budget_tracker(self, task: TaskSpec):
        """
        Return the BudgetTracker to use for this run.

        Behaviour:
          - If the caller injected one at construction (tests, custom
            callers), use it unchanged. Callers own its lifetime.
          - Otherwise build a fresh BudgetTracker from `task.budget` so
            `max_pages`, `max_usd`, `max_llm_tokens` and
            `max_scraperapi_credits` on the TaskSpec are actually
            enforced on this run.

        Wall-clock and soft-limit fields fall back to sensible defaults
        because TaskSpec.budget deliberately keeps only the four
        primitives a user would set.
        """
        if self.budget_tracker is not None:
            return self.budget_tracker

        from src.jobs.budget import Budget, BudgetTracker

        return BudgetTracker(Budget.from_wire(task.budget))

    # ------------------------------------------------------------------
    # Entitlement gate (§47B)
    # ------------------------------------------------------------------
    async def _check_entitlement(
        self,
        task: TaskSpec,
        requested_pages: int,
        warnings: list[str],
    ) -> bool:
        """
        Pre-flight entitlement check. Returns True if the run may proceed,
        False if it must be refused.

        A refused run is NOT a failure — it's a policy decision. The
        caller (JobExecutor / API) surfaces the refusal as a distinct
        state; the pipeline just stops cleanly with a clear reason in
        `warnings`.

        Failures inside the entitlement engine are warnings, not refusals:
        if the store is temporarily down, we let the run proceed rather
        than blocking all traffic. The next successful check catches up.
        """
        if self.entitlement_engine is None:
            return True
        if requested_pages <= 0:
            return True

        try:
            decision = await self.entitlement_engine.check(
                self.client_id,
                ResourceType.PAGE.value,
                requested=float(requested_pages),
            )
        except Exception as e:
            warnings.append(
                f"entitlement check failed: {type(e).__name__}: {e}"
            )
            return True

        if not decision.allowed:
            warnings.append(
                f"run refused: entitlement denied — {decision.reason}"
            )
            return False

        if decision.warning:
            warnings.append(f"entitlement warning: {decision.warning}")

        return True

    # ------------------------------------------------------------------
    # Budget circuit breaker (§38)
    # ------------------------------------------------------------------
    def _budget_gate(
        self,
        pages_attempted: int,
        warnings: list[str],
    ) -> bool:
        """
        Called before each fetch. Returns True if the run may continue,
        False if the budget circuit breaker has tripped.

        On the first soft-limit hit, we append one warning and keep
        going. When the hard limit trips, we stop cleanly. The soft
        warning is not re-emitted if the previous page already logged
        it — that's what `_seen_soft_warnings` tracks.
        """
        if self.budget_tracker is None:
            return True

        state = self.budget_tracker.check()
        if state.tripped:
            warnings.append(
                f"budget circuit breaker tripped ({state.reason}); "
                f"stopped after {pages_attempted} page(s)"
            )
            return False

        if state.soft_warning and not getattr(
            self, "_seen_soft_warning", False,
        ):
            warnings.append(
                f"budget soft limit reached: "
                f"{', '.join(state.soft_reasons)}"
            )
            self._seen_soft_warning = True

        return True

    # ------------------------------------------------------------------
    # Crawl integration (§14.3A-C)
    # ------------------------------------------------------------------
    @staticmethod
    def _discover_links(html: str, base_url: str) -> list[str]:
        """
        Extract http(s) links from HTML, resolved against base_url.

        Called on the *sanitized* HTML produced by `_scrape_source`, so
        hidden honeypot links have already been stripped. Policy
        filtering (same-domain, include/exclude, depth) is applied
        downstream by the CrawlFrontier.
        """
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        links: list[str] = []
        if not html:
            return links
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return links

        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href:
                continue
            try:
                absolute = urljoin(base_url, href)
            except Exception:
                continue
            if absolute.startswith(("http://", "https://")):
                links.append(absolute)
        return links

    async def _crawl_with_frontier(
        self,
        source_urls: list[str],
        query: str,
        field_names: list[str],
        policy,
        traces: list[SecurityTrace],
        per_source_records: dict[str, list[dict]],
        warnings: list[str],
        task_id: str = "",
    ) -> tuple[int, int]:
        """
        BFS crawl driven by CrawlFrontier.

        Returns (pages_attempted, pages_succeeded). The caller's
        `traces`, `per_source_records`, and `warnings` are mutated in
        place — this is a helper on the same run, not an isolated unit.

        Relationship to `src.crawl.engine.CrawlEngine`:

            CrawlEngine is a *standalone utility* — a fetch(url) and
            process(url, html) pair that drives the frontier, applies
            checkpointing, and honours the budget. It is tested and
            usable directly (e.g. by a future HTML-only link-health
            checker), but its contract fuses fetch and extraction
            into two separate calls.

            This pipeline loop is richer than CrawlEngine's contract
            because every page must go through `_scrape_source`,
            which fuses fetch + SSRF check + DOM sanitize + honeypot
            scan + LLM extraction + injection guard into one atomic
            call. Splitting that would require an extraction cache
            keyed on URL identity, which introduces a stale-cache
            failure mode on retry that the current code does not
            have.

            Both implementations use `CrawlFrontier` (the pure state
            machine) and `CheckpointStore` (durable progress). Only
            this one is wired into production. That's a deliberate
            choice, not an oversight.
        """
        from src.crawl.frontier import CrawlFrontier

        from src.crawl.types import CrawlCheckpoint

        frontier = CrawlFrontier(policy=policy)

        # Resume from the latest durable checkpoint when available.
        resumed = False
        if self.checkpoint_store is not None and task_id:
            checkpoint = self.checkpoint_store.load_latest(
                self.client_id,
                task_id,
            )
            if checkpoint is not None:
                for target in checkpoint.frontier:
                    frontier.add(
                        target.url,
                        parent_url=target.parent_url,
                        depth=target.depth,
                    )
                    restored = frontier.get(target.url)
                    if restored is not None:
                        restored.state = target.state
                        restored.attempts = target.attempts
                        restored.discovered_at = target.discovered_at
                        restored.last_attempt_at = target.last_attempt_at
                        restored.error = target.error
                        restored.records_extracted = target.records_extracted
                        restored.notes = target.notes

                pages_attempted = checkpoint.budget_pages_used
                resumed = True
            else:
                for u in source_urls:
                    frontier.add(u, depth=0)
                pages_attempted = 0
        else:
            for u in source_urls:
                frontier.add(u, depth=0)
            pages_attempted = 0

        pages_succeeded = 0

        # Persist state after each processed page. This is intentionally
        # conservative for the first production integration: every page
        # boundary becomes a recovery point.
        async def save_checkpoint() -> None:
            if self.checkpoint_store is None or not task_id:
                return

            checkpoint = CrawlCheckpoint(
                client_id=self.client_id,
                task_id=task_id,
                frontier=list(frontier.snapshot()),
                budget_pages_used=pages_attempted,
                budget_bytes_used=0,
                budget_wall_clock_used=0.0,
                stats_processed=sum(
                    1
                    for target in frontier.all_targets()
                    if target.state.value == "PROCESSED"
                ),
                stats_failed=sum(
                    1
                    for target in frontier.all_targets()
                    if target.state.value == "FAILED"
                ),
                stats_skipped=sum(
                    1
                    for target in frontier.all_targets()
                    if target.state.value == "SKIPPED"
                ),
                stats_policy_refused=sum(
                    1
                    for target in frontier.all_targets()
                    if target.state.value == "POLICY_REFUSED"
                ),
                last_successful_url=(
                    next(
                        (
                            target.url
                            for target in reversed(frontier.all_targets())
                            if target.state.value == "PROCESSED"
                        ),
                        "",
                    )
                ),
                policy=policy,
            )
            self.checkpoint_store.save(checkpoint)

        while frontier.has_pending():
            # --- page-count gate (§14.3A) ---
            # The CrawlFrontier tracks state but does not enforce the
            # policy's max_pages — that's the engine's job, and since
            # we're driving the frontier ourselves, it's ours.
            if policy.max_pages and pages_attempted >= policy.max_pages:
                warnings.append(
                    f"crawl stopped: reached max_pages={policy.max_pages}"
                )
                break

            # --- budget circuit breaker (§38) ---
            if not self._budget_gate(pages_attempted, warnings):
                break

            target = frontier.next_target()
            if target is None:
                break

            trace = SecurityTrace(url=target.url)
            pages_attempted += 1

            try:
                outcome = await _scrape_source(
                    self.scraper, self.extractor,
                    target.url, query,
                    fields=field_names, notes="", trace=trace,
                    network_manager=self.network_manager,
                    return_html=True,
                )
                items, html = outcome  # type: ignore[misc]
            except Exception as e:
                warnings.append(f"scrape failed on {target.url}: {e}")
                items, html = [], ""

            # Charge the page against the budget regardless of outcome —
            # a failed fetch still consumed real network cost.
            if self.budget_tracker is not None:
                self.budget_tracker.consume(pages=1)

            traces.append(trace)
            per_source_records[target.url] = items
            if items:
                pages_succeeded += 1

            frontier.mark_processed(target, records_extracted=len(items))
            await save_checkpoint()

            # --- discover and enqueue new links ---
            if html:
                for link in self._discover_links(html, target.url):
                    frontier.add(
                        link,
                        parent_url=target.url,
                        depth=target.depth + 1,
                    )
        # A fully drained frontier means the crawl completed cleanly.
        # There is nothing left to resume, so remove durable checkpoints.
        if (
            self.checkpoint_store is not None
            and task_id
            and not frontier.has_pending()
        ):
            self.checkpoint_store.clear_for_task(
                self.client_id,
                task_id,
            )
        return pages_attempted, pages_succeeded

    # ------------------------------------------------------------------
    # Usage metering (§41.4)
    # ------------------------------------------------------------------
    async def _record_run_usage(
        self,
        task: TaskSpec,
        pages_attempted: int,
        pages_succeeded: int,
        warnings: list[str],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        llm_calls: int = 0,
        browser_seconds: float = 0.0,
        browser_calls: int = 0,
        provider_credits: int = 0,
        provider_credits_breakdown: Optional[dict] = None,
        provider_calls: int = 0,
        vision_calls: int = 0,
        storage_bytes: int = 0,
        storage_writes: int = 0,
    ) -> None:
        """
        Write usage events for this run.

        Events written:
            PAGE             — total page count (when > 0)
            TOKEN            — total LLM tokens (when > 0)
            BROWSER_SECOND   — total browser wall-clock seconds (when > 0)
            PROVIDER_CREDIT  — total paid-provider credits (when > 0),
                               per-provider breakdown in metadata

        Never raises: metering failures land in `warnings`. A broken
        usage store must not be able to break the pipeline.
        """
        if self.usage_store is None:
            return

        events: list[UsageEvent] = []

        if pages_attempted > 0:
            events.append(UsageEvent(
                client_id=self.client_id,
                job_id=self.job_id,
                task_id=task.task_id,
                resource_type=ResourceType.PAGE,
                quantity=float(pages_attempted),
                metadata={"succeeded": pages_succeeded},
            ))

        total_tokens = max(0, int(input_tokens)) + max(0, int(output_tokens))
        if total_tokens > 0:
            events.append(UsageEvent(
                client_id=self.client_id,
                job_id=self.job_id,
                task_id=task.task_id,
                resource_type=ResourceType.TOKEN,
                quantity=float(total_tokens),
                metadata={
                    "input_tokens": max(0, int(input_tokens)),
                    "output_tokens": max(0, int(output_tokens)),
                    "llm_calls": max(0, int(llm_calls)),
                },
            ))

        if browser_seconds and browser_seconds > 0:
            events.append(UsageEvent(
                client_id=self.client_id,
                job_id=self.job_id,
                task_id=task.task_id,
                resource_type=ResourceType.BROWSER_SECOND,
                quantity=float(browser_seconds),
                metadata={"browser_calls": max(0, int(browser_calls))},
            ))

        if provider_credits and provider_credits > 0:
            events.append(UsageEvent(
                client_id=self.client_id,
                job_id=self.job_id,
                task_id=task.task_id,
                resource_type=ResourceType.PROVIDER_CREDIT,
                quantity=float(provider_credits),
                metadata={
                    "provider_calls": max(0, int(provider_calls)),
                    "breakdown": dict(provider_credits_breakdown or {}),
                },
            ))

        if vision_calls and vision_calls > 0:
            events.append(UsageEvent(
                client_id=self.client_id,
                job_id=self.job_id,
                task_id=task.task_id,
                resource_type=ResourceType.VISION_CALL,
                quantity=float(vision_calls),
            ))

        if storage_bytes and storage_bytes > 0:
            events.append(UsageEvent(
                client_id=self.client_id,
                job_id=self.job_id,
                task_id=task.task_id,
                resource_type=ResourceType.STORAGE_BYTE,
                quantity=float(storage_bytes),
                metadata={"writes": max(0, int(storage_writes))},
            ))

        if not events:
            return

        try:
            await self.usage_store.record_batch(events)
        except Exception as e:
            warnings.append(
                f"usage record failed: {type(e).__name__}: {e}"
            )

    # ------------------------------------------------------------------
    # Alert → Incident bridge (§30A + §40.6)
    # ------------------------------------------------------------------

    async def _route_alerts_to_connectors(
        self,
        fired: list[FiredAlert],
        warnings: list[str],
    ) -> None:
        """
        Deliver fired alerts to every notification connector reachable
        from the registry. Uses AlertRouter, which resolves each alert's
        targets (or broadcasts to all notification connectors when no
        rule-specific targets are set).

        Failures never propagate — they land in `warnings`. Observability
        must not be able to crash the pipeline it observes.
        """
        if not fired or self.connector_registry is None:
            return

        # Lazy import so pipeline_runner doesn't pull in the whole
        # integrations package unless routing is actually used.
        from src.alerts.routing import AlertRouter

        router = AlertRouter(
            self.connector_registry, client_id=self.client_id,
        )

        try:
            results = await router.route_many(fired)
        except Exception as e:
            warnings.append(
                f"alert routing failed: {type(e).__name__}: {e}"
            )
            return

        delivered = sum(1 for r in results if r.any_succeeded)
        undelivered = sum(1 for r in results if not r.any_succeeded)

        if delivered:
            warnings.append(
                f"delivered {delivered} alert(s) to notification connector(s)"
            )
        if undelivered:
            warnings.append(
                f"{undelivered} alert(s) did not reach any connector"
            )

    async def _bridge_alerts_to_incidents(
        self,
        task: TaskSpec,
        records: list[dict],
        quality_result,
        confidence_mean: float,
        conflict_count: int,
        warnings: list[str],
    ) -> list[FiredAlert]:
        """
        Evaluate this run's outcome against the alert engine, then route
        any fired alerts through the incident tracker.

        Two directions:
            - Clean run + no fired alerts → resolve prior incidents that
              mention this task. (The underlying problem went away.)
            - Alerts fired → open/touch incidents via the tracker.

        Failures inside the bridge never crash the run — everything is
        appended to `warnings` and swallowed. Observability must not be
        able to break the pipeline it observes.
        """
        if self.alert_engine is None or self.incident_tracker is None:
            return []

        event = AlertEvent(
            kind="job_completed",
            client_id=self.client_id,
            task_id=task.task_id,
            job_id=task.task_id,
            payload={
                "records": len(records),
                "quality_score": quality_result.score,
                "confidence_mean": confidence_mean,
                "conflicts": conflict_count,
            },
        )

        try:
            fired = self.alert_engine.evaluate(event)
        except Exception as e:
            warnings.append(
                f"alert engine failed: {type(e).__name__}: {e}"
            )
            return []

        # --- Case 1: clean run, no alerts → resolve prior incidents ---
        if quality_result.passed and not fired:
            try:
                resolved = await self.incident_tracker.on_job_succeeded(
                    task.task_id,
                )
            except Exception as e:
                warnings.append(
                    f"incident resolve failed: {type(e).__name__}: {e}"
                )
                return []
            if resolved:
                warnings.append(
                    f"auto-resolved {resolved} prior incident(s) after "
                    f"successful run"
                )
            return []

        # --- Case 2: alerts fired → open/touch incidents ---
        if fired:
            try:
                incident_ids = await self.incident_tracker.on_alerts_fired(
                    fired, client_id=self.client_id,
                )
            except Exception as e:
                warnings.append(
                    f"incident open failed: {type(e).__name__}: {e}"
                )
                return []
            warnings.append(
                f"opened/touched {len(incident_ids)} incident(s) "
                f"from {len(fired)} fired alert(s)"
            )
            return fired

    # ------------------------------------------------------------------
    # Tenant scoping helpers
    # ------------------------------------------------------------------

    @property
    def is_authenticated(self) -> bool:
        return self.context.is_authenticated

    def assert_can_write(self) -> None:
        """
        Raise PermissionDenied if the context cannot write.
        Call this at the top of `run()` when the caller wants strict
        enforcement; the default for backward compat is to trust the caller.
        """
        from src.auth.models import PermissionDenied
        if not self.context.can_write():
            raise PermissionDenied(
                f"client {self.client_id!r} is read-only for this user"
            )

    # ------------------------------------------------------------------
    async def run(
        self,
        task: TaskSpec,
        output_path: Optional[str | Path] = None,
        previous_dataset: Optional[VersionedDataset] = None,
        default_currency: Optional[str] = None,
        default_country: Optional[str] = None,
        failed_page_count: int = 0,
    ) -> PipelineResult:
        started_at = _utc_iso()
        warnings: list[str] = []

        # Reset per-run usage meters at the start of every run so metering
        # is not cumulative across a reused instance (§41.4). `hasattr`
        # guards keep test fakes without these methods working.
        if hasattr(self.extractor, "reset_usage"):
            self.extractor.reset_usage()
        if hasattr(self.scraper, "reset_usage"):
            self.scraper.reset_usage()
        if hasattr(self.network_manager, "reset_usage"):
            self.network_manager.reset_usage()

        # Resolve the per-run budget. If no tracker was injected at
        # construction, build one from `task.budget` so max_pages /
        # max_usd / max_llm_tokens are actually enforced this run (§38).
        self.budget_tracker = self._resolve_budget_tracker(task)

        source_urls = list(task.target.start_urls)
        if not source_urls:
            warnings.append("task has no start URLs; nothing to fetch")
            return self._empty_result(task, warnings)

        # --- 0. Entitlement gate (§47B) ---
        # Refuse before touching the network if this run would exceed
        # the client's remaining quota.
        allowed = await self._check_entitlement(
            task, requested_pages=len(source_urls), warnings=warnings,
        )
        if not allowed:
            return self._empty_result(task, warnings)

        field_names = task.field_names or ["title", "price", "description"]
        query = task.natural_language_prompt or task.target.source_hint or "task"

        # --- 1. Security-aware scrape ---
        traces: list[SecurityTrace] = []
        per_source_records: dict[str, list[dict]] = {}
        pages_attempted = 0
        pages_succeeded = 0

        if task.navigation.follow_links:
            # --- crawl mode: BFS discovery via CrawlFrontier (§14.3A-C) ---
            from src.crawl.types import CrawlPolicy

            policy = CrawlPolicy(
                max_depth=task.navigation.max_depth,
                max_pages=task.navigation.max_pages,
                same_domain_only=task.navigation.same_domain_only,
                discover_links=True,
            )
            pages_attempted, pages_succeeded = await self._crawl_with_frontier(
                source_urls=source_urls,
                query=query,
                field_names=field_names,
                policy=policy,
                traces=traces,
                per_source_records=per_source_records,
                warnings=warnings,
                task_id=task.task_id,
            )
        else:
            # --- naive mode: one fetch per start URL (original behavior) ---
            for url in source_urls:
                # --- budget circuit breaker (§38) ---
                if not self._budget_gate(pages_attempted, warnings):
                    break

                trace = SecurityTrace(url=url)
                pages_attempted += 1
                try:
                    items = await _scrape_source(
                        self.scraper, self.extractor, url, query,
                        fields=field_names, notes="", trace=trace,
                        network_manager=self.network_manager,
                    )
                except Exception as e:
                    warnings.append(f"scrape failed on {url}: {e}")
                    items = []

                # Charge one page against the budget regardless of outcome.
                # A failed fetch is still a real cost (bandwidth, retries).
                if self.budget_tracker is not None:
                    self.budget_tracker.consume(pages=1)

                traces.append(trace)
                per_source_records[url] = items
                if items:
                    pages_succeeded += 1

        # --- 2. Cleaning + PII guard ---
        pii_guard = PIIGuard(declared_fields=field_names)
        for url in source_urls:
            raw = per_source_records.get(url, [])
            cleaned, clean_warnings = clean_records(
                raw, default_currency=default_currency,
                default_country=default_country,
            )
            warnings.extend(clean_warnings)
            pii_report = pii_guard.scan(cleaned)
            warnings.extend(
                f"PII redacted in {url}: {r.kind} @ {r.field_name}"
                for r in pii_report.redactions
            )
            per_source_records[url] = pii_report.records

        # --- 3. Triangulation + reputation feedback ---
        triangulated_records: list[dict] = []
        consensus_by_record: dict[str, list] = {}
        reputation_updates: dict[str, dict] = {}
        conflict_count = 0
        consensus: list = []

        if len(source_urls) >= 2 and any(per_source_records.values()):
            all_obs = []
            for url, recs in per_source_records.items():
                domain = _domain_of(url)
                trust = self.reputation_store.trust(domain)
                all_obs.extend(observations_from_records(
                    recs, source_domain=domain, trust_score=trust,
                    record_id_fn=self.record_id_fn,
                ))
            triangulator = Triangulator()
            consensus = triangulator.triangulate(all_obs)
            conflict_count = sum(len(c.conflicts) for c in consensus)
            for c in consensus:
                consensus_by_record.setdefault(c.record_id, []).append(c)
            triangulated_records = self._consensus_to_records(
                consensus, per_source_records,
            )
            reputation_updates = self._update_reputation(consensus, source_urls)
        else:
            for recs in per_source_records.values():
                triangulated_records.extend(recs)

        # --- 3b. Attach quantity + unit price to each record ---
        for rec in triangulated_records:
            attach_unit_price(rec)

        # --- 4. Per-record confidence scoring ---
        confidence_mean = self._score_confidence(
            triangulated_records, consensus_by_record, source_urls,
        )

        # --- 5. Provenance ---
        provenance = self._build_provenance(
            triangulated_records, per_source_records, source_urls,
        )

        # --- 6. Quality evaluation ---
        # If the caller didn't inject rules at construction, derive them
        # from `task.quality` so per-task thresholds (§11.2) are enforced
        # on this run. Injected rules win — callers who set them own them.
        if not self._quality_rules_injected:
            from src.quality.rules import QualityRules as _QR
            self.quality_rules = _QR(
                min_records=int(task.quality.min_records),
                min_populated_field_pct=float(
                    task.quality.min_populated_field_pct
                ),
                max_failed_page_pct=float(task.quality.max_failed_page_pct),
                max_empty_page_pct=float(task.quality.max_empty_page_pct),
                max_source_disagreement_rate=float(
                    getattr(
                        task.quality,
                        "max_source_disagreement_rate",
                        0.25,
                    )
                ),
                freshness_window_seconds=int(
                    getattr(
                        task.quality,
                        "freshness_window_seconds",
                        0,
                    )
                ),
                freshness_fields=list(
                    getattr(
                        task.quality,
                        "freshness_fields",
                        [],
                    ) or []
                ),
                min_freshness_rate=float(
                    getattr(
                        task.quality,
                        "min_freshness_rate",
                        1.0,
                    )
                ),
            )

        self.quality_rules.failed_page_count = failed_page_count
        self.quality_rules.total_page_count = pages_attempted
        self.quality_rules.empty_page_count = pages_attempted - pages_succeeded

        # Triangulation already computes field-level conflicts.
        # Promote that signal into the quality layer so a noisy
        # multi-source dataset can be quarantined before publication.
        if consensus:
            conflict_fields = sum(
                1 for c in consensus
                if c.conflicts
            )
            self.quality_rules.source_disagreement_rate = (
                conflict_fields / len(consensus)
            )
        else:
            self.quality_rules.source_disagreement_rate = None

        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate(triangulated_records, self.quality_rules)

        # Emit a warning per failed rule so the UI trace explains *why*
        if not quality_result.passed:
            for rule_name in quality_result.failed_rules:
                rule = next(
                    (r for r in quality_result.per_rule_results
                     if r.name == rule_name),
                    None,
                )
                detail = rule.detail if rule else ""
                warnings.append(
                    f"quality rule failed: {rule_name} ({detail})"
                )

        # --- 7. Publication gate ---
        gate = PublicationGate()
        decision = gate.evaluate(quality_result)

        # --- 8. Change detection ---
        previous_records: list[dict] = []
        if previous_dataset is not None:
            latest = previous_dataset.latest_valid()
            if latest:
                previous_records = latest.records
        change_set = ChangeClassifier().classify(
            previous_records, triangulated_records,
        )

        # --- 9. Workbook ---
        workbook_result: Optional[WorkbookResult] = None
        if output_path is not None and decision.allowed:
            wb = WorkbookBuilder()
            workbook_result = wb.build(
                output_path,
                triangulated_records,
                change_set=change_set,
                quality_result=quality_result,
                provenance_records=provenance,
                run_metadata={
                    "task_id": task.task_id,
                    "client_id": self.client_id,
                    "started_at": started_at,
                    "completed_at": _utc_iso(),
                    "pages_attempted": pages_attempted,
                    "pages_succeeded": pages_succeeded,
                    "pages_failed": failed_page_count,
                    "records": len(triangulated_records),
                    "quality_score": quality_result.score,
                    "confidence_mean": confidence_mean,
                    "change_summary": change_set.to_dict()["summary"],
                },
                errors=warnings,
            )

        # Capture the workbook's on-disk size so the run's storage usage
        # is metered (§41.4 — STORAGE_BYTE). Zero when no workbook was
        # written (quality gate failed, or output_path wasn't provided).
        storage_bytes = 0
        storage_writes = 0
        if workbook_result is not None:
            try:
                storage_bytes = workbook_result.path.stat().st_size
                storage_writes = 1
            except OSError:
                storage_bytes = 0
                storage_writes = 0

        # --- 10. Receipt ---
        receipt_signature: Optional[str] = None
        if self.receipt_signer is not None:
            builder = ReceiptBuilder(
                job_id=task.task_id, task_id=task.task_id,
                client_id=self.client_id, signer=self.receipt_signer,
            )
            _, receipt_signature = builder.build(
                started_at=started_at,
                completed_at=_utc_iso(),
                records=triangulated_records,
                pages_attempted=pages_attempted,
                pages_succeeded=pages_succeeded,
                pages_failed=failed_page_count,
                sources=[{"url": u} for u in source_urls],
                quality_score=quality_result.score,
            )

        # --- 11. Record usage (§41.4) ---
        # `getattr` guards keep test fakes without these attributes working.
        await self._record_run_usage(
            task=task,
            pages_attempted=pages_attempted,
            pages_succeeded=pages_succeeded,
            warnings=warnings,
            input_tokens=getattr(self.extractor, "total_input_tokens", 0),
            output_tokens=getattr(self.extractor, "total_output_tokens", 0),
            llm_calls=getattr(self.extractor, "call_count", 0),
            browser_seconds=getattr(
                self.scraper, "total_browser_seconds", 0.0,
            ),
            browser_calls=getattr(self.scraper, "browser_call_count", 0),
            provider_credits=getattr(
                self.network_manager, "total_provider_credits", 0,
            ),
            provider_credits_breakdown=getattr(
                self.network_manager, "provider_credits_by_provider", {},
            ),
            provider_calls=getattr(
                self.network_manager, "provider_call_count", 0,
            ),
            vision_calls=getattr(
                self.extractor, "vision_call_count", 0,
            ),
            storage_bytes=storage_bytes,
            storage_writes=storage_writes,
        )

        # --- 12. Bridge alerts → incidents (§30A + §40.6) ---
        fired_alerts = await self._bridge_alerts_to_incidents(
            task=task,
            records=triangulated_records,
            quality_result=quality_result,
            confidence_mean=confidence_mean,
            conflict_count=conflict_count,
            warnings=warnings,
        )

        # --- 13. Route fired alerts → notification connectors (§47A) ---
        await self._route_alerts_to_connectors(fired_alerts, warnings)

        return PipelineResult(
            task_id=task.task_id,
            records=triangulated_records,
            quality_passed=quality_result.passed,
            quality_score=quality_result.score,
            publication_decision=decision,
            change_set_summary=change_set.to_dict()["summary"],
            workbook=workbook_result,
            receipt_signature=receipt_signature,
            security_traces=traces,
            confidence_mean=confidence_mean,
            provenance=provenance,
            reputation_updates=reputation_updates,
            warnings=warnings,
            client_id=self.client_id,
            is_authenticated=self.context.is_authenticated,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _empty_result(self, task: TaskSpec, warnings: list[str]) -> PipelineResult:
        return PipelineResult(
            task_id=task.task_id,
            records=[],
            quality_passed=False,
            quality_score=0.0,
            publication_decision=None,
            change_set_summary={"new": 0, "removed": 0, "modified": 0,
                                 "unchanged": 0, "total": 0},
            workbook=None,
            receipt_signature=None,
            warnings=warnings,
        )

    def _consensus_to_records(self, consensus, per_source_records) -> list[dict]:
        base_by_id: dict[str, dict] = {}
        for recs in per_source_records.values():
            for r in recs:
                rid = self.record_id_fn(r)
                base_by_id.setdefault(rid, r)

        by_record: dict[str, dict] = {}
        for c in consensus:
            by_record.setdefault(c.record_id, {})[c.field_name] = c.consensus_value

        out: list[dict] = []
        for rid, fields in by_record.items():
            base = dict(base_by_id.get(rid, {}))
            for fname, value in fields.items():
                base[fname] = value
            out.append(base)
        return out

    def _score_confidence(
        self,
        records: list[dict],
        consensus_by_record: dict[str, list],
        source_urls: list[str],
    ) -> float:
        if not records:
            return 0.0

        scores: list[float] = []
        for rec in records:
            rid = self.record_id_fn(rec)
            consensus_list = consensus_by_record.get(rid, [])

            if consensus_list:
                # Average consensus strength across fields
                consensus_strength = sum(
                    c.confidence for c in consensus_list
                ) / len(consensus_list)
                cluster_count = max((c.total_clusters for c in consensus_list),
                                    default=1)
                mean_trust = self._mean_trust(source_urls)
                inputs = ConfidenceInputs(
                    consensus=round(consensus_strength, 3),
                    source_count=cluster_count,
                    mean_trust=mean_trust,
                    freshness=1.0,
                    validation_passed=True,
                )
            else:
                inputs = ConfidenceInputs(
                    source_count=1,
                    freshness=1.0,
                    validation_passed=True,
                )

            score = self.confidence_scorer.score(inputs)
            rec["confidence"] = score.value
            scores.append(score.value)

        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def _mean_trust(self, source_urls: list[str]) -> float:
        if not source_urls:
            return 0.5
        trusts = [self.reputation_store.trust(_domain_of(u)) for u in source_urls]
        return round(sum(trusts) / len(trusts), 3)

    def _build_provenance(
        self,
        records: list[dict],
        per_source_records: dict[str, list[dict]],
        source_urls: list[str],
    ) -> list[dict]:
        store = ProvenanceStore(run_id="", job_id="")
        for url in source_urls:
            recs = per_source_records.get(url, [])
            if not recs:
                continue
            store.record_batch(
                recs, source_url=url, extraction_method="pipeline",
                ignored_fields={"diff_status", "_numeric_price", "confidence"},
            )
        return store.to_list()

    def _update_reputation(self, consensus, source_urls) -> dict:
        """For each consensus, sources in the winning cluster agreed."""
        before = {_domain_of(u): self.reputation_store.trust(_domain_of(u))
                  for u in source_urls}

        for c in consensus:
            winners = set(c.winning_sources)
            dissenters = {s for conflict in c.conflicts for s in conflict.sources}

            for domain in winners:
                self.reputation_store.record_agreement(domain)
            for domain in dissenters:
                if domain not in winners:
                    self.reputation_store.record_disagreement(domain)

        after = {_domain_of(u): self.reputation_store.trust(_domain_of(u))
                 for u in source_urls}

        return {
            domain: {"before": before[domain], "after": after[domain]}
            for domain in before
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _domain_of(url: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(url).netloc or "").lower()


def run_pipeline(task: TaskSpec, scraper, extractor, **kwargs) -> PipelineResult:
    runner_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("client_id", "quality_rules", "reputation_store",
                              "receipt_signer", "record_id_fn")}
    call_kwargs = {k: v for k, v in kwargs.items()
                   if k in ("output_path", "previous_dataset",
                            "default_currency", "default_country",
                            "failed_page_count")}
    return asyncio.run(
        PipelineRunner(scraper, extractor, **runner_kwargs).run(task, **call_kwargs)
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class FakeScraper:
        def __init__(self, html_map): self.html_map = html_map
        async def fetch_html(self, url, timeout=None):
            return self.html_map.get(url, "")
        async def fetch_screenshot(self, url, timeout=None):
            return b""

    class FakeExtractor:
        def __init__(self, items_map): self.items_map = items_map
        def extract_list(self, html, instruction):
            for sentinel, items in self.items_map.items():
                if sentinel in html:
                    return list(items)
            return []
        def extract_from_image(self, img, instr):
            return []

    from src.core.task_spec import Target, FieldSpec

    scraper = FakeScraper({
        "https://93.184.216.34/a": "<html>SA</html>",
        "https://93.184.216.35/b": "<html>SB</html>",
    })
    extractor = FakeExtractor({
        "SA": [{"title": "Mouse", "price": "$10"}],
        "SB": [{"title": "Mouse", "price": "$10"}],
    })

    task = TaskSpec(
        natural_language_prompt="test",
        target=Target(start_urls=[
            "https://93.184.216.34/a",
            "https://93.184.216.35/b",
        ]),
        fields=[FieldSpec(name="title"), FieldSpec(name="price", type="currency")],
    )

    result = run_pipeline(task, scraper, extractor)
    print(f"records: {len(result.records)}")
    print(f"confidence_mean: {result.confidence_mean}")
    print(f"provenance entries: {len(result.provenance)}")
    print(f"reputation updates: {result.reputation_updates}")
    assert len(result.records) == 1
    assert result.confidence_mean > 0
    assert len(result.provenance) > 0

    print("Pipeline runner v2 OK.")