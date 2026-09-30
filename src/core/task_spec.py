"""
TaskSpec — the single internal representation every input mode resolves to.
Per project knowledge Section 11.2. Deliberately plain dataclasses (not a
framework model) so it stays simple, debuggable, and JSON-serializable for
Supabase storage.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import uuid
import json


# ---------------------------------------------------------------------------
# Nested specs (spec §11.2)
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    name: str
    type: str = "text"                 # text | currency | date | phone | email | url | number | enum
    required: bool = False
    expected_source: str = "either"    # listing | detail | either
    normalizer: Optional[str] = None


@dataclass
class EntitySpec:
    entity_name: str = ""
    record_boundary_hint: str = ""     # "repeated card" | "row" | "single page"
    identity_hint: str = ""            # "url" | "sku" | "name"


@dataclass
class Target:
    start_urls: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    source_hint: str = ""


@dataclass
class Constraints:
    filters: str = ""
    language: str = ""
    geography: str = ""
    time_range: str = ""


@dataclass
class Navigation:
    pagination: str = "auto"           # auto | css | scroll | cursor | none
    detail_pages: str = "optional"     # required | optional | none
    max_depth: int = 2
    max_pages: int = 5
    # Opt-in crawl mode. When False (default), the pipeline fetches each
    # start_url exactly once, exactly as before. When True, it uses the
    # CrawlEngine: follows discovered links up to `max_depth`, respects
    # `max_pages`, dedupes URLs, and (if a checkpoint store is wired in)
    # resumes a crashed run from the last successful page.
    follow_links: bool = False
    # When crawling, restrict to the same domain as the seed URLs.
    # Set False to allow cross-domain discovery — rare, but occasionally
    # wanted for federated sources.
    same_domain_only: bool = True


@dataclass
class SourceRequirements:
    min_independent_sources: int = 1
    triangulation_required: bool = False
    preferred_domains: list[str] = field(default_factory=list)


@dataclass
class Quality:
    min_records: int = 1
    min_populated_field_pct: float = 0.5
    max_failed_page_pct: float = 0.3
    max_empty_page_pct: float = 0.5

    # Extended trust-layer quality controls
    max_source_disagreement_rate: float = 0.25

    # Freshness is opt-in: 0 means the rule is disabled.
    freshness_window_seconds: int = 0
    freshness_fields: list[str] = field(default_factory=list)

    # When freshness is enabled, all configured records are expected
    # to have a usable timestamp inside the configured window by default.
    min_freshness_rate: float = 1.0


@dataclass
class Output:
    format: str = "xlsx"               # xlsx | csv | json | parquet
    sheets: list[str] = field(default_factory=list)
    delivery_target: str = "local"


@dataclass
class Schedule:
    """
    Wire-format schedule — this is what `TaskSpec.to_dict()` serializes
    to and what Supabase stores.

    Note: a *runtime* Schedule also exists in `src.jobs.scheduler`.
    That one is consumed by `next_fire_time()` and carries the exact
    same cadence fields, but its contract is defined by the scheduler,
    not by the task's persisted shape. Keeping them separate means
    the persisted format can stay stable while the scheduler evolves.

    Bridge: `to_jobs_schedule()` converts one to the other.
    """
    cadence: str = "once"              # once | hourly | daily | weekly | interval | cron
    timezone: str = "UTC"

    # --- cadence config (consumed by src.jobs.scheduler.next_fire_time) ---
    run_at: Optional[str] = None                    # ISO 8601, for cadence="once"
    at_time: str = "09:00"                          # "HH:MM", for daily/weekly
    weekdays: list[str] = field(default_factory=list)  # ["mon","wed"] for weekly
    interval_seconds: int = 0                       # for cadence="interval"
    cron_expression: str = ""                       # "M H * * DOW" for cadence="cron"

    # --- runtime state (managed by SchedulerLoop) ---
    # `next_fire_at` and `last_fired_at` are the scheduler's own bookkeeping.
    # They're populated by the loop on its first tick and updated after
    # every firing. Callers should treat them as read-only.
    enabled: bool = True
    next_fire_at: Optional[str] = None              # ISO 8601 UTC
    last_fired_at: Optional[str] = None             # ISO 8601 UTC

    def to_jobs_schedule(self):
        """
        Convert to the richer Schedule in src.jobs.scheduler.

        Two Schedule dataclasses exist for historical reasons: this one
        is the wire format embedded in TaskSpec; the jobs version is
        what `next_fire_time()` consumes. This converter is the bridge.
        """
        from src.jobs.scheduler import (
            Cadence as JobsCadence,
            Schedule as JobsSchedule,
        )
        try:
            cadence = JobsCadence(self.cadence)
        except ValueError:
            cadence = JobsCadence.ONCE
        return JobsSchedule(
            cadence=cadence,
            timezone=self.timezone,
            run_at=self.run_at,
            at_time=self.at_time,
            weekdays=list(self.weekdays),
            interval_seconds=self.interval_seconds,
            cron_expression=self.cron_expression,
        )


@dataclass
class Compliance:
    user_authorization_declared: bool = False
    robots_policy: str = "respect"     # respect | ignore
    refusal_reason: str = ""           # set if the engine refused the task


@dataclass
class Budget:
    """
    Wire-format budget — the *limits* a user configures on a task,
    serialized into the persisted TaskSpec.

    A separate *runtime* Budget exists in `src.jobs.budget`: it has
    the same four limit fields plus live consumption counters
    (`llm_tokens_used`, `usd_used`, `scraperapi_credits_used`,
    `pages_used`, `wall_clock_used`) and a `soft_limit_fraction`.
    Those are per-run state and must not be persisted with the task.

    Bridge: `PipelineRunner._resolve_budget_tracker()` builds a
    runtime `BudgetTracker` from this wire Budget when no tracker was
    injected by the caller.
    """
    max_llm_tokens: int = 100_000
    max_usd: float = 0.50
    max_scraperapi_credits: int = 0
    max_pages: int = 5_000


# ---------------------------------------------------------------------------
# Top-level TaskSpec
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = "2.0"
    natural_language_prompt: str = ""           # preserved verbatim for audit

    target: Target = field(default_factory=Target)
    objective: str = "extract"                  # extract | monitor | compare | lead_gen
    entities: list[EntitySpec] = field(default_factory=list)
    fields: list[FieldSpec] = field(default_factory=list)
    constraints: Constraints = field(default_factory=Constraints)
    navigation: Navigation = field(default_factory=Navigation)
    source_requirements: SourceRequirements = field(default_factory=SourceRequirements)
    quality: Quality = field(default_factory=Quality)
    output: Output = field(default_factory=Output)
    schedule: Schedule = field(default_factory=Schedule)
    compliance: Compliance = field(default_factory=Compliance)
    budget: Budget = field(default_factory=Budget)

    # multi-tenant scoping (our addition — Part X)
    client_id: str = "default"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpec":
        """Rebuild a TaskSpec from a plain dict (e.g. loaded from Supabase)."""
        d = dict(data)

        nested_map = {
            "target": Target,
            "constraints": Constraints,
            "navigation": Navigation,
            "source_requirements": SourceRequirements,
            "quality": Quality,
            "output": Output,
            "schedule": Schedule,
            "compliance": Compliance,
            "budget": Budget,
        }
        for key, klass in nested_map.items():
            if isinstance(d.get(key), dict):
                d[key] = klass(**d[key])

        if isinstance(d.get("fields"), list):
            d["fields"] = [
                f if isinstance(f, FieldSpec) else FieldSpec(**f)
                for f in d["fields"]
            ]
        if isinstance(d.get("entities"), list):
            d["entities"] = [
                e if isinstance(e, EntitySpec) else EntitySpec(**e)
                for e in d["entities"]
            ]

        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    # ------------------------------------------------------------------
    # Backward-compatible convenience accessors
    # (callers written before the v2 restructure keep working)
    # ------------------------------------------------------------------
    @property
    def source_hint(self) -> str:
        return self.target.source_hint

    @property
    def entity_name(self) -> str:
        return self.entities[0].entity_name if self.entities else ""

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def min_records(self) -> int:
        return self.quality.min_records


if __name__ == "__main__":
    spec = TaskSpec(
        natural_language_prompt="find best 3 prices of wireless mouse",
        objective="compare",
        fields=[FieldSpec(name="title"), FieldSpec(name="price", type="currency")],
        quality=Quality(min_records=3),
    )
    print(spec.to_json())
    roundtrip = TaskSpec.from_dict(spec.to_dict())
    assert roundtrip.field_names == ["title", "price"]
    print("\nRound-trip OK.")