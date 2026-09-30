-- ============================================================================
-- Migration 003 — domain outcome tracking
--
-- Rationale: the pipeline discovers 10-20 candidate URLs per run, but
-- empirically only a subset produce records (Pakistani pharmacies do;
-- Amazon/Walmart/CVS return bot shells). Without memory, we keep
-- re-fetching domains that never work.
--
-- We store a small rolling record per domain so future runs can
-- deprioritize dead ends and prioritize sources that have worked.
-- ============================================================================

create table if not exists public.domain_outcomes (
    domain           text primary key,
    attempts         integer not null default 0,
    successful_runs  integer not null default 0,
    total_records    integer not null default 0,
    last_success_at  timestamptz,
    last_failure_at  timestamptz,
    updated_at       timestamptz not null default now()
);

create index if not exists idx_domain_outcomes_updated
    on public.domain_outcomes (updated_at desc);

-- RLS: this table is used only by the backend service-role client,
-- so we disable row-level security for simplicity in this milestone.
-- (In a future milestone we can scope by client_id like the other tables.)
alter table public.domain_outcomes disable row level security;