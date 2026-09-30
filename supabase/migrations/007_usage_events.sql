-- ============================================================================
-- Migration 007 — usage events
--
-- Every meaningful resource consumption writes a row here. Rows are
-- append-only — never updated, never deleted. Corrections happen by
-- writing a new compensating event.
--
-- Feeds:
--   - per-job cost reports
--   - per-client quotas / entitlements
--   - monthly usage summaries
--   - cost-per-validated-record metrics
--   - provider-selection evidence
--   - future billing
--
-- Safe to run on an existing database — this only adds a new table.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- 1. Table
-- ----------------------------------------------------------------------------
create table if not exists public.usage_events (
    event_id            text primary key,
    client_id           text not null default 'default',
    job_id              text not null default '',
    task_id             text not null default '',
    resource_type       text not null
        check (resource_type in (
            'page','browser_second','token','vision_call',
            'provider_credit','storage_byte','api_call'
        )),
    quantity            numeric(20,6) not null default 0,
    unit                text not null default 'unit',
    unit_cost_snapshot  numeric(20,10) not null default 0,
    total_cost_usd      numeric(20,8) not null default 0,
    provider            text not null default '',
    metadata            jsonb not null default '{}'::jsonb,
    occurred_at         timestamptz not null default now()
);


-- ----------------------------------------------------------------------------
-- 2. Indexes
-- ----------------------------------------------------------------------------

-- Per-client time-ordered queries (most common)
create index if not exists idx_usage_client_time
    on public.usage_events (client_id, occurred_at desc);

-- Per-job lookups (job cost reports)
create index if not exists idx_usage_job
    on public.usage_events (job_id)
    where job_id <> '';

-- Per-task lookups
create index if not exists idx_usage_task
    on public.usage_events (task_id)
    where task_id <> '';

-- Resource-type rollups
create index if not exists idx_usage_resource_time
    on public.usage_events (resource_type, occurred_at desc);


-- ----------------------------------------------------------------------------
-- 3. RPC: usage_summary
--
-- Aggregate usage for a client over a time window. Returns one row per
-- resource_type plus a total row with resource_type = '__total__'.
-- ----------------------------------------------------------------------------
create or replace function public.usage_summary(
    p_client_id text,
    p_start timestamptz,
    p_end timestamptz
)
returns table (
    resource_type text,
    event_count   bigint,
    total_quantity numeric,
    total_cost_usd numeric
)
language sql
stable
as $$
    with per_resource as (
        select
            resource_type,
            count(*)                as event_count,
            sum(quantity)           as total_quantity,
            sum(total_cost_usd)     as total_cost_usd
        from public.usage_events
        where client_id = p_client_id
          and occurred_at >= p_start
          and occurred_at < p_end
        group by resource_type
    ),
    totals as (
        select
            '__total__'             as resource_type,
            coalesce(sum(event_count), 0)      as event_count,
            coalesce(sum(total_quantity), 0)   as total_quantity,
            coalesce(sum(total_cost_usd), 0)   as total_cost_usd
        from per_resource
    )
    select * from per_resource
    union all
    select * from totals;
$$;


-- ----------------------------------------------------------------------------
-- 4. RPC: usage_batch_insert
--
-- Insert an array of usage events in one round-trip. Idempotent on
-- event_id — a duplicate insert is silently ignored.
-- ----------------------------------------------------------------------------
create or replace function public.usage_batch_insert(p_events jsonb)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    inserted integer := 0;
begin
    if p_events is null or jsonb_typeof(p_events) <> 'array' then
        return 0;
    end if;

    with raw as (
        select value as e
        from jsonb_array_elements(p_events)
    ),
    ins as (
        insert into public.usage_events (
            event_id, client_id, job_id, task_id,
            resource_type, quantity, unit,
            unit_cost_snapshot, total_cost_usd,
            provider, metadata, occurred_at
        )
        select
            e->>'event_id',
            coalesce(e->>'client_id', 'default'),
            coalesce(e->>'job_id', ''),
            coalesce(e->>'task_id', ''),
            coalesce(e->>'resource_type', 'page'),
            coalesce((e->>'quantity')::numeric, 0),
            coalesce(e->>'unit', 'unit'),
            coalesce((e->>'unit_cost_snapshot')::numeric, 0),
            coalesce((e->>'total_cost_usd')::numeric, 0),
            coalesce(e->>'provider', ''),
            coalesce(e->'metadata', '{}'::jsonb),
            coalesce((e->>'occurred_at')::timestamptz, now())
        from raw
        on conflict (event_id) do nothing
        returning 1
    )
    select count(*) into inserted from ins;

    return inserted;
end;
$$;


-- Backend-only table, accessed via the service-role key.
alter table public.usage_events disable row level security;