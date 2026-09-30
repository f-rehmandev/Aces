-- ============================================================================
-- Migration 008 — incidents + SLOs (spec §40.6)
--
-- Three tables:
--   incidents         — open and historical incidents, deduped by key
--   slo_definitions   — SLO configuration (rarely changes)
--   slo_samples       — append-only samples for each SLO
--
-- RPCs:
--   incident_open_or_touch  — atomically create or bump an open incident
--   slo_samples_batch       — batch-insert samples idempotent on sample_id
--
-- Safe to run on an existing database — this only adds new tables.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- 1. incidents
-- ----------------------------------------------------------------------------
create table if not exists public.incidents (
    incident_id         uuid primary key,
    dedup_key           text not null default '',
    title               text not null default '',
    description         text not null default '',
    severity            text not null default 'warning'
        check (severity in ('info','warning','critical')),
    status              text not null default 'open'
        check (status in (
            'open','acknowledged','mitigating','resolved','closed'
        )),
    failure_class       text not null default 'unknown'
        check (failure_class in (
            'network','auth','compliance','budget','provider',
            'data_quality','schema','infrastructure','security','unknown'
        )),
    detection_source    text not null default 'automatic'
        check (detection_source in (
            'alert_rule','slo_breach','circuit_breaker','operator','automatic'
        )),
    client_id           text not null default 'default',
    affected_jobs       jsonb not null default '[]'::jsonb,
    affected_domains    jsonb not null default '[]'::jsonb,
    acknowledged_by     text not null default '',
    mitigation          text not null default '',
    root_cause_note     text not null default '',
    linked_alert_ids    jsonb not null default '[]'::jsonb,
    linked_healing_ids  jsonb not null default '[]'::jsonb,
    linked_provider_events jsonb not null default '[]'::jsonb,
    started_at          timestamptz not null default now(),
    acknowledged_at     timestamptz,
    resolved_at         timestamptz,
    closed_at           timestamptz,
    updated_at          timestamptz not null default now(),
    occurrence_count    integer not null default 1,
    metadata            jsonb not null default '{}'::jsonb
);

-- Open incident lookup by dedup_key (partial unique so a dedup_key can
-- only be "in flight" once).
create unique index if not exists idx_incidents_open_dedup
    on public.incidents (dedup_key)
    where dedup_key <> ''
      and status not in ('resolved','closed');

-- Recent-first listings
create index if not exists idx_incidents_client_time
    on public.incidents (client_id, started_at desc);

create index if not exists idx_incidents_status_time
    on public.incidents (status, started_at desc);


-- ----------------------------------------------------------------------------
-- 2. slo_definitions
-- ----------------------------------------------------------------------------
create table if not exists public.slo_definitions (
    slo_id          uuid primary key,
    name            text not null,
    description     text not null default '',
    metric          text not null,
    unit            text not null default 'ratio'
        check (unit in ('ratio','seconds','count')),
    direction       text not null default 'at_least'
        check (direction in ('at_least','at_most')),
    target          numeric not null,
    window_days     integer not null default 7 check (window_days > 0),
    min_samples     integer not null default 30 check (min_samples > 0),
    scope           text not null default 'client'
        check (scope in ('client','system','domain')),
    scope_id        text not null default '',
    enabled         boolean not null default true,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create index if not exists idx_slo_scope
    on public.slo_definitions (scope, scope_id)
    where enabled = true;


-- ----------------------------------------------------------------------------
-- 3. slo_samples
-- ----------------------------------------------------------------------------
create table if not exists public.slo_samples (
    sample_id   uuid primary key,
    slo_id      uuid not null references public.slo_definitions(slo_id)
        on delete cascade,
    client_id   text not null default 'default',
    value       numeric not null,
    sampled_at  timestamptz not null default now(),
    metadata    jsonb not null default '{}'::jsonb
);

create index if not exists idx_slo_samples_window
    on public.slo_samples (slo_id, sampled_at desc);

create index if not exists idx_slo_samples_client
    on public.slo_samples (client_id, sampled_at desc);


-- ----------------------------------------------------------------------------
-- 4. RPC: incident_open_or_touch
--
-- Atomically:
--   - if an OPEN incident exists for (client_id, dedup_key), increment
--     occurrence_count and update the given fields
--   - otherwise insert a new incident
-- Returns the incident_id.
-- ----------------------------------------------------------------------------
create or replace function public.incident_open_or_touch(p_incident jsonb)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare
    v_id uuid;
    v_dedup text := coalesce(p_incident->>'dedup_key', '');
    v_client text := coalesce(p_incident->>'client_id', 'default');
begin
    if v_dedup <> '' then
        select incident_id into v_id
          from public.incidents
         where client_id = v_client
           and dedup_key = v_dedup
           and status not in ('resolved','closed')
         limit 1;

        if v_id is not null then
            update public.incidents
               set occurrence_count = occurrence_count + 1,
                   updated_at = now(),
                   severity = coalesce(p_incident->>'severity', severity),
                   affected_jobs = coalesce(
                       p_incident->'affected_jobs', affected_jobs),
                   affected_domains = coalesce(
                       p_incident->'affected_domains', affected_domains)
             where incident_id = v_id;
            return v_id;
        end if;
    end if;

    insert into public.incidents (
        incident_id, dedup_key, title, description,
        severity, status, failure_class, detection_source,
        client_id, affected_jobs, affected_domains,
        mitigation, root_cause_note, started_at, updated_at,
        occurrence_count, metadata
    )
    values (
        (p_incident->>'incident_id')::uuid,
        v_dedup,
        coalesce(p_incident->>'title', ''),
        coalesce(p_incident->>'description', ''),
        coalesce(p_incident->>'severity', 'warning'),
        coalesce(p_incident->>'status', 'open'),
        coalesce(p_incident->>'failure_class', 'unknown'),
        coalesce(p_incident->>'detection_source', 'automatic'),
        v_client,
        coalesce(p_incident->'affected_jobs', '[]'::jsonb),
        coalesce(p_incident->'affected_domains', '[]'::jsonb),
        coalesce(p_incident->>'mitigation', ''),
        coalesce(p_incident->>'root_cause_note', ''),
        coalesce((p_incident->>'started_at')::timestamptz, now()),
        now(),
        coalesce((p_incident->>'occurrence_count')::integer, 1),
        coalesce(p_incident->'metadata', '{}'::jsonb)
    )
    returning incident_id into v_id;

    return v_id;
end;
$$;


-- ----------------------------------------------------------------------------
-- 5. RPC: slo_samples_batch
--
-- Idempotent on sample_id.
-- ----------------------------------------------------------------------------
create or replace function public.slo_samples_batch(p_samples jsonb)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    inserted integer := 0;
begin
    if p_samples is null or jsonb_typeof(p_samples) <> 'array' then
        return 0;
    end if;

    with raw as (
        select value as e from jsonb_array_elements(p_samples)
    ),
    ins as (
        insert into public.slo_samples (
            sample_id, slo_id, client_id, value, sampled_at, metadata
        )
        select
            (e->>'sample_id')::uuid,
            (e->>'slo_id')::uuid,
            coalesce(e->>'client_id', 'default'),
            coalesce((e->>'value')::numeric, 0),
            coalesce((e->>'sampled_at')::timestamptz, now()),
            coalesce(e->'metadata', '{}'::jsonb)
        from raw
        on conflict (sample_id) do nothing
        returning 1
    )
    select count(*) into inserted from ins;

    return inserted;
end;
$$;


alter table public.incidents disable row level security;
alter table public.slo_definitions disable row level security;
alter table public.slo_samples disable row level security;