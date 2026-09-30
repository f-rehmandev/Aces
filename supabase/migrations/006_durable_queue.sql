-- ============================================================================
-- Migration 006 — durable job queue
--
-- Persists queued jobs so they survive worker restarts, enforces
-- idempotency, and supports atomic leasing via a Postgres RPC.
--
-- Two functions do the stateful work:
--   queue_enqueue      — idempotent insert
--   queue_lease_next   — recover expired leases + atomically claim the
--                        highest-priority QUEUED job
--
-- Everything else (heartbeat, complete, fail, requeue) is safe to do
-- with plain single-row updates from Python.
--
-- Safe to run on an existing database — this only adds a new table.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- 1. Table
-- ----------------------------------------------------------------------------
create table if not exists public.job_queue (
    job_id              text primary key,
    idempotency_key     text not null default '',
    priority            smallint not null default 2
        check (priority between 0 and 4),
    state               text not null default 'queued'
        check (state in (
            'queued','leased','running','completed','failed','dead_letter'
        )),
    payload             jsonb not null default '{}'::jsonb,
    client_id           text not null default 'default',
    label               text not null default '',
    attempts            integer not null default 0,
    max_attempts        integer not null default 3,
    last_error          text not null default '',
    leased_by           text not null default '',
    leased_at           timestamptz,
    lease_expires_at    timestamptz,
    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now()
);


-- ----------------------------------------------------------------------------
-- 2. Indexes
-- ----------------------------------------------------------------------------

-- Lease query: find next QUEUED job by priority then FIFO.
create index if not exists idx_job_queue_lease
    on public.job_queue (priority asc, created_at asc)
    where state = 'queued';

-- Expired lease recovery: scan leased/running jobs by expiry.
create index if not exists idx_job_queue_expiry
    on public.job_queue (lease_expires_at)
    where state in ('leased','running');

-- Idempotency: only one NON-TERMINAL job may share an idempotency_key.
-- Terminal jobs (completed/dead_letter) don't block re-submission of
-- the same key — that lets a user re-run a task after it finishes.
create unique index if not exists idx_job_queue_idem
    on public.job_queue (idempotency_key)
    where idempotency_key <> ''
      and state not in ('completed','dead_letter');

-- Dead letters list: newest first.
create index if not exists idx_job_queue_dead
    on public.job_queue (updated_at desc)
    where state = 'dead_letter';


-- ----------------------------------------------------------------------------
-- 3. RPC: queue_enqueue — idempotent insert
--
-- Returns the job_id that should be used. If the same idempotency_key
-- is already present as a non-terminal job, returns that job's id.
-- ----------------------------------------------------------------------------
create or replace function public.queue_enqueue(p_job jsonb)
returns text
language plpgsql
security definer
set search_path = public
as $$
declare
    v_job_id text := p_job->>'job_id';
    v_key text := coalesce(p_job->>'idempotency_key', '');
    v_existing text;
begin
    if v_job_id is null or v_job_id = '' then
        raise exception 'queue_enqueue: job_id is required';
    end if;

    if v_key <> '' then
        select job_id
          into v_existing
          from public.job_queue
         where idempotency_key = v_key
           and state not in ('completed','dead_letter')
         limit 1;

        if v_existing is not null then
            return v_existing;
        end if;
    end if;

    insert into public.job_queue (
        job_id, idempotency_key, priority, state, payload,
        client_id, label, attempts, max_attempts, last_error,
        leased_by, created_at, updated_at
    )
    values (
        v_job_id,
        v_key,
        coalesce((p_job->>'priority')::smallint, 2),
        coalesce(p_job->>'state', 'queued'),
        coalesce(p_job->'payload', '{}'::jsonb),
        coalesce(p_job->>'client_id', 'default'),
        coalesce(p_job->>'label', ''),
        coalesce((p_job->>'attempts')::integer, 0),
        coalesce((p_job->>'max_attempts')::integer, 3),
        coalesce(p_job->>'last_error', ''),
        '',
        now(),
        now()
    )
    on conflict (job_id) do nothing;

    return v_job_id;
end;
$$;


-- ----------------------------------------------------------------------------
-- 4. RPC: queue_lease_next — recover + atomic claim
--
-- Step 1: sweep expired leases back to QUEUED (or DEAD_LETTER if
--         attempts exhausted).
-- Step 2: FOR UPDATE SKIP LOCKED a QUEUED job at the highest priority
--         and mark it LEASED with a fresh expiry.
--
-- Returns the claimed job as JSON, or NULL if nothing is ready.
-- ----------------------------------------------------------------------------
create or replace function public.queue_lease_next(
    p_worker_id text,
    p_lease_seconds integer
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
    v_job public.job_queue;
    v_now timestamptz := now();
    v_expires timestamptz := now() + (p_lease_seconds || ' seconds')::interval;
begin
    -- Step 1: recover expired leases
    update public.job_queue
       set attempts = attempts + 1,
           leased_by = '',
           leased_at = null,
           lease_expires_at = null,
           state = case
               when attempts + 1 >= max_attempts then 'dead_letter'
               else 'queued'
           end,
           last_error = case
               when attempts + 1 >= max_attempts
                    then 'lease expired before completion'
               else 'lease expired; re-queued'
           end,
           updated_at = v_now
     where state in ('leased','running')
       and lease_expires_at is not null
       and lease_expires_at <= v_now;

    -- Step 2: claim the next QUEUED job
    with next_job as (
        select job_id
          from public.job_queue
         where state = 'queued'
         order by priority asc, created_at asc
         limit 1
         for update skip locked
    )
    update public.job_queue j
       set state = 'leased',
           leased_by = p_worker_id,
           leased_at = v_now,
           lease_expires_at = v_expires,
           updated_at = v_now
      from next_job
     where j.job_id = next_job.job_id
     returning j.* into v_job;

    if v_job.job_id is null then
        return null;
    end if;

    return to_jsonb(v_job);
end;
$$;


-- Backend-only table, accessed via the service-role key.
alter table public.job_queue disable row level security;