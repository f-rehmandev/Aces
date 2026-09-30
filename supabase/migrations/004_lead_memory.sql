-- ============================================================================
-- Migration 004 — persistent lead memory
--
-- Remembers which Google Maps `place_id`s each client has already seen,
-- so a "discovery mode" run can return only genuinely-new leads, and a
-- "monitoring mode" run can mark every lead as NEW or EXISTING.
--
-- Keyed on (client_id, place_id): the same lead seen by two different
-- clients is remembered independently for each.
--
-- Safe to run on an existing database — this only adds a new table.
-- ============================================================================


create table if not exists public.lead_memory (
    id              bigint generated always as identity primary key,
    client_id       text not null,
    place_id        text not null,
    query_key       text not null default '',   -- which search first produced this lead
    first_seen_at   timestamptz not null default now(),
    last_seen_at    timestamptz not null default now(),
    unique (client_id, place_id)
);

create index if not exists idx_lead_memory_client
    on public.lead_memory (client_id);

create index if not exists idx_lead_memory_client_place
    on public.lead_memory (client_id, place_id);


-- ----------------------------------------------------------------------------
-- Batch-upsert helper.
--
-- One DB round-trip per batch. On conflict, only `last_seen_at` is updated;
-- `first_seen_at` and `query_key` keep their original values, so the
-- "when did this lead first appear, and from which search" record is preserved.
-- ----------------------------------------------------------------------------
create or replace function public.upsert_lead_memory(
    p_client_id text,
    p_place_ids text[],
    p_query_key text
)
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
    pid text;
begin
    if p_place_ids is null or array_length(p_place_ids, 1) is null then
        return;
    end if;

    foreach pid in array p_place_ids loop
        if pid is null or pid = '' then
            continue;
        end if;

        insert into public.lead_memory (client_id, place_id, query_key)
        values (p_client_id, pid, coalesce(p_query_key, ''))
        on conflict (client_id, place_id) do update
            set last_seen_at = now();
    end loop;
end;
$$;


-- This table is backend-only, accessed via the service-role key.
alter table public.lead_memory disable row level security;
