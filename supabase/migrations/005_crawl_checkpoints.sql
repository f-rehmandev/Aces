-- ============================================================================
-- Migration 005 — crawl checkpoints
--
-- Persists resumable state for crawl runs so a crashed/interrupted crawl
-- can resume from the last successful page instead of starting over.
--
-- Each row stores the full CrawlCheckpoint as JSONB. A crawl writes a new
-- row every N pages (N = checkpoint_interval). Resumption loads the most
-- recent checkpoint for (client_id, task_id).
--
-- Safe to run on an existing database — this only adds a new table.
-- ============================================================================


create table if not exists public.crawl_checkpoints (
    checkpoint_id  uuid primary key,
    client_id      text not null,
    task_id        text not null,
    created_at     timestamptz not null,
    data           jsonb not null
);

create index if not exists idx_crawl_checkpoints_client_task
    on public.crawl_checkpoints (client_id, task_id, created_at desc);

-- Helper: latest checkpoint for a task
create or replace function public.latest_crawl_checkpoint(
    p_client_id text,
    p_task_id text
)
returns setof public.crawl_checkpoints
language sql
stable
as $$
    select *
    from public.crawl_checkpoints
    where client_id = p_client_id
      and task_id = p_task_id
    order by created_at desc
    limit 1;
$$;

-- Helper: forget all checkpoints for a task (e.g. after a clean completion)
create or replace function public.clear_crawl_checkpoints(
    p_client_id text,
    p_task_id text
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    removed integer;
begin
    delete from public.crawl_checkpoints
    where client_id = p_client_id
      and task_id = p_task_id;
    get diagnostics removed = row_count;
    return removed;
end;
$$;

-- Backend-only table, accessed via the service-role key.
alter table public.crawl_checkpoints disable row level security;