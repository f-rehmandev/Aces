-- ============================================================================
-- Migration 011 — durable batches and chunks
--
-- Persists first-class Batch state and its individual chunks.
-- ============================================================================

create table if not exists public.batches (
    batch_id         text primary key,
    client_id        text not null,
    task_id          text not null,
    task_version     integer not null check (task_version >= 1),

    state            text not null check (
        state in (
            'QUEUED',
            'RUNNING',
            'PAUSED',
            'COMPLETED',
            'FAILED',
            'CANCELLED'
        )
    ),

    total_urls       integer not null default 0,
    unique_urls      integer not null default 0,
    completed_count  integer not null default 0,
    failed_count     integer not null default 0,
    skipped_count    integer not null default 0,
    paused_count     integer not null default 0,
    retry_count      integer not null default 0,

    chunk_size       integer not null check (chunk_size >= 1),
    input_manifest   jsonb not null default '{}'::jsonb,

    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now(),

    data             jsonb not null default '{}'::jsonb
);

create index if not exists idx_batches_client_task
    on public.batches (client_id, task_id, created_at desc);

create index if not exists idx_batches_state
    on public.batches (state);


create table if not exists public.batch_chunks (
    chunk_id        text primary key,

    batch_id        text not null
        references public.batches(batch_id)
        on delete cascade,

    chunk_index     integer not null check (chunk_index >= 0),

    state           text not null check (
        state in (
            'QUEUED',
            'RUNNING',
            'SUCCEEDED',
            'FAILED',
            'PAUSED',
            'CANCELLED'
        )
    ),

    attempts        integer not null default 0,
    max_attempts    integer not null default 1
        check (max_attempts >= 1),

    records_count   integer not null default 0,
    error           text not null default '',

    created_at      timestamptz,
    started_at      timestamptz,
    completed_at    timestamptz,

    data            jsonb not null default '{}'::jsonb
);

create unique index if not exists idx_batch_chunks_batch_index
    on public.batch_chunks (batch_id, chunk_index);

create index if not exists idx_batch_chunks_batch_state
    on public.batch_chunks (batch_id, state);


-- Backend-internal persistence.
alter table public.batches disable row level security;
alter table public.batch_chunks disable row level security;