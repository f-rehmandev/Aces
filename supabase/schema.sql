-- Stores every extracted item from every run, for history/diffing later
create table if not exists run_history (
    id bigint generated always as identity primary key,
    query text not null,
    source_url text,
    data jsonb not null,
    created_at timestamptz not null default now()
);

-- Anti-bot strategy memory per domain (Phase 4 groundwork)
create table if not exists strategies (
    domain text primary key,
    strategy jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

-- Token/cost/execution tracking per run
create table if not exists audit_logs (
    id bigint generated always as identity primary key,
    event text not null,
    provider text,
    query text,
    details jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_run_history_query on run_history (query);
create index if not exists idx_run_history_created_at on run_history (created_at);

-- Remembers which source URLs were used for a given query, so repeat
-- searches compare apples-to-apples instead of hitting random new sources
create table if not exists tracked_sources (
    id bigint generated always as identity primary key,
    query text not null,
    url text not null,
    created_at timestamptz not null default now(),
    unique (query, url)
);

alter table tracked_sources enable row level security;

alter table tracked_sources add column if not exists client_id text not null default 'default';
alter table run_history add column if not exists client_id text not null default 'default';