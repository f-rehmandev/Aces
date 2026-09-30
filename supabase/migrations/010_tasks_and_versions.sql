-- ============================================================================
-- Migration 010 — persistent tasks + immutable task versions
-- ============================================================================

create table if not exists public.tasks (
    task_id         uuid primary key,
    client_id       text not null,
    current_version integer not null default 1,
    status          text not null default 'active',
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create index if not exists idx_tasks_client
    on public.tasks (client_id);

create index if not exists idx_tasks_updated
    on public.tasks (updated_at desc);


create table if not exists public.task_versions (
    task_id     uuid not null
        references public.tasks(task_id) on delete cascade,
    version     integer not null,
    client_id   text not null,
    spec        jsonb not null,
    created_at  timestamptz not null default now(),

    primary key (task_id, version)
);

create index if not exists idx_task_versions_client
    on public.task_versions (client_id);

create index if not exists idx_task_versions_task_version
    on public.task_versions (task_id, version desc);


drop trigger if exists trg_tasks_updated_at on public.tasks;

create trigger trg_tasks_updated_at
before update on public.tasks
for each row
execute function public.touch_updated_at();


-- ============================================================================
-- RLS
-- ============================================================================

alter table public.tasks enable row level security;
alter table public.task_versions enable row level security;


-- ---------------------------------------------------------------------------
-- tasks: members can read/insert/update; owners/admins can delete
-- client_id stores the existing workspace slug.
-- ---------------------------------------------------------------------------

drop policy if exists tasks_select on public.tasks;
create policy tasks_select
on public.tasks
for select
using (
    exists (
        select 1
        from public.clients c
        where c.slug = tasks.client_id
          and public.is_client_member(c.id)
    )
);


drop policy if exists tasks_insert on public.tasks;
create policy tasks_insert
on public.tasks
for insert
with check (
    exists (
        select 1
        from public.clients c
        where c.slug = tasks.client_id
          and public.is_client_member(c.id)
    )
);


drop policy if exists tasks_update on public.tasks;
create policy tasks_update
on public.tasks
for update
using (
    exists (
        select 1
        from public.clients c
        where c.slug = tasks.client_id
          and public.is_client_member(c.id)
    )
)
with check (
    exists (
        select 1
        from public.clients c
        where c.slug = tasks.client_id
          and public.is_client_member(c.id)
    )
);


drop policy if exists tasks_delete on public.tasks;
create policy tasks_delete
on public.tasks
for delete
using (
    exists (
        select 1
        from public.clients c
        where c.slug = tasks.client_id
          and public.has_client_role(
              c.id,
              array['owner', 'admin']::public.client_role[]
          )
    )
);


-- ---------------------------------------------------------------------------
-- task_versions: immutable after creation
-- ---------------------------------------------------------------------------

drop policy if exists task_versions_select on public.task_versions;
create policy task_versions_select
on public.task_versions
for select
using (
    exists (
        select 1
        from public.clients c
        where c.slug = task_versions.client_id
          and public.is_client_member(c.id)
    )
);


drop policy if exists task_versions_insert on public.task_versions;
create policy task_versions_insert
on public.task_versions
for insert
with check (
    exists (
        select 1
        from public.clients c
        where c.slug = task_versions.client_id
          and public.is_client_member(c.id)
    )
);