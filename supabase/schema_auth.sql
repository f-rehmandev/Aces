-- ============================================================================
-- ACES — Auth + multi-tenancy schema
-- Spec §43 (client workspaces & isolation)
--
-- Run this in the Supabase SQL editor after the base schema.sql.
-- Supabase manages `auth.users`; we only own application tables.
-- ============================================================================


-- ---------------------------------------------------------------------------
-- 1. clients — a tenant (one freelancer's project space, one team, etc.)
-- ---------------------------------------------------------------------------
create table if not exists public.clients (
    id              uuid primary key default gen_random_uuid(),
    name            text not null,
    slug            text unique not null,       -- URL-safe identifier
    created_at      timestamptz not null default now(),
    created_by      uuid references auth.users(id) on delete set null,
    -- plan / billing fields live in a separate table so this stays clean
    metadata        jsonb not null default '{}'::jsonb
);

create index if not exists idx_clients_created_by on public.clients (created_by);


-- ---------------------------------------------------------------------------
-- 2. client_members — which users belong to which clients, and with what role
-- ---------------------------------------------------------------------------
do $$ begin
    create type public.client_role as enum ('owner', 'admin', 'member', 'viewer');
exception
    when duplicate_object then null;
end $$;

create table if not exists public.client_members (
    client_id       uuid not null references public.clients(id) on delete cascade,
    user_id         uuid not null references auth.users(id) on delete cascade,
    role            public.client_role not null default 'member',
    invited_by      uuid references auth.users(id) on delete set null,
    joined_at       timestamptz not null default now(),
    primary key (client_id, user_id)
);

create index if not exists idx_client_members_user on public.client_members (user_id);


-- ---------------------------------------------------------------------------
-- 3. users — application profile mirroring auth.users
--    We don't store credentials here. Supabase owns that.
-- ---------------------------------------------------------------------------
create table if not exists public.users (
    id              uuid primary key references auth.users(id) on delete cascade,
    email           text,
    full_name       text,
    avatar_url      text,
    -- the client this user last worked in (UI convenience)
    default_client_id uuid references public.clients(id) on delete set null,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now(),
    -- profile metadata (never credentials)
    metadata        jsonb not null default '{}'::jsonb
);

create index if not exists idx_users_email on public.users (email);


-- ---------------------------------------------------------------------------
-- 4. Auto-create a public.users row when a new auth.users row appears
--    (sign-up trigger — keeps the two tables in sync forever)
-- ---------------------------------------------------------------------------
create or replace function public.handle_new_auth_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.users (id, email, full_name, avatar_url, metadata)
    values (
        new.id,
        new.email,
        coalesce(
            new.raw_user_meta_data->>'full_name',
            new.raw_user_meta_data->>'name',
            ''
        ),
        new.raw_user_meta_data->>'avatar_url',
        coalesce(new.raw_user_meta_data, '{}'::jsonb)
    )
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_auth_user();


-- ---------------------------------------------------------------------------
-- 5. Row Level Security — the authoritative gate
--
-- Rule: a user can read/write rows belonging to clients they are a member of.
-- We express that once, as a helper, and reuse it.
-- ---------------------------------------------------------------------------
create or replace function public.is_client_member(cid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists (
        select 1 from public.client_members
        where client_id = cid and user_id = auth.uid()
    );
$$;

create or replace function public.has_client_role(cid uuid, roles public.client_role[])
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists (
        select 1 from public.client_members
        where client_id = cid
          and user_id = auth.uid()
          and role = any(roles)
    );
$$;


-- clients: members can read; owners/admins can update; only owners can delete
alter table public.clients enable row level security;

drop policy if exists clients_select on public.clients;
create policy clients_select on public.clients
    for select using (public.is_client_member(id));

drop policy if exists clients_insert on public.clients;
create policy clients_insert on public.clients
    for insert with check (auth.uid() = created_by);

drop policy if exists clients_update on public.clients;
create policy clients_update on public.clients
    for update using (public.has_client_role(id, array['owner','admin']::public.client_role[]));

drop policy if exists clients_delete on public.clients;
create policy clients_delete on public.clients
    for delete using (public.has_client_role(id, array['owner']::public.client_role[]));


-- client_members: members can see the roster; owners/admins can modify
alter table public.client_members enable row level security;

drop policy if exists members_select on public.client_members;
create policy members_select on public.client_members
    for select using (public.is_client_member(client_id));

drop policy if exists members_insert on public.client_members;
create policy members_insert on public.client_members
    for insert with check (public.has_client_role(client_id, array['owner','admin']::public.client_role[]));

drop policy if exists members_update on public.client_members;
create policy members_update on public.client_members
    for update using (public.has_client_role(client_id, array['owner','admin']::public.client_role[]));

drop policy if exists members_delete on public.client_members;
create policy members_delete on public.client_members
    for delete using (public.has_client_role(client_id, array['owner','admin']::public.client_role[]));


-- users: you can read/update your own profile; you can read other members
-- of clients you belong to
alter table public.users enable row level security;

drop policy if exists users_self_select on public.users;
create policy users_self_select on public.users
    for select using (
        auth.uid() = id
        or exists (
            select 1
            from public.client_members me
            join public.client_members them on them.client_id = me.client_id
            where me.user_id = auth.uid() and them.user_id = public.users.id
        )
    );

drop policy if exists users_self_update on public.users;
create policy users_self_update on public.users
    for update using (auth.uid() = id);


-- ---------------------------------------------------------------------------
-- 6. Retrofit existing tables with client_id (auth-aware)
--
--    Note: the existing tables use a text client_id ('default').
--    We keep that for backward compatibility and add a nullable uuid FK.
--    Migration to full uuid client_id is a separate milestone.
-- ---------------------------------------------------------------------------
alter table public.run_history
    add column if not exists client_uuid uuid references public.clients(id) on delete set null;

alter table public.tracked_sources
    add column if not exists client_uuid uuid references public.clients(id) on delete set null;

create index if not exists idx_run_history_client_uuid on public.run_history (client_uuid);
create index if not exists idx_tracked_sources_client_uuid on public.tracked_sources (client_uuid);


-- ---------------------------------------------------------------------------
-- 7. updated_at trigger for users
-- ---------------------------------------------------------------------------
create or replace function public.touch_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists users_touch_updated_at on public.users;
create trigger users_touch_updated_at
    before update on public.users
    for each row execute function public.touch_updated_at();