-- ============================================================================
-- Migration 002 — auto-create a personal client for each new auth user
--
-- Rationale: the previous version of handle_new_auth_user() only created
-- the public.users row. That leaves a brand-new user with zero clients,
-- so they can't do anything until they manually create one. Real products
-- give every user a personal workspace on sign-up; additional clients are
-- opt-in.
--
-- The function runs SECURITY DEFINER, which is what lets it insert into
-- clients/client_members despite the RLS chicken-and-egg (a user can't be
-- added to a client they're not already a member of).
--
-- Safe to run on an existing database: it replaces the function and the
-- trigger only.
-- ============================================================================


create or replace function public.handle_new_auth_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
    new_client_id  uuid;
    slug_base      text;
    slug_final     text;
    display_name   text;
begin
    display_name := coalesce(
        nullif(new.raw_user_meta_data->>'full_name', ''),
        nullif(new.raw_user_meta_data->>'name', ''),
        split_part(new.email, '@', 1)
    );

    -- 1. Mirror the auth user into public.users (as before)
    insert into public.users (id, email, full_name, avatar_url, metadata)
    values (
        new.id,
        new.email,
        display_name,
        new.raw_user_meta_data->>'avatar_url',
        coalesce(new.raw_user_meta_data, '{}'::jsonb)
    )
    on conflict (id) do nothing;

    -- 2. Create the user's personal workspace (client)
    slug_base := 'workspace-' || substr(replace(new.id::text, '-', ''), 1, 10);
    slug_final := slug_base;
    while exists (select 1 from public.clients where slug = slug_final) loop
        slug_final := slug_base || '-' || floor(random() * 100000)::text;
    end loop;

    insert into public.clients (name, slug, created_by, metadata)
    values (
        display_name || '''s Workspace',
        slug_final,
        new.id,
        jsonb_build_object('personal', true)
    )
    returning id into new_client_id;

    -- 3. Make the user an owner of that workspace
    insert into public.client_members (client_id, user_id, role)
    values (new_client_id, new.id, 'owner')
    on conflict (client_id, user_id) do nothing;

    -- 4. Set it as the user's default
    update public.users
    set default_client_id = new_client_id
    where id = new.id;

    return new;
end;
$$;


-- Trigger already exists; make sure it fires our updated function.
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_auth_user();