-- ============================================================================
-- Migration 009 — severity escalation in incident_open_or_touch
--
-- Bug: the original RPC did
--     severity = coalesce(p_incident->>'severity', severity)
-- which *replaces* severity unconditionally. Re-touching an open CRITICAL
-- incident with a later WARNING event silently downgraded it.
--
-- Fix: only escalate. Severity is compared on a numeric rank, not the
-- text value ("critical" < "info" < "warning" alphabetically — useless
-- for ordering).
--
-- Safe to run on an existing database — this replaces the function only.
-- ============================================================================

create or replace function public.incident_open_or_touch(p_incident jsonb)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare
    v_id            uuid;
    v_dedup         text := coalesce(p_incident->>'dedup_key', '');
    v_client        text := coalesce(p_incident->>'client_id', 'default');
    v_incoming_sev  text := coalesce(p_incident->>'severity', 'warning');
    v_incoming_rank integer;
    v_existing_rank integer;
    v_next_sev      text;
begin
    -- Numeric rank for incoming severity
    v_incoming_rank := case v_incoming_sev
        when 'critical' then 2
        when 'warning'  then 1
        when 'info'     then 0
        else 1
    end;

    if v_dedup <> '' then
        select incident_id,
               case severity
                   when 'critical' then 2
                   when 'warning'  then 1
                   when 'info'     then 0
                   else 1
               end
          into v_id, v_existing_rank
          from public.incidents
         where client_id = v_client
           and dedup_key = v_dedup
           and status not in ('resolved','closed')
         limit 1;

        if v_id is not null then
            -- Escalate only. Never downgrade an open incident.
            if v_incoming_rank > v_existing_rank then
                v_next_sev := v_incoming_sev;
            else
                v_next_sev := null;   -- coalesce keeps existing
            end if;

            update public.incidents
               set occurrence_count = occurrence_count + 1,
                   updated_at = now(),
                   severity = coalesce(v_next_sev, severity),
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
        v_incoming_sev,
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