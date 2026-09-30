-- ============================================================================
-- Migration 012 — align batch state constraints with Python enum values
--
-- Python serializes BatchState / ChunkState using lowercase values:
-- queued, running, paused, completed, failed, cancelled
-- and:
-- queued, running, succeeded, failed, paused, cancelled
-- ============================================================================

alter table public.batches
    drop constraint if exists batches_state_check;

alter table public.batches
    add constraint batches_state_check
    check (
        state in (
            'queued',
            'running',
            'paused',
            'completed',
            'failed',
            'cancelled'
        )
    );


alter table public.batch_chunks
    drop constraint if exists batch_chunks_state_check;

alter table public.batch_chunks
    add constraint batch_chunks_state_check
    check (
        state in (
            'queued',
            'running',
            'succeeded',
            'failed',
            'paused',
            'cancelled'
        )
    );